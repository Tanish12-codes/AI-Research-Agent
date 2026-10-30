"""Tests for evidence filtering."""
import pytest
from datetime import datetime, timezone, timedelta

from research_agent.filtering import (
    normalize_url, content_hash, is_near_duplicate,
    compute_lexical_relevance, compute_authority_score,
    compute_recency_weight, filter_and_rank_evidence,
    check_coverage,
)
from research_agent.schemas import Evidence


def _make_evidence(eid, url="https://example.com", text="sample", domain="example.com", **kwargs) -> Evidence:
    return Evidence(
        evidence_id=eid, url=url, domain=domain, title="Test",
        text=text, retrieval_timestamp=datetime.now(timezone.utc),
        provider="test", **kwargs,
    )


class TestURLNormalization:
    def test_remove_fragment(self):
        result = normalize_url('https://example.com/page#section')
        assert '#' not in result
    
    def test_remove_tracking_params(self):
        result = normalize_url('https://example.com/page?utm_source=twitter&id=1')
        assert 'utm_source' not in result
        assert 'id=1' in result
    
    def test_lowercase_host(self):
        result = normalize_url('https://EXAMPLE.COM/Page')
        assert 'example.com' in result
    
    def test_strip_trailing_slash(self):
        r1 = normalize_url('https://example.com/page/')
        r2 = normalize_url('https://example.com/page')
        assert r1 == r2


class TestContentHash:
    def test_deterministic(self):
        h1 = content_hash('Hello World')
        h2 = content_hash('Hello World')
        assert h1 == h2
    
    def test_case_insensitive(self):
        h1 = content_hash('Hello World')
        h2 = content_hash('hello world')
        assert h1 == h2
    
    def test_whitespace_normalized(self):
        h1 = content_hash('Hello  World')
        h2 = content_hash('Hello World')
        assert h1 == h2


class TestDuplicateDetection:
    def test_exact_duplicate(self):
        ev1 = _make_evidence('e1', text='Exactly the same text')
        ev2 = _make_evidence('e2', text='Exactly the same text')
        assert is_near_duplicate(ev2, [ev1]) is True
    
    def test_near_duplicate(self):
        ev1 = _make_evidence('e1', text='Python is a great programming language for developers')
        ev2 = _make_evidence('e2', text='Python is a great programming language for all developers')
        assert is_near_duplicate(ev2, [ev1], similarity_threshold=0.8) is True
    
    def test_not_duplicate(self):
        ev1 = _make_evidence('e1', text='Python is a programming language')
        ev2 = _make_evidence('e2', text='JavaScript runs in the browser for web development')
        assert is_near_duplicate(ev2, [ev1]) is False


class TestRelevanceScoring:
    def test_high_relevance(self):
        score = compute_lexical_relevance(
            'Python is a programming language created by Guido van Rossum',
            'Python programming language',
            ['Python'],
        )
        assert score > 0.3
    
    def test_low_relevance(self):
        score = compute_lexical_relevance(
            'Today the weather is sunny and warm',
            'Python programming language',
            ['Python'],
        )
        assert score < 0.3


class TestAuthorityScore:
    def test_high_authority(self):
        score = compute_authority_score('github.com')
        assert score >= 0.8
    
    def test_medium_authority(self):
        score = compute_authority_score('medium.com')
        assert 0.4 < score < 0.9
    
    def test_unknown_domain(self):
        score = compute_authority_score('randomsite.xyz')
        assert 0.3 <= score <= 0.6


class TestRecencyWeight:
    def test_recent_high_weight(self):
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        weight = compute_recency_weight(recent)
        assert weight >= 0.9
    
    def test_old_low_weight(self):
        old = datetime.now(timezone.utc) - timedelta(days=500)
        weight = compute_recency_weight(old)
        assert weight < 0.5
    
    def test_none_date_neutral(self):
        weight = compute_recency_weight(None)
        assert weight == 0.5


class TestCoverage:
    def test_full_coverage(self, sample_evidence):
        ratio, covered, missing = check_coverage(
            sample_evidence, ['Python'], ['programming']
        )
        assert ratio == 1.0
        assert len(missing) == 0
    
    def test_partial_coverage(self, sample_evidence):
        ratio, covered, missing = check_coverage(
            sample_evidence, ['Python', 'Rust'], ['programming']
        )
        assert ratio < 1.0
        assert 'Rust' in missing
    
    def test_empty_requirements(self, sample_evidence):
        ratio, _, _ = check_coverage(sample_evidence, [], [])
        assert ratio == 1.0


class TestFilterPipeline:
    def test_deduplication(self):
        evs = [
            _make_evidence('e1', text='Same content here'),
            _make_evidence('e2', text='Same content here'),
            _make_evidence('e3', text='Different content entirely'),
        ]
        result = filter_and_rank_evidence(evs, 'content', [])
        assert len(result) <= 2
    
    def test_quarantine_excluded(self):
        evs = [
            _make_evidence('e1', text='Good content'),
            _make_evidence('e2', text='Bad content'),
        ]
        evs[1].is_quarantined = True
        result = filter_and_rank_evidence(evs, 'content', [])
        assert all(not ev.is_quarantined for ev in result)
    
    def test_budget_enforced(self):
        evs = [_make_evidence(f'e{i}', text=f'Unique content number {i}') for i in range(20)]
        result = filter_and_rank_evidence(evs, 'content', [], max_evidence=5)
        assert len(result) <= 5
