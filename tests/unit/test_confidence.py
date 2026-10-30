"""Tests for confidence scoring."""
import pytest
from datetime import datetime, timezone

from research_agent.confidence import compute_confidence, should_abstain
from research_agent.schemas import (
    Evidence, Claim, VerifiedFinding, RejectedClaim, QuestionType,
)


def _make_finding(claim_text='test', evidence_text='test evidence', authority=0.8):
    ev = Evidence(
        evidence_id='ev-1', url='https://example.com', domain='example.com',
        title='Test', text=evidence_text,
        retrieval_timestamp=datetime.now(timezone.utc),
        provider='test', authority_score=authority,
    )
    claim = Claim(
        claim_id='c1', claim_text=claim_text,
        evidence_ids=['ev-1'], verbatim_quotes=['test'],
    )
    return VerifiedFinding(
        claim=claim, supporting_evidence=[ev],
        verification_checks_passed=['evidence_id_existence', 'quote_verification'],
    )


class TestConfidence:
    def test_zero_with_no_findings(self):
        score, _ = compute_confidence([], [], [], 0.0, QuestionType.FACTUAL, [])
        assert score == 0.0
    
    def test_positive_with_findings(self):
        finding = _make_finding()
        score, explanation = compute_confidence(
            [finding], [], [finding.supporting_evidence[0]],
            1.0, QuestionType.FACTUAL, [],
        )
        assert score > 0.0
        assert 'heuristic' in explanation.lower()
    
    def test_comparison_type_caps_on_missing_entity(self):
        finding = _make_finding(evidence_text='Python is great')
        score, _ = compute_confidence(
            [finding], [], [finding.supporting_evidence[0]],
            0.5, QuestionType.COMPARISON, ['Python', 'Rust'],
        )
        assert score <= 0.5


class TestAbstention:
    def test_abstain_no_findings(self):
        should, reason = should_abstain(0.0, 0.3, 0, 0)
        assert should is True
    
    def test_abstain_low_confidence(self):
        should, reason = should_abstain(0.1, 0.3, 1, 5)
        assert should is True
    
    def test_no_abstain_good_confidence(self):
        should, reason = should_abstain(0.5, 0.3, 3, 10)
        assert should is False
