"""Tests for claim verification."""
import pytest
from datetime import datetime, timezone

from research_agent.verifier import (
    verify_evidence_ids, verify_quotes, verify_numeric_consistency,
    verify_entity_presence, check_negation_flips, verify_claim,
    verify_all_claims, _normalize_number,
)
from research_agent.schemas import Evidence, Claim, VerifiedFinding, RejectedClaim


def _make_evidence_map():
    return {
        'ev-001': Evidence(
            evidence_id='ev-001',
            url='https://example.com',
            domain='example.com',
            title='Test',
            text='Python has over 100,000 packages on PyPI. It was created by Guido van Rossum in 1991.',
            retrieval_timestamp=datetime.now(timezone.utc),
            provider='test',
        ),
        'ev-002': Evidence(
            evidence_id='ev-002',
            url='https://github.com/test',
            domain='github.com',
            title='Test Repo',
            text='Repository has 58000 stars. Last updated 2024.',
            retrieval_timestamp=datetime.now(timezone.utc),
            provider='github',
        ),
    }


class TestEvidenceIDVerification:
    def test_valid_ids(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1', claim_text='test',
            evidence_ids=['ev-001'], verbatim_quotes=['test'],
        )
        passed, issues = verify_evidence_ids(claim, evidence_map)
        assert passed is True
        assert len(issues) == 0
    
    def test_missing_id(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1', claim_text='test',
            evidence_ids=['ev-999'], verbatim_quotes=['test'],
        )
        passed, issues = verify_evidence_ids(claim, evidence_map)
        assert passed is False
        assert len(issues) > 0


class TestQuoteVerification:
    def test_exact_quote_found(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1', claim_text='test',
            evidence_ids=['ev-001'],
            verbatim_quotes=['created by Guido van Rossum'],
        )
        passed, issues = verify_quotes(claim, evidence_map)
        assert passed is True
    
    def test_quote_not_found(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1', claim_text='test',
            evidence_ids=['ev-001'],
            verbatim_quotes=['this quote does not exist in the evidence'],
        )
        passed, issues = verify_quotes(claim, evidence_map)
        assert passed is False
    
    def test_case_insensitive_match(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1', claim_text='test',
            evidence_ids=['ev-001'],
            verbatim_quotes=['CREATED BY GUIDO VAN ROSSUM'],
        )
        passed, issues = verify_quotes(claim, evidence_map)
        assert passed is True


class TestNumericConsistency:
    def test_matching_number(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1', claim_text='100000 packages',
            evidence_ids=['ev-001'],
            verbatim_quotes=['100,000 packages'],
            numeric_assertions=['100,000 packages'],
        )
        passed, issues = verify_numeric_consistency(claim, evidence_map)
        assert passed is True
    
    def test_mismatched_number(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1', claim_text='200000 packages',
            evidence_ids=['ev-001'],
            verbatim_quotes=['packages'],
            numeric_assertions=['200,000 packages'],
        )
        passed, issues = verify_numeric_consistency(claim, evidence_map)
        assert passed is False
    
    def test_no_numeric_assertions(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1', claim_text='test',
            evidence_ids=['ev-001'],
            verbatim_quotes=['test'],
            numeric_assertions=[],
        )
        passed, issues = verify_numeric_consistency(claim, evidence_map)
        assert passed is True


class TestNormalizeNumber:
    def test_percentage(self):
        assert _normalize_number('50%') == 50.0
    
    def test_multiplier(self):
        assert _normalize_number('3x') == 3.0
    
    def test_shorthand_k(self):
        assert _normalize_number('58k') == 58000.0
    
    def test_shorthand_m(self):
        assert _normalize_number('2.5m') == 2500000.0
    
    def test_fraction(self):
        assert _normalize_number('3/4') == 0.75
    
    def test_plain_number(self):
        assert _normalize_number('42') == 42.0
    
    def test_comma_number(self):
        assert _normalize_number('100,000') == 100000.0


class TestNegationFlips:
    def test_no_negation(self):
        claim = Claim(
            claim_id='c1',
            claim_text='Python is popular',
            evidence_ids=['ev-001'],
            verbatim_quotes=['Python is popular'],
        )
        passed, issues = check_negation_flips(claim, {})
        assert passed is True
    
    def test_negation_flip(self):
        claim = Claim(
            claim_id='c1',
            claim_text='Python is not popular',
            evidence_ids=['ev-001'],
            verbatim_quotes=['Python is popular'],
        )
        passed, issues = check_negation_flips(claim, {})
        assert passed is False
        assert 'negation' in issues[0].lower()


class TestFullVerification:
    def test_verified_claim(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1',
            claim_text='Python was created by Guido van Rossum in 1991',
            evidence_ids=['ev-001'],
            verbatim_quotes=['created by Guido van Rossum in 1991'],
            relevant_entities=['Python', 'Guido van Rossum'],
        )
        is_verified, result, checks = verify_claim(claim, evidence_map)
        assert is_verified is True
        assert isinstance(result, VerifiedFinding)
    
    def test_rejected_claim_missing_evidence(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1',
            claim_text='test',
            evidence_ids=['ev-nonexistent'],
            verbatim_quotes=['test'],
        )
        is_verified, result, checks = verify_claim(claim, evidence_map)
        assert is_verified is False
        assert isinstance(result, RejectedClaim)
    
    def test_rejected_claim_bad_quote(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1',
            claim_text='Python has 500K stars',
            evidence_ids=['ev-001'],
            verbatim_quotes=['500K stars on GitHub'],
        )
        is_verified, result, checks = verify_claim(claim, evidence_map)
        assert is_verified is False
        assert isinstance(result, RejectedClaim)

    def test_verify_all_claims(self):
        evidence_map = _make_evidence_map()
        claims = [
            Claim(
                claim_id='c1',
                claim_text='Created by Guido',
                evidence_ids=['ev-001'],
                verbatim_quotes=['created by Guido van Rossum'],
            ),
            Claim(
                claim_id='c2',
                claim_text='Fabricated fact',
                evidence_ids=['ev-999'],
                verbatim_quotes=['nonexistent quote'],
            ),
        ]
        verified, rejected = verify_all_claims(claims, evidence_map)
        assert len(verified) == 1
        assert len(rejected) == 1


class TestNonQuoteChecksAreCritical:
    """A real quote must not launder a claim that fails another check."""

    def test_negation_flip_rejected(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1',
            claim_text='Python was not created by Guido van Rossum.',
            evidence_ids=['ev-001'],
            verbatim_quotes=['created by Guido van Rossum in 1991'],
        )
        is_verified, result, _ = verify_claim(claim, evidence_map)
        assert is_verified is False
        assert 'negation_check' in result.failed_checks

    def test_fabricated_number_rejected(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1',
            claim_text='Python has over 900,000 packages on PyPI.',
            evidence_ids=['ev-001'],
            verbatim_quotes=['Python has over 100,000 packages on PyPI'],
            numeric_assertions=['900,000 packages'],
        )
        is_verified, result, _ = verify_claim(claim, evidence_map)
        assert is_verified is False
        assert 'numeric_consistency' in result.failed_checks

    def test_unsupported_entity_rejected(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1',
            claim_text='Python has more packages than FastAPI.',
            evidence_ids=['ev-001'],
            verbatim_quotes=['Python has over 100,000 packages on PyPI'],
            relevant_entities=['Python', 'FastAPI'],
        )
        is_verified, result, _ = verify_claim(claim, evidence_map)
        assert is_verified is False
        assert 'entity_presence' in result.failed_checks

    def test_all_checks_recorded_on_verified_finding(self):
        evidence_map = _make_evidence_map()
        claim = Claim(
            claim_id='c1',
            claim_text='Python has over 100,000 packages on PyPI.',
            evidence_ids=['ev-001'],
            verbatim_quotes=['Python has over 100,000 packages on PyPI'],
            numeric_assertions=['100,000 packages'],
            relevant_entities=['Python', 'PyPI'],
        )
        is_verified, result, _ = verify_claim(claim, evidence_map)
        assert is_verified is True
        assert len(result.verification_checks_passed) == 5


class TestQuoteEntityNormalization:
    def test_entity_encoded_quote_matches_plain_evidence(self):
        evidence_map = {
            'ev-q': Evidence(
                evidence_id='ev-q',
                url='https://example.com/gil',
                domain='example.com',
                title='GIL',
                text="Python's GIL was made optional in 3.13 & later.",
                retrieval_timestamp=datetime.now(timezone.utc),
                provider='test',
            )
        }
        claim = Claim(
            claim_id='c1',
            claim_text='The GIL is optional from Python 3.13.',
            evidence_ids=['ev-q'],
            verbatim_quotes=['Python&#x27;s GIL was made optional in 3.13 &amp; later'],
        )
        passed, issues = verify_quotes(claim, evidence_map)
        assert passed is True, issues


class TestEntityConceptPhrases:
    """Concept phrases match on word stems; named entities still need an exact match."""

    def _claim(self, entities):
        from research_agent.schemas import Claim
        return Claim(claim_id="c", claim_text="t", evidence_ids=["e1"], verbatim_quotes=["x"],
                     relevant_entities=entities)

    def _evidence(self, text):
        from datetime import datetime, timezone
        from research_agent.schemas import Evidence
        return {"e1": Evidence(evidence_id="e1", url="https://x.com", domain="x.com", title="t", text=text,
                               retrieval_timestamp=datetime.now(timezone.utc), provider="mock")}

    def test_reworded_concept_phrase_passes(self):
        from research_agent.verifier import verify_entity_presence
        ev = self._evidence("Multi-agent systems split customer support work across specialized agents.")
        ok, issues = verify_entity_presence(self._claim(["multi-agent architecture", "customer support"]), ev)
        assert ok, issues

    def test_unrelated_concept_phrase_fails(self):
        from research_agent.verifier import verify_entity_presence
        ev = self._evidence("Vector databases store embeddings.")
        ok, _ = verify_entity_presence(self._claim(["customer support automation"]), ev)
        assert not ok

    def test_invented_named_entity_still_rejected(self):
        from research_agent.verifier import verify_entity_presence
        ev = self._evidence("Milvus added tiered storage and quantization.")
        ok, issues = verify_entity_presence(self._claim(["RaBitQ", "Milvus 2.6"]), ev)
        assert not ok
        assert len(issues) == 2


class TestTypographicNormalization:
    def test_quote_with_unicode_hyphens_and_curly_quotes_matches_ascii_source(self):
        from datetime import datetime, timezone
        from research_agent.schemas import Claim, Evidence
        from research_agent.verifier import verify_quotes
        ev = {"e1": Evidence(evidence_id="e1", url="https://x.com", domain="x.com", title="t",
                             text='Long-term memory uses a "vector store" for retrieval.',
                             retrieval_timestamp=datetime.now(timezone.utc), provider="mock")}
        claim = Claim(claim_id="c", claim_text="t", evidence_ids=["e1"],
                      verbatim_quotes=["Long\u2011term memory uses a \u201cvector store\u201d"])
        ok, issues = verify_quotes(claim, ev)
        assert ok, issues

    def test_word_changes_still_fail(self):
        from research_agent.verifier import _normalize_for_match
        assert _normalize_for_match("short\u2011term") != _normalize_for_match("long-term")


class TestElidedQuotes:
    def test_in_order_fragments_pass(self):
        from research_agent.verifier import _elided_quote_in
        text = "rag combines the strengths of traditional information retrieval systems with the generative power of llms"
        assert _elided_quote_in("rag combines the strengths of ... with the generative power of llms", text)

    def test_out_of_order_or_tiny_fragments_fail(self):
        from research_agent.verifier import _elided_quote_in
        text = "rag combines the strengths of traditional retrieval with the generative power of llms"
        assert not _elided_quote_in("with the generative power of ... rag combines the strengths of", text)
        assert not _elided_quote_in("rag ... llms", text)
        assert not _elided_quote_in("no ellipsis here at all", text)


class TestEntityTokens:
    def _check(self, entities, text):
        from datetime import datetime, timezone
        from research_agent.schemas import Claim, Evidence
        from research_agent.verifier import verify_entity_presence
        ev = {"e1": Evidence(evidence_id="e1", url="https://x.com", domain="x.com", title="t", text=text,
                             retrieval_timestamp=datetime.now(timezone.utc), provider="mock")}
        claim = Claim(claim_id="c", claim_text="t", evidence_ids=["e1"], verbatim_quotes=["x"],
                      relevant_entities=entities)
        return verify_entity_presence(claim, ev)[0]

    def test_unicode_hyphen_concept_phrase(self):
        assert self._check(["long‑term memory"], "Agents keep long-term memories in a vector store.")

    def test_named_token_with_generic_word(self):
        assert self._check(["TWIST benchmark"], "TWIST measures intervention quality in benchmarks.")
        assert self._check(["LLM‑based agents"], "Evaluating LLM-based agent memory.")

    def test_named_token_must_be_exact(self):
        assert not self._check(["TWIST benchmark"], "A new benchmark for memory.")
        assert not self._check(["Milvus 2.6"], "Milvus 2.5 added tiered storage.")
