"""Citation precision and expected-finding recall must measure something real."""

from datetime import datetime, timezone

from research_agent.evaluation.metrics import calculate_run_metrics, expected_finding_hit
from research_agent.schemas import Claim, Evidence, FinalAnswer, VerifiedFinding


def _ev(eid: str, text: str) -> Evidence:
    return Evidence(
        evidence_id=eid, url=f"https://example.com/{eid}", domain="example.com", title=eid, text=text,
        retrieval_timestamp=datetime.now(timezone.utc), provider="mock",
    )


def _answer(findings: list[VerifiedFinding], short: str = "") -> FinalAnswer:
    return FinalAnswer(
        question="q", short_answer=short, key_findings=[], source_links=[],
        verified_findings=findings, confidence_score=0.8, confidence_explanation="",
    )


def _finding(quote: str, evidence: list[Evidence], text: str = "claim") -> VerifiedFinding:
    claim = Claim(
        claim_id="c1", claim_text=text, evidence_ids=[e.evidence_id for e in evidence], verbatim_quotes=[quote],
    )
    return VerifiedFinding(claim=claim, supporting_evidence=evidence, verification_checks_passed=[])


def test_citation_precision_penalizes_overcitation():
    holds = _ev("a", "Qdrant is written in Rust.")
    unrelated = _ev("b", "Milvus is cloud native.")
    answer = _answer([_finding("written in Rust", [holds, unrelated])])
    m = calculate_run_metrics([{"answer": answer}], {})
    assert m.cited_pairs == 2
    assert m.citation_precision == 0.5


def test_citation_precision_is_none_without_citations():
    m = calculate_run_metrics([{"answer": _answer([])}], {})
    assert m.citation_precision is None


def test_expected_finding_recall():
    ev = _ev("a", "Python was created by Guido van Rossum.")
    answer = _answer([_finding("created by Guido van Rossum", [ev], text="Python was created by Guido van Rossum.")])
    m = calculate_run_metrics(
        [{"answer": answer, "expected_findings": ["Created by Guido van Rossum", "Python 3 broke compatibility"]}],
        {},
    )
    assert m.expected_findings_total == 2
    assert m.expected_findings_hit == 1
    assert m.expected_finding_recall == 0.5


def test_expected_finding_hit_ignores_short_words():
    assert expected_finding_hit("Flask is a synchronous WSGI microframework", "Flask: a synchronous WSGI microframework")
    assert not expected_finding_hit("Flask is a synchronous WSGI microframework", "FastAPI is async")
