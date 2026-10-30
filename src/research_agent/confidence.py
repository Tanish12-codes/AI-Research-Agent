"""Evidence-based confidence scoring."""

from __future__ import annotations

import re

from .schemas import Evidence, QuestionType, RejectedClaim, VerifiedFinding
from .verifier import _entity_in_text, _normalize_for_match


def compute_confidence(
    verified_findings: list[VerifiedFinding],
    rejected_claims: list[RejectedClaim],
    all_evidence: list[Evidence],
    coverage_ratio: float,
    question_type: QuestionType,
    required_entities: list[str],
) -> tuple[float, str]:
    """Compute confidence score from evidence signals."""
    if not verified_findings and not all_evidence:
        return 0.0, "No evidence or verified findings available."

    if not verified_findings:
        return 0.05, "No claims survived verification."

    explanation_parts: list[str] = []

    total_claims = len(verified_findings) + len(rejected_claims)
    verification_rate = len(verified_findings) / total_claims if total_claims > 0 else 0.0
    explanation_parts.append(
        f"Verification rate: {verification_rate:.0%} ({len(verified_findings)}/{total_claims} claims)"
    )

    explanation_parts.append(f"Coverage ratio: {coverage_ratio:.0%}")

    domains: set[str] = set()
    for f in verified_findings:
        for ev in f.supporting_evidence:
            if ev.domain:
                domains.add(ev.domain.lower())
    diversity_score = min(len(domains) / 3.0, 1.0)
    explanation_parts.append(f"Source diversity: {len(domains)} unique domain(s)")

    authority_scores: list[float] = []
    for f in verified_findings:
        for ev in f.supporting_evidence:
            if ev.authority_score is not None:
                authority_scores.append(ev.authority_score)
    avg_authority = sum(authority_scores) / len(authority_scores) if authority_scores else 0.5
    explanation_parts.append(f"Average authority: {avg_authority:.2f}")

    total_checks = sum(len(f.verification_checks_passed) for f in verified_findings)
    expected_checks = len(verified_findings) * 5
    check_rate = total_checks / expected_checks if expected_checks > 0 else 0.0
    explanation_parts.append(f"Verification check rate: {check_rate:.0%}")

    trust_scores = [ev.trust_level for ev in all_evidence if not ev.is_quarantined]
    avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.5

    base_confidence = (
        0.25 * verification_rate
        + 0.25 * coverage_ratio
        + 0.15 * diversity_score
        + 0.15 * avg_authority
        + 0.10 * check_rate
        + 0.10 * avg_trust
    )

    type_cap, type_explanation = _apply_question_type_rules(
        question_type,
        verified_findings,
        all_evidence,
        coverage_ratio,
        required_entities,
    )

    if type_explanation:
        explanation_parts.append(type_explanation)

    final_confidence = min(base_confidence, type_cap)
    final_confidence = max(min(final_confidence, 1.0), 0.0)

    explanation = (
        f"Confidence: {final_confidence:.2f} (heuristic, not calibrated probability). "
        + "; ".join(explanation_parts)
    )

    return round(final_confidence, 2), explanation


def _apply_question_type_rules(
    question_type: QuestionType,
    verified_findings: list[VerifiedFinding],
    evidence: list[Evidence],
    coverage_ratio: float,
    required_entities: list[str],
) -> tuple[float, str]:
    """Apply question-type-specific confidence caps and rules."""
    all_text = " ".join(ev.text.lower() for ev in evidence if ev.text)
    all_words = set(re.findall(r"[a-z]+", all_text))

    if question_type == QuestionType.COMPARISON:
        if required_entities:
            # Same matcher as the verifier: typographic hyphens and topic phrases
            # worded differently from the source must not count as "missing".
            normalized = _normalize_for_match(all_text)
            covered = sum(1 for e in required_entities if _entity_in_text(e, normalized))
            if covered < len(required_entities):
                missing = len(required_entities) - covered
                cap = 0.5 - (0.1 * missing)
                return max(cap, 0.1), f"Comparison: {missing} entity(ies) missing from evidence"
        return 1.0, "Comparison: all entities covered"

    elif question_type == QuestionType.PROS_CONS:
        pro_words = {
            "advantage", "advantages", "benefit", "benefits", "pro", "pros", "strength",
            "strengths", "good", "better", "positive", "upside", "upsides",
        }
        con_words = {
            "disadvantage", "disadvantages", "drawback", "drawbacks", "con", "cons",
            "weakness", "weaknesses", "bad", "worse", "negative", "limitation",
            "limitations", "downside", "downsides",
        }
        has_pros = bool(all_words & pro_words)
        has_cons = bool(all_words & con_words)
        if not has_pros or not has_cons:
            return 0.5, "Pros/cons: evidence for only one side found"
        return 1.0, "Pros/cons: both sides covered"

    elif question_type == QuestionType.FACTUAL:
        avg_auth = 0.0
        count = 0
        for f in verified_findings:
            for ev in f.supporting_evidence:
                if ev.authority_score is not None:
                    avg_auth += ev.authority_score
                    count += 1
        if count > 0:
            avg_auth /= count
        if avg_auth < 0.5:
            return 0.6, f"Factual: low authority evidence (avg: {avg_auth:.2f})"
        return 1.0, f"Factual: adequate authority (avg: {avg_auth:.2f})"

    elif question_type == QuestionType.HOW_TO:
        if coverage_ratio < 0.5:
            return 0.4, "How-to: insufficient step coverage"
        return 1.0, f"How-to: coverage {coverage_ratio:.0%}"

    elif question_type == QuestionType.DISCOVERY:
        return 1.0, "Discovery: standard evaluation"

    else:
        return 1.0, "Exploratory: standard evaluation"


def should_abstain(
    confidence: float,
    threshold: float,
    verified_count: int,
    evidence_count: int,
) -> tuple[bool, str]:
    """Decide whether to abstain from answering."""
    if verified_count == 0:
        return True, "No verified findings remain after verification."

    if evidence_count == 0:
        return True, "No evidence was found for the research question."

    if confidence < threshold:
        return True, f"Confidence ({confidence:.2f}) is below threshold ({threshold:.2f})."

    return False, ""


__all__ = [
    "compute_confidence",
    "should_abstain",
]
