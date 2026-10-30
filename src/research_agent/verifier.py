"""Deterministic claim verification."""

from __future__ import annotations

import html
import re
from typing import Optional, Protocol

from .schemas import Claim, Evidence, RejectedClaim, VerifiedFinding

CRITICAL_CHECKS: tuple[str, ...] = (
    "evidence_id_existence",
    "quote_verification",
    "numeric_consistency",
    "entity_presence",
    "negation_check",
)


class EntailmentChecker(Protocol):
    """Optional interface for model-based or NLI-based entailment checking."""

    async def check_entailment(self, premise: str, hypothesis: str) -> tuple[bool, float]:
        """Check if premise entails hypothesis."""
        ...


def verify_evidence_ids(claim: Claim, evidence_map: dict[str, Evidence]) -> tuple[bool, list[str]]:
    """Check that all cited evidence IDs exist."""
    if not claim.evidence_ids:
        return False, ["No evidence IDs cited in claim"]

    missing = [eid for eid in claim.evidence_ids if eid not in evidence_map]
    passed = len(missing) == 0
    issues = [f"Evidence ID '{eid}' not found" for eid in missing]
    return passed, issues


# Typographic variants models emit (and some pages use) for plain ASCII punctuation.
# Without this, a quote copied with a non-breaking hyphen (U+2011) or curly quotes
# fails to match a source that uses '-' and '"', and a true claim is rejected.
_PUNCT_MAP = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', " ": " ", " ": " ", " ": " ",
    "…": "...",
})


def _normalize_for_match(text: str) -> str:
    """Decode HTML entities, fold typographic punctuation, lowercase, collapse whitespace.

    Only punctuation and spacing are folded: words and numbers must still match exactly.
    """
    return " ".join(html.unescape(text).translate(_PUNCT_MAP).lower().split())


def verify_quotes(claim: Claim, evidence_map: dict[str, Evidence]) -> tuple[bool, list[str]]:
    """Check that each verbatim quote exists in its cited evidence text."""
    if not claim.verbatim_quotes:
        return False, ["No verbatim quotes provided for claim"]

    issues: list[str] = []
    all_found = True

    for quote in claim.verbatim_quotes:
        normalized_quote = _normalize_for_match(quote)
        if not normalized_quote:
            continue

        found_in_any = False
        for eid in claim.evidence_ids:
            if eid in evidence_map:
                ev_text = _normalize_for_match(evidence_map[eid].text)
                if normalized_quote in ev_text or _elided_quote_in(normalized_quote, ev_text):
                    found_in_any = True
                    break

        if not found_in_any:
            all_found = False
            issues.append(
                f"Quote not found in cited evidence: '{quote[:80]}...'"
                if len(quote) > 80
                else f"Quote not found in cited evidence: '{quote}'"
            )

    return all_found, issues


def _elided_quote_in(quote: str, text: str, min_words: int = 4) -> bool:
    """Accept 'A ... B' only if every fragment is a real span of the source, in order.

    Each fragment must be at least min_words words, so an ellipsis cannot be used
    to stitch a claim together from scattered single words.
    """
    if "..." not in quote:
        return False
    fragments = [f.strip(" ,;:-") for f in quote.split("...")]
    fragments = [f for f in fragments if f]
    if len(fragments) < 2 or any(len(f.split()) < min_words for f in fragments):
        return False
    pos = 0
    for frag in fragments:
        idx = text.find(frag, pos)
        if idx < 0:
            return False
        pos = idx + len(frag)
    return True


def _normalize_number(text: str) -> Optional[float]:
    """Attempt to extract and normalize a number from text."""
    text = text.strip().lower()

    m = re.match(r"^([\d,.]+)\s*%", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    m = re.match(r"^([\d,.]+)\s*x$", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    m = re.match(r"^([\d,.]+)\s*([kmbt])$", text)
    if m:
        multipliers = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
        try:
            return float(m.group(1).replace(",", "")) * multipliers.get(m.group(2), 1.0)
        except ValueError:
            pass

    m = re.match(r"^(\d+)\s*/\s*(\d+)$", text)
    if m:
        try:
            denom = int(m.group(2))
            if denom != 0:
                return int(m.group(1)) / denom
        except (ValueError, ZeroDivisionError):
            pass

    m = re.match(r"^[\d,.]+$", text)
    if m:
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            pass

    return None


def verify_numeric_consistency(claim: Claim, evidence_map: dict[str, Evidence]) -> tuple[bool, list[str]]:
    """Check numeric assertions against evidence text."""
    if not claim.numeric_assertions:
        return True, []

    issues: list[str] = []
    all_consistent = True

    cited_text = " ".join(
        evidence_map[eid].text for eid in claim.evidence_ids if eid in evidence_map
    )
    evidence_numbers = re.findall(r"[\d,.]+[%xkmbt]?(?:\s*/\s*\d+)?", cited_text.lower())

    for assertion in claim.numeric_assertions:
        claim_numbers = re.findall(r"[\d,.]+[%xkmbt]?(?:\s*/\s*\d+)?", assertion.lower())

        for num_str in claim_numbers:
            claim_num = _normalize_number(num_str)
            if claim_num is None:
                continue

            found_match = False
            for ev_num_str in evidence_numbers:
                ev_num = _normalize_number(ev_num_str)
                if ev_num is not None:
                    if claim_num == 0 and ev_num == 0:
                        found_match = True
                        break
                    elif claim_num != 0:
                        ratio = abs(ev_num - claim_num) / abs(claim_num)
                        if ratio < 0.01:
                            found_match = True
                            break

            if not found_match:
                all_consistent = False
                issues.append(f"Numeric assertion '{assertion}' not supported by cited evidence")
                break

    return all_consistent, issues


def verify_entity_presence(claim: Claim, evidence_map: dict[str, Evidence]) -> tuple[bool, list[str]]:
    """Check that each relevant entity is grounded in the cited evidence.

    Name-like tokens (any capital letter or digit: "Qdrant", "LLM-based", "2.6")
    must appear exactly; that is what rejects invented products and versions.
    Generic words ("benchmark", "long-term memory") match on word stems, so a topic
    phrase worded differently from the source is not a false rejection.
    """
    if not claim.relevant_entities:
        return True, []

    issues: list[str] = []
    cited_text = _normalize_for_match(
        " ".join(evidence_map[eid].text for eid in claim.evidence_ids if eid in evidence_map)
    )
    for entity in claim.relevant_entities:
        if not _entity_in_text(entity, cited_text):
            issues.append(f"Entity '{entity}' not found in cited evidence")
    return len(issues) == 0, issues


def _entity_in_text(entity: str, text: str) -> bool:
    folded = _normalize_for_match(entity)
    if not folded or folded in text:
        return bool(folded)
    raw_tokens = entity.translate(_PUNCT_MAP).split()
    named = [t for t in raw_tokens if any(c.isupper() or c.isdigit() for c in t)]
    generic = [t for t in raw_tokens if t not in named]

    for tok in named:
        if _normalize_for_match(tok).strip(".,;:()") not in text:
            return False

    stems = [w[:5] for t in generic for w in re.split(r"[-/]", t.lower()) if len(w) >= 3]
    if not stems:
        return bool(named)
    found = sum(1 for st in stems if st in text)
    return found / len(stems) >= 2 / 3


def check_negation_flips(claim: Claim, evidence_map: dict[str, Evidence]) -> tuple[bool, list[str]]:
    """Heuristic check for negation flips between claim and evidence."""
    negation_words = {
        "not", "no", "never", "neither", "none", "nor", "n't", "cannot",
        "can't", "won't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    }

    claim_words = set(re.findall(r"\b[\w']+\b", claim.claim_text.lower()))
    claim_has_negation = bool(claim_words & negation_words)

    issues: list[str] = []
    for quote in claim.verbatim_quotes:
        quote_words = set(re.findall(r"\b[\w']+\b", quote.lower()))
        quote_has_negation = bool(quote_words & negation_words)

        if claim_has_negation != quote_has_negation:
            issues.append(f"Potential negation flip detected between claim and quote: '{quote[:60]}'")

    return len(issues) == 0, issues


def verify_claim(
    claim: Claim,
    evidence_map: dict[str, Evidence],
    entailment_checker: Optional[EntailmentChecker] = None,
) -> tuple[bool, VerifiedFinding | RejectedClaim, list[str]]:
    """Run all verification checks on a claim."""
    checks_passed: list[str] = []
    all_issues: list[str] = []
    checks_run: list[str] = []

    checks_run.append("evidence_id_existence")
    passed, issues = verify_evidence_ids(claim, evidence_map)
    if passed:
        checks_passed.append("evidence_id_existence")
    all_issues.extend(issues)

    if not passed:
        claim.verification_status = "rejected"
        claim.rejection_reason = "; ".join(all_issues)
        return False, RejectedClaim(
            claim=claim,
            rejection_reasons=all_issues,
            failed_checks=["evidence_id_existence"],
        ), checks_run

    checks_run.append("quote_verification")
    passed, issues = verify_quotes(claim, evidence_map)
    if passed:
        checks_passed.append("quote_verification")
    all_issues.extend(issues)

    checks_run.append("numeric_consistency")
    passed, issues = verify_numeric_consistency(claim, evidence_map)
    if passed:
        checks_passed.append("numeric_consistency")
    all_issues.extend(issues)

    checks_run.append("entity_presence")
    passed, issues = verify_entity_presence(claim, evidence_map)
    if passed:
        checks_passed.append("entity_presence")
    all_issues.extend(issues)

    checks_run.append("negation_check")
    passed, issues = check_negation_flips(claim, evidence_map)
    if passed:
        checks_passed.append("negation_check")
    all_issues.extend(issues)

    critical_checks = set(CRITICAL_CHECKS)
    critical_passed = critical_checks.issubset(set(checks_passed))

    if critical_passed:
        claim.verification_status = "verified"
        supporting = [
            evidence_map[eid] for eid in claim.evidence_ids if eid in evidence_map
        ]
        finding = VerifiedFinding(
            claim=claim,
            supporting_evidence=supporting,
            verification_checks_passed=checks_passed,
        )
        return True, finding, checks_run
    else:
        claim.verification_status = "rejected"
        claim.rejection_reason = "; ".join(all_issues)
        failed_checks = [c for c in checks_run if c not in checks_passed]
        return False, RejectedClaim(
            claim=claim,
            rejection_reasons=all_issues,
            failed_checks=failed_checks,
        ), checks_run


def verify_all_claims(
    claims: list[Claim],
    evidence_map: dict[str, Evidence],
) -> tuple[list[VerifiedFinding], list[RejectedClaim]]:
    """Verify all claims and separate into verified findings and rejected claims."""
    verified: list[VerifiedFinding] = []
    rejected: list[RejectedClaim] = []

    for claim in claims:
        is_verified, result, _ = verify_claim(claim, evidence_map)
        if is_verified:
            assert isinstance(result, VerifiedFinding)
            verified.append(result)
        else:
            assert isinstance(result, RejectedClaim)
            rejected.append(result)

    return verified, rejected


__all__ = [
    "CRITICAL_CHECKS",
    "EntailmentChecker",
    "verify_evidence_ids",
    "verify_quotes",
    "verify_numeric_consistency",
    "verify_entity_presence",
    "check_negation_flips",
    "verify_claim",
    "verify_all_claims",
]
