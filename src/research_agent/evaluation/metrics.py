"""Evaluation metrics computation and reporting."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field

from ..schemas import FinalAnswer
from ..verifier import _normalize_for_match

_STOPWORDS = {
    "with", "from", "that", "this", "have", "which", "their", "they", "into", "than",
    "also", "used", "using", "based", "built", "more", "most", "other", "such",
}


def _content_words(text: str) -> set[str]:
    words = {w.strip(".,;:()[]'\"").lower() for w in text.split()}
    return {w for w in words if len(w) >= 4 and w not in _STOPWORDS}


def expected_finding_hit(expected: str, answer_text: str, threshold: float = 0.5) -> bool:
    """Lexical proxy for 'the answer states this finding': share of the expected
    finding's content words that appear in the answer. Not semantic; see docs/evaluation.md."""
    words = _content_words(expected)
    if not words:
        return False
    answer_words = _content_words(answer_text)
    return len(words & answer_words) / len(words) >= threshold


class EvaluationMetrics(BaseModel):
    """Structured evaluation report containing explicitly calculated metrics."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: str = "mock"
    model_name: str = "unknown"
    search_provider: str = "unknown"

    total_queries: int = 0
    answerable_queries: int = 0
    unanswerable_queries: int = 0

    total_claims: int = 0
    verified_claims: int = 0
    rejected_claims: int = 0

    claim_support_rate: Optional[float] = None

    unsupported_claims_per_answer: float = 0.0

    citation_precision: Optional[float] = None
    cited_pairs: int = 0
    expected_findings_total: int = 0
    expected_findings_hit: int = 0
    expected_finding_recall: Optional[float] = None

    correct_abstention_rate: Optional[float] = None

    abstention_rate_on_answerable: Optional[float] = None

    mean_entity_coverage: float = 0.0

    redteam_total_attacks: int = 0
    redteam_blocked_attacks: int = 0
    redteam_attack_success_rate: float = 0.0

    redteam_benign_controls: int = 0
    redteam_false_positives: int = 0
    redteam_false_positive_rate: float = 0.0

    latencies: list[float] = Field(default_factory=list)
    p50_latency_seconds: Optional[float] = None
    p95_latency_seconds: Optional[float] = None

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: Optional[float] = None

    provider_errors: int = 0
    provider_retries: int = 0

    baseline_total_queries: int = 0
    baseline_p50_latency: Optional[float] = None
    baseline_unsupported_rate_estimate: Optional[float] = None


def _percentile(data: list[float], p: float) -> Optional[float]:
    """Calculate percentile from a list of floats."""
    if not data:
        return None
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def calculate_run_metrics(
    query_results: list[dict[str, Any]],
    redteam_results: dict[str, Any],
    baseline_results: Optional[list[dict[str, Any]]] = None,
    mode: str = "mock",
    model_name: str = "unknown",
    search_provider: str = "unknown",
) -> EvaluationMetrics:
    """Calculate all evaluation metrics from test outputs with explicit denominators."""
    metrics = EvaluationMetrics(
        mode=mode,
        model_name=model_name,
        search_provider=search_provider,
    )

    total_queries = len(query_results)
    metrics.total_queries = total_queries

    total_claims = 0
    verified_claims = 0
    rejected_claims = 0
    answerable_count = 0
    unanswerable_count = 0
    answerable_abstained = 0
    unanswerable_abstained = 0

    valid_citations = 0
    total_citations = 0
    expected_total = 0
    expected_hit = 0
    coverage_ratios: list[float] = []
    latencies: list[float] = []

    total_tokens_in = 0
    total_tokens_out = 0
    provider_errors = 0
    provider_retries = 0

    for res in query_results:
        answer: FinalAnswer = res["answer"]
        is_answerable: bool = res.get("is_answerable", True)
        required_entities: list[str] = res.get("required_entities", [])

        if answer.run_metadata and answer.run_metadata.wall_clock_seconds is not None:
            latencies.append(answer.run_metadata.wall_clock_seconds)
            if answer.run_metadata.total_input_tokens:
                total_tokens_in += answer.run_metadata.total_input_tokens
            if answer.run_metadata.total_output_tokens:
                total_tokens_out += answer.run_metadata.total_output_tokens
            if answer.run_metadata.errors:
                provider_errors += len(answer.run_metadata.errors)
            provider_retries += answer.run_metadata.retries

        num_v = len(answer.verified_findings)
        num_r = len(answer.rejected_claims)
        verified_claims += num_v
        rejected_claims += num_r
        total_claims += (num_v + num_r)

        # A citation is precise when the cited source itself contains one of the
        # claim's quotes. Over-citation (listing sources that don't hold the quote)
        # lowers this; a verified claim needs only one matching source.
        for finding in answer.verified_findings:
            quotes = [_normalize_for_match(q) for q in finding.claim.verbatim_quotes if q.strip()]
            for ev in finding.supporting_evidence:
                total_citations += 1
                ev_text = _normalize_for_match(ev.text)
                if ev.url and any(q in ev_text for q in quotes):
                    valid_citations += 1

        expected: list[str] = res.get("expected_findings", []) or []
        if expected and is_answerable:
            answer_text = " ".join([f.claim.claim_text for f in answer.verified_findings] + [answer.short_answer])
            expected_total += len(expected)
            expected_hit += sum(1 for e in expected if expected_finding_hit(e, answer_text))

        if is_answerable:
            answerable_count += 1
            if answer.execution_status == "abstained":
                answerable_abstained += 1
        else:
            unanswerable_count += 1
            if answer.execution_status == "abstained":
                unanswerable_abstained += 1

        if required_entities:
            text_corpus = " ".join(
                [f.claim.claim_text for f in answer.verified_findings]
                + [answer.short_answer]
            ).lower()
            matched = sum(1 for e in required_entities if e.lower() in text_corpus)
            coverage_ratios.append(matched / len(required_entities))
        else:
            coverage_ratios.append(1.0)

    metrics.answerable_queries = answerable_count
    metrics.unanswerable_queries = unanswerable_count
    metrics.total_claims = total_claims
    metrics.verified_claims = verified_claims
    metrics.rejected_claims = rejected_claims

    metrics.claim_support_rate = (
        (verified_claims / total_claims) if total_claims > 0 else 0.0
    )
    metrics.unsupported_claims_per_answer = (
        (rejected_claims / total_queries) if total_queries > 0 else 0.0
    )
    metrics.cited_pairs = total_citations
    metrics.citation_precision = (valid_citations / total_citations) if total_citations > 0 else None
    metrics.expected_findings_total = expected_total
    metrics.expected_findings_hit = expected_hit
    metrics.expected_finding_recall = (expected_hit / expected_total) if expected_total > 0 else None
    metrics.correct_abstention_rate = (
        (unanswerable_abstained / unanswerable_count)
        if unanswerable_count > 0
        else None
    )
    metrics.abstention_rate_on_answerable = (
        (answerable_abstained / answerable_count)
        if answerable_count > 0
        else None
    )
    metrics.mean_entity_coverage = (
        (sum(coverage_ratios) / len(coverage_ratios)) if coverage_ratios else 0.0
    )

    metrics.latencies = latencies
    metrics.p50_latency_seconds = _percentile(latencies, 50)
    metrics.p95_latency_seconds = _percentile(latencies, 95)

    metrics.total_input_tokens = total_tokens_in
    metrics.total_output_tokens = total_tokens_out
    metrics.provider_errors = provider_errors
    metrics.provider_retries = provider_retries

    metrics.redteam_total_attacks = redteam_results.get("total_attacks", 0)
    metrics.redteam_blocked_attacks = redteam_results.get("blocked_attacks", 0)
    total_atk = metrics.redteam_total_attacks
    blocked_atk = metrics.redteam_blocked_attacks
    metrics.redteam_attack_success_rate = (
        ((total_atk - blocked_atk) / total_atk) if total_atk > 0 else 0.0
    )

    metrics.redteam_benign_controls = redteam_results.get("benign_controls", 0)
    metrics.redteam_false_positives = redteam_results.get("false_positives", 0)
    controls = metrics.redteam_benign_controls
    fps = metrics.redteam_false_positives
    metrics.redteam_false_positive_rate = (fps / controls) if controls > 0 else 0.0

    if baseline_results:
        metrics.baseline_total_queries = len(baseline_results)
        base_lats = [
            b["answer"].run_metadata.wall_clock_seconds
            for b in baseline_results
            if b["answer"].run_metadata and b["answer"].run_metadata.wall_clock_seconds is not None
        ]
        metrics.baseline_p50_latency = _percentile(base_lats, 50)
        metrics.baseline_unsupported_rate_estimate = 1.0

    return metrics


def generate_markdown_summary(metrics: EvaluationMetrics) -> str:
    """Generate a clean, structured Markdown evaluation report."""
    md = []
    md.append(f"# Research Agent Evaluation Report")
    md.append(f"**Execution Mode:** `{metrics.mode.upper()}`  ")
    md.append(f"**Run Timestamp:** {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    md.append(f"**LLM Model:** `{metrics.model_name}`  ")
    md.append(f"**Search Engine:** `{metrics.search_provider}`  \n")

    md.append("## 1. Core Research Quality Metrics\n")
    md.append("| Metric | Value | Denominator / Formula |")
    md.append("| :--- | :--- | :--- |")
    
    csr = f"{metrics.claim_support_rate * 100:.1f}%" if metrics.claim_support_rate is not None else "N/A"
    md.append(f"| **Claim Support Rate** | `{csr}` | {metrics.verified_claims} verified / {metrics.total_claims} claims |")

    cp = f"{metrics.citation_precision * 100:.1f}%" if metrics.citation_precision is not None else "N/A"
    md.append(f"| **Citation Precision** | `{cp}` | Cited sources containing the claim's quote / {metrics.cited_pairs} citations |")
    efr = f"{metrics.expected_finding_recall * 100:.1f}%" if metrics.expected_finding_recall is not None else "N/A"
    md.append(f"| **Expected-Finding Recall** | `{efr}` | {metrics.expected_findings_hit} / {metrics.expected_findings_total} golden findings stated (lexical overlap >= 50%) |")

    md.append(f"| **Unsupported Claims / Answer** | `{metrics.unsupported_claims_per_answer:.2f}` | {metrics.rejected_claims} rejected / {metrics.total_queries} queries |")

    car = f"{metrics.correct_abstention_rate * 100:.1f}%" if metrics.correct_abstention_rate is not None else "N/A"
    md.append(f"| **Correct Abstention Rate** | `{car}` | Abstained unanswerable queries / {metrics.unanswerable_queries} |")

    ara = f"{metrics.abstention_rate_on_answerable * 100:.1f}%" if metrics.abstention_rate_on_answerable is not None else "N/A"
    md.append(f"| **False Abstention Rate (Answerable)** | `{ara}` | Abstained answerable queries / {metrics.answerable_queries} |")

    md.append(f"| **Mean Entity Coverage** | `{metrics.mean_entity_coverage * 100:.1f}%` | Matched key target entities across queries |")

    md.append("\n## 2. Red-Team & Adversarial Security Metrics\n")
    md.append("| Metric | Value | Details |")
    md.append("| :--- | :--- | :--- |")
    asr = f"{metrics.redteam_attack_success_rate * 100:.1f}%"
    md.append(f"| **Attack Success Rate (Defenses ON)** | `{asr}` | Blocked {metrics.redteam_blocked_attacks}/{metrics.redteam_total_attacks} attacks |")
    fpr = f"{metrics.redteam_false_positive_rate * 100:.1f}%"
    md.append(f"| **Benign False Positive Rate** | `{fpr}` | Quarantined {metrics.redteam_false_positives}/{metrics.redteam_benign_controls} benign controls |")

    md.append("\n## 3. Operational Performance & Baseline Comparison\n")
    md.append("| Metric | Research Agent Pipeline | Single-Call LLM Baseline |")
    md.append("| :--- | :--- | :--- |")
    p50 = f"{metrics.p50_latency_seconds:.2f}s" if metrics.p50_latency_seconds else "N/A"
    bp50 = f"{metrics.baseline_p50_latency:.2f}s" if metrics.baseline_p50_latency else "N/A"
    md.append(f"| **P50 Latency** | `{p50}` | `{bp50}` |")

    p95 = f"{metrics.p95_latency_seconds:.2f}s" if metrics.p95_latency_seconds else "N/A"
    md.append(f"| **P95 Latency** | `{p95}` | N/A |")

    md.append(f"| **Total Tokens** | `{metrics.total_input_tokens + metrics.total_output_tokens}` ({metrics.total_input_tokens} in / {metrics.total_output_tokens} out) | Standard prompt context |")
    md.append(f"| **Citation Grounding** | **Deterministic receipts & quotes** | Uncited internal model weights (0 receipts) |")
    md.append(f"| **Provider Failures / Retries** | `{metrics.provider_errors}` / `{metrics.provider_retries}` | N/A |")

    md.append("\n> [!NOTE]\n> Denominators are explicitly tracked. Offline/mock runs reflect synthetic fixture performance; live runs require live credentials.\n")

    return "\n".join(md)
