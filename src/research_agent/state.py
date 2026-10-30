"""Execution state tracking for research agent runs."""

from __future__ import annotations

import time
from typing import Optional

from .config import Settings
from .schemas import (
    Claim,
    Evidence,
    RejectedClaim,
    ResearchPlan,
    VerifiedFinding,
)


class ResearchState:
    """Encapsulates the mutable lifecycle state of a research session."""

    def __init__(
        self,
        plan: Optional[ResearchPlan] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Initialize a fresh research state session."""
        self.plan: Optional[ResearchPlan] = plan
        self.run_id: Optional[str] = run_id
        self.collected_evidence: dict[str, Evidence] = {}
        self.claims: list[Claim] = []
        self._claim_map: dict[str, Claim] = {}
        self.verified_findings: list[VerifiedFinding] = []
        self.rejected_claims: list[RejectedClaim] = []
        self.tool_call_count: int = 0
        self.llm_call_count: int = 0
        self.total_input_tokens: Optional[int] = None
        self.total_output_tokens: Optional[int] = None
        self.errors: list[str] = []
        self.budget_truncations: list[str] = []
        self.retry_count: int = 0
        # Provider/model that actually answered (differs from config under failover).
        self.llm_provider_used: str = ""
        self.llm_model_used: str = ""
        self._start_time: float = time.monotonic()

    @property
    def current_plan(self) -> Optional[ResearchPlan]:
        """Return the current research plan."""
        return self.plan

    @current_plan.setter
    def current_plan(self, plan: Optional[ResearchPlan]) -> None:
        """Set or update the active research plan."""
        self.plan = plan

    def elapsed_seconds(self) -> float:
        """Return the number of seconds elapsed since state initialization."""
        return time.monotonic() - self._start_time

    def add_evidence(self, evidence: Evidence) -> None:
        """Store a retrieved evidence item, keyed by its unique evidence_id."""
        self.collected_evidence[evidence.evidence_id] = evidence

    def get_evidence_by_id(self, eid: str) -> Optional[Evidence]:
        """Retrieve evidence by its ID if present."""
        return self.collected_evidence.get(eid)

    def get_all_evidence(self) -> list[Evidence]:
        """Return all collected evidence items as a list."""
        return list(self.collected_evidence.values())

    def add_claim(self, claim: Claim) -> None:
        """Register a new claim candidate for verification."""
        self.claims.append(claim)
        self._claim_map[claim.claim_id] = claim

    def get_claim_by_id(self, cid: str) -> Optional[Claim]:
        """Retrieve claim by its ID if present."""
        return self._claim_map.get(cid)

    def mark_verified(self, finding: VerifiedFinding) -> None:
        """Mark a finding as verified and update the corresponding claim."""
        finding.claim.verification_status = "verified"
        if finding.claim.claim_id in self._claim_map:
            self._claim_map[finding.claim.claim_id].verification_status = "verified"
        self.verified_findings.append(finding)

    def mark_rejected(self, rejected: RejectedClaim) -> None:
        """Mark a claim as rejected and record the failure details."""
        rejected.claim.verification_status = "rejected"
        if rejected.rejection_reasons:
            rejected.claim.rejection_reason = "; ".join(rejected.rejection_reasons)
        if rejected.claim.claim_id in self._claim_map:
            self._claim_map[rejected.claim.claim_id].verification_status = "rejected"
            self._claim_map[rejected.claim.claim_id].rejection_reason = rejected.claim.rejection_reason
        self.rejected_claims.append(rejected)

    def increment_tool_calls(self, n: int = 1) -> None:
        """Increment the count of external tool calls."""
        self.tool_call_count += n

    def increment_llm_calls(self, n: int = 1) -> None:
        """Increment the count of LLM API invocations."""
        self.llm_call_count += n

    def add_token_usage(self, input_tokens: Optional[int], output_tokens: Optional[int]) -> None:
        """Accumulate token usage reported by the LLM provider."""
        if input_tokens is not None:
            self.total_input_tokens = (self.total_input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.total_output_tokens = (self.total_output_tokens or 0) + output_tokens

    def add_error(self, error: str) -> None:
        """Log an error message into state."""
        self.errors.append(error)

    def check_budget(self, settings: Settings) -> bool:
        """Check whether execution remains within configured tool, LLM, and time budgets."""
        if self.tool_call_count >= settings.max_tool_calls_per_run:
            return False
        if self.llm_call_count >= settings.max_llm_calls_per_run:
            return False
        if self.elapsed_seconds() >= settings.max_wall_clock_seconds:
            return False
        return True

    def budget_remaining(self, settings: Settings) -> dict[str, float | int]:
        """Compute the remaining resource budget for this run."""
        return {
            "remaining_tool_calls": max(0, settings.max_tool_calls_per_run - self.tool_call_count),
            "remaining_llm_calls": max(0, settings.max_llm_calls_per_run - self.llm_call_count),
            "remaining_wall_clock_seconds": max(0.0, settings.max_wall_clock_seconds - self.elapsed_seconds()),
        }
