"""Pydantic schemas and data contracts for the AI Research Agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Classification of the user's research question."""

    COMPARISON = "comparison"
    DISCOVERY = "discovery"
    PROS_CONS = "pros_cons"
    HOW_TO = "how_to"
    FACTUAL = "factual"
    EXPLORATORY = "exploratory"


class SuccessCriterion(BaseModel):
    """Specific measurable criterion determining if a research plan succeeded."""

    description: str = Field(description="Description of what must be verified or found")
    required_entities: list[str] = Field(default_factory=list, description="Entities that must be covered")
    required_aspects: list[str] = Field(default_factory=list, description="Specific aspects/facets required")
    measurable: bool = Field(default=True, description="Whether this criterion can be programmatically evaluated")


class PlanStep(BaseModel):
    """An individual step in a decomposed research plan."""

    step_id: int = Field(description="Sequential identifier for the step")
    description: str = Field(description="Goal of this step")
    search_queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="Search engine queries targeted for this step (1 to 4)",
    )
    target_entities: list[str] = Field(default_factory=list, description="Entities targeted in this step")
    assumptions: list[str] = Field(default_factory=list, description="Working assumptions for this step")


class ResearchPlan(BaseModel):
    """A structured multi-step research strategy produced before execution."""

    question: str = Field(description="The original user query")
    question_type: QuestionType = Field(description="Categorization of the research question")
    steps: list[PlanStep] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="List of structured research steps (1 to 4)",
    )
    success_criteria: list[SuccessCriterion] = Field(
        default_factory=list, description="List of criteria to evaluate completeness"
    )
    assumptions: list[str] = Field(default_factory=list, description="Assumptions made about the question")
    required_entities: list[str] = Field(default_factory=list, description="Core entities that must be addressed")
    required_aspects: list[str] = Field(default_factory=list, description="Core dimensions that must be examined")


class Evidence(BaseModel):
    """A single piece of retrieved information backed by a verified source."""

    evidence_id: str = Field(description="Unique identifier for this evidence item")
    url: str = Field(description="Canonical URL of source")
    domain: str = Field(description="Domain host of the source")
    title: str = Field(description="Title of page or document")
    text: str = Field(description="Extracted text snippet or bounded passage")
    retrieval_timestamp: datetime = Field(description="When the content was fetched")
    publication_date: Optional[datetime] = Field(default=None, description="Reported publication date")
    provider: str = Field(description="Provider name, e.g. 'tavily', 'brave', 'github', 'fetch'")
    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary provider-specific metadata"
    )
    relevance_score: Optional[float] = Field(default=None, description="Estimated topical relevance score (0.0-1.0)")
    authority_score: Optional[float] = Field(default=None, description="Estimated domain authority score (0.0-1.0)")
    content_hash: Optional[str] = Field(default=None, description="Cryptographic hash of the raw content")
    is_quarantined: bool = Field(default=False, description="Whether content is quarantined due to security flags")
    quarantine_reason: Optional[str] = Field(default=None, description="Explanation for quarantining")
    security_flags: list[str] = Field(default_factory=list, description="List of triggered security checks")
    trust_level: float = Field(default=1.0, description="Trust factor (0.0 - 1.0) adjusted by security evaluation")


class Claim(BaseModel):
    """A statement extracted from evidence that requires strict verification."""

    claim_id: str = Field(description="Unique identifier for the claim")
    claim_text: str = Field(description="Natural language claim assertion")
    evidence_ids: list[str] = Field(description="List of evidence IDs cited in support")
    verbatim_quotes: list[str] = Field(
        ...,
        min_length=1,
        description="At least one verbatim quote extracted from source evidence text",
    )
    relevant_entities: list[str] = Field(default_factory=list, description="Entities mentioned in claim")
    relevant_criteria: list[str] = Field(default_factory=list, description="Success criteria addressed")
    numeric_assertions: list[str] = Field(default_factory=list, description="Extracted numerical figures/metrics")
    verification_status: str = Field(
        default="pending", description="Status: 'pending', 'verified', or 'rejected'"
    )
    rejection_reason: Optional[str] = Field(default=None, description="Reason if claim was rejected")


class RejectedClaim(BaseModel):
    """A claim that failed validation, cross-checking, or source verification."""

    claim: Claim = Field(description="The rejected claim object")
    rejection_reasons: list[str] = Field(description="Reasons explaining why the claim was rejected")
    failed_checks: list[str] = Field(default_factory=list, description="Specific check IDs or names that failed")


class VerifiedFinding(BaseModel):
    """A verified factual finding with direct grounding in reliable evidence."""

    claim: Claim = Field(description="The verified claim object")
    supporting_evidence: list[Evidence] = Field(description="Evidence items supporting this finding")
    verification_checks_passed: list[str] = Field(description="List of verification checks that passed")
    confidence_contribution: float = Field(
        default=0.0, description="Contribution weight of this finding to overall confidence"
    )


class RunMetadata(BaseModel):
    """Observability metadata recording resource consumption and operational telemetry."""

    run_id: str = Field(description="Unique execution run identifier")
    start_time: datetime = Field(description="UTC timestamp of run start")
    end_time: Optional[datetime] = Field(default=None, description="UTC timestamp of run finish")
    wall_clock_seconds: Optional[float] = Field(default=None, description="Total execution duration in seconds")
    llm_calls: int = Field(default=0, description="Total number of LLM invocations performed")
    tool_calls: int = Field(default=0, description="Total number of external tool invocations")
    total_input_tokens: Optional[int] = Field(default=None, description="Total prompt/input tokens used")
    total_output_tokens: Optional[int] = Field(default=None, description="Total completion/output tokens generated")
    estimated_cost_usd: Optional[float] = Field(default=None, description="Estimated total cost in USD")
    search_provider: str = Field(default="unknown", description="Primary search engine used")
    llm_provider: str = Field(default="unknown", description="Primary LLM backend used")
    llm_model: str = Field(default="unknown", description="LLM model name used")
    retries: int = Field(default=0, description="Retried LLM/search attempts after transient errors")
    errors: list[str] = Field(default_factory=list, description="Errors encountered during run")
    warnings: list[str] = Field(default_factory=list, description="Warnings logged during run")


class FinalAnswer(BaseModel):
    """The synthesized output delivered to the user at the conclusion of research."""

    question: str = Field(description="User question answered")
    short_answer: str = Field(description="Concise direct answer / executive summary")
    key_findings: list[str] = Field(description="List of key verified takeaway findings")
    source_links: list[str] = Field(description="Referenced canonical URLs")
    verified_findings: list[VerifiedFinding] = Field(
        default_factory=list, description="Structured verified findings"
    )
    rejected_claims: list[RejectedClaim] = Field(
        default_factory=list, description="Claims rejected during cross-checking"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Overall confidence score bounded between 0.0 and 1.0"
    )
    confidence_explanation: str = Field(description="Explanation of confidence calculation rationale")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions underlying the answer")
    limitations: list[str] = Field(default_factory=list, description="Identified gaps or limitations in evidence")
    suggested_next_steps: list[str] = Field(default_factory=list, description="Recommended follow-up research")
    execution_status: str = Field(
        default="completed", description="Execution status: completed, partial, abstained, error"
    )
    run_metadata: Optional[RunMetadata] = Field(default=None, description="Run operational telemetry")
    abstention_reason: Optional[str] = Field(default=None, description="Reason if system abstained from answering")
