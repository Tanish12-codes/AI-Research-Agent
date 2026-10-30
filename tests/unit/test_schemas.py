"""Tests for Pydantic data contracts."""
import pytest
from datetime import datetime, timezone

from research_agent.schemas import (
    QuestionType, SuccessCriterion, PlanStep, ResearchPlan,
    Evidence, Claim, RejectedClaim, VerifiedFinding,
    RunMetadata, FinalAnswer,
)


class TestEvidence:
    def test_evidence_creation(self):
        ev = Evidence(
            evidence_id="test-1",
            url="https://example.com",
            domain="example.com",
            title="Test",
            text="Sample text",
            retrieval_timestamp=datetime.now(timezone.utc),
            provider="tavily",
        )
        assert ev.evidence_id == "test-1"
        assert ev.trust_level == 1.0
        assert ev.is_quarantined is False
    
    def test_evidence_defaults(self):
        ev = Evidence(
            evidence_id="test-2",
            url="https://example.com",
            domain="example.com",
            title="Test",
            text="Text",
            retrieval_timestamp=datetime.now(timezone.utc),
            provider="mock",
        )
        assert ev.relevance_score is None
        assert ev.authority_score is None
        assert ev.publication_date is None
        assert ev.security_flags == []
        assert ev.retrieval_metadata == {}


class TestClaim:
    def test_claim_requires_quotes(self):
        with pytest.raises(Exception):
            Claim(
                claim_id="c1",
                claim_text="test",
                evidence_ids=["ev1"],
                verbatim_quotes=[],
            )
    
    def test_claim_creation(self):
        claim = Claim(
            claim_id="c1",
            claim_text="Python is popular",
            evidence_ids=["ev1"],
            verbatim_quotes=["Python is popular"],
            numeric_assertions=["100K stars"],
        )
        assert claim.verification_status == "pending"
        assert claim.rejection_reason is None


class TestResearchPlan:
    def test_plan_bounds(self):
        """Plans must have 1-4 steps."""
        with pytest.raises(Exception):
            ResearchPlan(
                question="test",
                question_type=QuestionType.FACTUAL,
                steps=[],
            )
    
    def test_step_query_bounds(self):
        """Steps must have 1-4 queries."""
        with pytest.raises(Exception):
            PlanStep(
                step_id=1,
                description="test",
                search_queries=[],
            )
    
    def test_valid_plan(self):
        plan = ResearchPlan(
            question="What is Python?",
            question_type=QuestionType.FACTUAL,
            steps=[
                PlanStep(
                    step_id=1,
                    description="Search",
                    search_queries=["Python language"],
                )
            ],
        )
        assert plan.question_type == QuestionType.FACTUAL


class TestFinalAnswer:
    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            FinalAnswer(
                question="test",
                short_answer="answer",
                key_findings=[],
                source_links=[],
                confidence_score=1.5,
                confidence_explanation="test",
            )
    
    def test_abstention(self):
        answer = FinalAnswer(
            question="test",
            short_answer="Cannot answer",
            key_findings=[],
            source_links=[],
            confidence_score=0.1,
            confidence_explanation="Low confidence",
            execution_status="abstained",
            abstention_reason="Insufficient evidence",
        )
        assert answer.execution_status == "abstained"
        assert answer.abstention_reason is not None
