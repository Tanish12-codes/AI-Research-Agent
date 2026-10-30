"""Shared test fixtures and configuration."""
import pytest
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / 'src'))

try:
    import research_agent
    sys.modules.setdefault('src.research_agent', research_agent)
except ImportError:
    pass

from research_agent.schemas import (
    Evidence, Claim, ResearchPlan, PlanStep, SuccessCriterion,
    QuestionType, VerifiedFinding, RejectedClaim,
)
from research_agent.llm import MockLLMProvider
from research_agent.tools.search import MockSearchProvider, SearchWithFallback
from research_agent.config import Settings


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    """Create sample evidence for testing."""
    return [
        Evidence(
            evidence_id="ev-001",
            url="https://example.com/article1",
            domain="example.com",
            title="Test Article 1",
            text="Python is a programming language created by Guido van Rossum. It has over 100,000 packages on PyPI. Python 3.12 was released in October 2023.",
            retrieval_timestamp=datetime.now(timezone.utc),
            provider="tavily",
            relevance_score=0.9,
            authority_score=0.7,
        ),
        Evidence(
            evidence_id="ev-002",
            url="https://github.com/python/cpython",
            domain="github.com",
            title="python/cpython",
            text="Repository: python/cpython. Description: The Python programming language. Stars: 58000. Language: Python. License: PSF License.",
            retrieval_timestamp=datetime.now(timezone.utc),
            provider="github",
            relevance_score=0.85,
            authority_score=0.9,
        ),
        Evidence(
            evidence_id="ev-003",
            url="https://docs.python.org/3/",
            domain="docs.python.org",
            title="Python 3.12 Documentation",
            text="Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
            retrieval_timestamp=datetime.now(timezone.utc),
            provider="fetch",
            relevance_score=0.95,
            authority_score=0.95,
        ),
    ]


@pytest.fixture
def sample_claims(sample_evidence) -> list[Claim]:
    """Create sample claims for testing."""
    return [
        Claim(
            claim_id="claim-1",
            claim_text="Python is a programming language created by Guido van Rossum.",
            evidence_ids=["ev-001"],
            verbatim_quotes=["Python is a programming language created by Guido van Rossum"],
            relevant_entities=["Python", "Guido van Rossum"],
            numeric_assertions=[],
        ),
        Claim(
            claim_id="claim-2",
            claim_text="Python has over 100,000 packages available on PyPI.",
            evidence_ids=["ev-001"],
            verbatim_quotes=["over 100,000 packages on PyPI"],
            relevant_entities=["Python", "PyPI"],
            numeric_assertions=["100,000 packages"],
        ),
        Claim(
            claim_id="claim-3",
            claim_text="The Python CPython repository has 58000 stars on GitHub.",
            evidence_ids=["ev-002"],
            verbatim_quotes=["Stars: 58000"],
            relevant_entities=["CPython", "GitHub"],
            numeric_assertions=["58000 stars"],
        ),
    ]


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Create a mock LLM provider."""
    plan_response = json.dumps({
        "question": "What is Python?",
        "question_type": "factual",
        "steps": [
            {
                "step_id": 1,
                "description": "Search for Python programming language overview",
                "search_queries": ["Python programming language overview"],
                "target_entities": ["Python"],
                "assumptions": ["Refers to the programming language"],
            }
        ],
        "success_criteria": [
            {
                "description": "Find basic information about Python",
                "required_entities": ["Python"],
                "required_aspects": ["description", "creator"],
                "measurable": True,
            }
        ],
        "assumptions": ["User asks about Python programming language"],
        "required_entities": ["Python"],
        "required_aspects": ["description", "creator"],
    })
    
    claims_response = json.dumps({
        "claims": [
            {
                "claim_id": "claim-1",
                "claim_text": "Python is a programming language created by Guido van Rossum.",
                "evidence_ids": ["ev-001"],
                "verbatim_quotes": ["Python is a programming language created by Guido van Rossum"],
                "relevant_entities": ["Python", "Guido van Rossum"],
                "relevant_criteria": ["description", "creator"],
                "numeric_assertions": [],
            }
        ]
    })
    
    llm = MockLLMProvider()
    llm.set_sequential_responses([plan_response, claims_response])
    return llm


@pytest.fixture
def mock_search(sample_evidence) -> SearchWithFallback:
    """Create a mock search provider with sample results."""
    provider = MockSearchProvider()
    provider.add_response("python", sample_evidence[:2])
    return SearchWithFallback(provider)


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        llm_provider="mock",
        llm_model="mock-model",
        openai_api_key="test-key",
        tavily_api_key="test-key",
        max_concurrent_searches=2,
        search_timeout_seconds=5,
        llm_timeout_seconds=5,
        cache_ttl_seconds=60,
        max_tool_calls_per_run=10,
        max_llm_calls_per_run=5,
        max_wall_clock_seconds=30,
        confidence_threshold=0.3,
        enable_paper_search=False,
    )
