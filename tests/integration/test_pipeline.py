"""Integration tests for the research pipeline."""
import pytest
import json
from datetime import datetime, timezone

from research_agent.pipeline import run_research_pipeline, run_baseline_no_tools
from research_agent.schemas import Evidence
from research_agent.llm import MockLLMProvider
from research_agent.tools.search import MockSearchProvider, SearchWithFallback
from research_agent.config import Settings


def _create_mock_providers():
    """Create mock LLM and search for integration tests."""
    evidence = [
        Evidence(
            evidence_id='ev-int-001',
            url='https://example.com/python',
            domain='example.com',
            title='Python Overview',
            text='Python is a high-level programming language created by Guido van Rossum in 1991. It has over 100,000 packages on PyPI and is used for web development, data science, and automation.',
            retrieval_timestamp=datetime.now(timezone.utc),
            provider='mock',
            relevance_score=0.9,
            authority_score=0.8,
        ),
        Evidence(
            evidence_id='ev-int-002',
            url='https://docs.python.org/',
            domain='docs.python.org',
            title='Python Docs',
            text='Python 3.12 documentation. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.',
            retrieval_timestamp=datetime.now(timezone.utc),
            provider='mock',
            relevance_score=0.85,
            authority_score=0.95,
        ),
    ]
    
    search_provider = MockSearchProvider()
    search_provider.add_response('python', evidence)
    search = SearchWithFallback(search_provider)
    
    plan_response = json.dumps({
        'question': 'What is Python?',
        'question_type': 'factual',
        'steps': [
            {
                'step_id': 1,
                'description': 'Search for Python overview',
                'search_queries': ['Python programming language overview'],
                'target_entities': ['Python'],
                'assumptions': [],
            }
        ],
        'success_criteria': [
            {
                'description': 'Find Python description',
                'required_entities': ['Python'],
                'required_aspects': ['description'],
                'measurable': True,
            }
        ],
        'assumptions': ['Refers to the programming language'],
        'required_entities': ['Python'],
        'required_aspects': ['description'],
    })
    
    claims_response = json.dumps({
        'claims': [
            {
                'claim_id': 'claim-1',
                'claim_text': 'Python is a high-level programming language created by Guido van Rossum in 1991.',
                'evidence_ids': ['ev-int-001'],
                'verbatim_quotes': ['Python is a high-level programming language created by Guido van Rossum in 1991'],
                'relevant_entities': ['Python', 'Guido van Rossum'],
                'numeric_assertions': ['1991'],
            },
            {
                'claim_id': 'claim-2',
                'claim_text': 'Python has over 100,000 packages on PyPI.',
                'evidence_ids': ['ev-int-001'],
                'verbatim_quotes': ['over 100,000 packages on PyPI'],
                'relevant_entities': ['Python', 'PyPI'],
                'numeric_assertions': ['100,000 packages'],
            },
        ]
    })
    
    llm = MockLLMProvider()
    llm.set_sequential_responses([plan_response, claims_response])
    
    settings = Settings(
        llm_provider='mock',
        llm_model='mock',
        max_concurrent_searches=2,
        search_timeout_seconds=5,
        llm_timeout_seconds=5,
        max_tool_calls_per_run=10,
        max_llm_calls_per_run=5,
        max_wall_clock_seconds=30,
        confidence_threshold=0.1,
        enable_paper_search=False,
    )
    
    return llm, search, settings


class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_successful_research(self):
        """Test complete pipeline with mock providers."""
        llm, search, settings = _create_mock_providers()
        
        answer = await run_research_pipeline(
            question='What is Python?',
            llm=llm,
            search=search,
            settings=settings,
        )
        
        assert answer.question == 'What is Python?'
        assert answer.execution_status in ('completed', 'abstained')
        assert answer.confidence_score >= 0.0
        assert answer.run_metadata is not None
        assert answer.run_metadata.wall_clock_seconds is not None
    
    @pytest.mark.asyncio
    async def test_empty_search_results(self):
        """Test pipeline handles no search results gracefully."""
        llm = MockLLMProvider()
        plan_response = json.dumps({
            'question': 'nonexistent topic xyz123',
            'question_type': 'exploratory',
            'steps': [{'step_id': 1, 'description': 'Search', 'search_queries': ['nonexistent xyz123']}],
            'assumptions': [], 'required_entities': [], 'required_aspects': [],
        })
        llm.set_sequential_responses([plan_response])
        
        search = SearchWithFallback(MockSearchProvider())
        settings = Settings(
            llm_provider='mock', llm_model='mock',
            max_tool_calls_per_run=10, max_llm_calls_per_run=5,
            max_wall_clock_seconds=30, confidence_threshold=0.3,
        )
        
        answer = await run_research_pipeline(
            question='nonexistent topic xyz123',
            llm=llm, search=search, settings=settings,
        )
        
        assert answer.execution_status == 'abstained'
        assert answer.abstention_reason is not None
    
    @pytest.mark.asyncio
    async def test_invalid_question(self):
        """Test pipeline rejects invalid questions."""
        llm = MockLLMProvider()
        search = SearchWithFallback(MockSearchProvider())
        settings = Settings(llm_provider='mock', llm_model='mock')
        
        answer = await run_research_pipeline(
            question='Hi',
            llm=llm, search=search, settings=settings,
        )
        
        assert answer.execution_status == 'error'


class TestBaselineComparison:
    @pytest.mark.asyncio
    async def test_baseline_runs(self):
        llm = MockLLMProvider(default_response=json.dumps({
            'answer': 'Python is a programming language.',
            'key_findings': ['General-purpose language'],
            'confidence': 'medium',
            'limitations': ['Based on training data'],
        }))
        settings = Settings(llm_provider='mock', llm_model='mock')
        
        answer = await run_baseline_no_tools('What is Python?', llm, settings)
        
        assert answer.execution_status == 'completed'
        assert answer.run_metadata.tool_calls == 0
        assert answer.run_metadata.llm_calls == 1
        assert 'no search' in answer.confidence_explanation.lower() or 'baseline' in answer.confidence_explanation.lower()


class TestBudgetDegradation:
    @pytest.mark.asyncio
    async def test_tool_budget_exhaustion_still_synthesizes(self):
        """Running out of tool calls truncates research but still returns an answer."""
        llm, search, settings = _create_mock_providers()
        plan = json.loads(llm._sequential[0])
        plan['steps'][0]['search_queries'] = [f'Python query {i}' for i in range(4)]
        llm._sequential[0] = json.dumps(plan)
        settings.max_tool_calls_per_run = 2

        answer = await run_research_pipeline(
            question='What is Python?', llm=llm, search=search, settings=settings,
        )

        assert answer.execution_status == 'partial'
        assert answer.verified_findings
        assert answer.run_metadata.tool_calls <= 2
        assert any('2 of 4 planned searches skipped' in lim for lim in answer.limitations)

    @pytest.mark.asyncio
    async def test_concurrent_searches_do_not_overshoot_tool_budget(self):
        llm, search, settings = _create_mock_providers()
        plan = json.loads(llm._sequential[0])
        plan['steps'] = [
            {'step_id': i, 'description': 'd', 'search_queries': [f'Python q{i}{j}' for j in range(4)]}
            for i in range(4)
        ]
        llm._sequential[0] = json.dumps(plan)
        settings.max_tool_calls_per_run = 3
        settings.max_concurrent_searches = 16

        answer = await run_research_pipeline(
            question='What is Python?', llm=llm, search=search, settings=settings,
        )

        assert len(search.primary._calls) == 3
        assert answer.run_metadata.tool_calls == 3

    @pytest.mark.asyncio
    async def test_llm_budget_exhaustion_returns_partial_with_sources(self):
        """With no LLM call left for synthesis, collected sources are still returned."""
        llm, search, settings = _create_mock_providers()
        settings.max_llm_calls_per_run = 1

        answer = await run_research_pipeline(
            question='What is Python?', llm=llm, search=search, settings=settings,
        )

        assert answer.execution_status == 'partial'
        assert answer.verified_findings == []
        assert 'https://example.com/python' in answer.source_links
        assert answer.confidence_score == 0.0


class TestRunAccounting:
    @pytest.mark.asyncio
    async def test_token_usage_reported(self):
        llm, search, settings = _create_mock_providers()
        answer = await run_research_pipeline(
            question='What is Python?', llm=llm, search=search, settings=settings,
        )
        assert answer.run_metadata.total_input_tokens
        assert answer.run_metadata.total_output_tokens
        assert answer.run_metadata.retries == 0


class TestSynthesisRetry:
    """When the checker strikes every claim, the pipeline rewrites once with the reasons."""

    @pytest.mark.asyncio
    async def test_rejected_first_attempt_is_retried_with_feedback(self):
        llm, search, settings = _create_mock_providers()
        plan_response, good_claims = llm._sequential
        bad_claims = json.dumps({'claims': [{
            'claim_id': 'bad-1',
            'claim_text': 'Python was created in 1989 by a committee.',
            'evidence_ids': ['ev-int-001'],
            'verbatim_quotes': ['Python was designed by a committee in 1989'],
            'relevant_entities': ['Python'],
            'numeric_assertions': ['1989'],
        }]})
        # plan, coverage replan, first synthesis (all struck), corrective retry
        llm.set_sequential_responses([plan_response, plan_response, bad_claims, good_claims])

        answer = await run_research_pipeline(
            question='What is Python?', llm=llm, search=search, settings=settings,
        )

        assert len(answer.verified_findings) >= 1
        assert any('rewritten once' in lim for lim in answer.limitations)
        retry_prompt = llm._calls[3]['messages'][-1]['content']
        assert 'checker rejected your previous claims' in retry_prompt
        assert 'Quote not found' in retry_prompt
