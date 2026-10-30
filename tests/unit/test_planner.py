"""Tests for planner module."""
import pytest
import json

from research_agent.planner import (
    validate_question, _parse_plan_response, _fallback_plan,
    create_plan,
)
from research_agent.schemas import QuestionType
from research_agent.llm import MockLLMProvider


class TestQuestionValidation:
    def test_valid_question(self):
        valid, result = validate_question('What is Python?')
        assert valid is True
        assert result == 'What is Python?'
    
    def test_empty_question(self):
        valid, result = validate_question('')
        assert valid is False
    
    def test_too_long_question(self):
        valid, result = validate_question('x' * 1001)
        assert valid is False
    
    def test_too_short_question(self):
        valid, result = validate_question('Hi')
        assert valid is False
    
    def test_whitespace_trimmed(self):
        valid, result = validate_question('  What is Python?  ')
        assert valid is True
        assert result == 'What is Python?'


class TestPlanParsing:
    def test_valid_json(self):
        data = json.dumps({
            'question': 'Test',
            'question_type': 'factual',
            'steps': [
                {'step_id': 1, 'description': 'Search', 'search_queries': ['test']}
            ],
            'success_criteria': [],
            'assumptions': [],
            'required_entities': [],
            'required_aspects': [],
        })
        plan = _parse_plan_response(data, 'Test')
        assert plan.question_type == QuestionType.FACTUAL
        assert len(plan.steps) == 1
    
    def test_invalid_json_fallback(self):
        plan = _parse_plan_response('not json at all', 'Test')
        assert len(plan.steps) == 1
        assert 'fallback' in plan.assumptions[0].lower() or 'Fallback' in plan.assumptions[0]
    
    def test_invalid_question_type_defaults(self):
        data = json.dumps({
            'question': 'Test',
            'question_type': 'invalid_type',
            'steps': [
                {'step_id': 1, 'description': 'Search', 'search_queries': ['test']}
            ],
        })
        plan = _parse_plan_response(data, 'Test')
        assert plan.question_type == QuestionType.EXPLORATORY


class TestCreatePlan:
    @pytest.mark.asyncio
    async def test_create_plan_with_mock(self):
        llm = MockLLMProvider()
        llm.set_sequential_responses([json.dumps({
            'question': 'What is Python?',
            'question_type': 'factual',
            'steps': [
                {'step_id': 1, 'description': 'Search', 'search_queries': ['Python language']}
            ],
            'success_criteria': [],
            'assumptions': [],
            'required_entities': ['Python'],
            'required_aspects': [],
        })])
        
        plan = await create_plan('What is Python?', llm)
        assert plan.question == 'What is Python?'
        assert len(plan.steps) >= 1
