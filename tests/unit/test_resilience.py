"""Tests for budget-aware retries and token accounting."""
import httpx
import pytest

from research_agent.budgets import BudgetTracker
from research_agent.config import Settings
from research_agent.llm import LLMProvider, LLMResponse
from research_agent.resilience import BudgetedLLM, is_transient_error, search_with_retry
from research_agent.state import ResearchState


class FlakyLLM(LLMProvider):
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    @property
    def name(self):
        return 'flaky'

    @property
    def model_name(self):
        return 'flaky-model'

    async def complete(self, messages, temperature=0.0, max_tokens=2000, response_format=None):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return LLMResponse(content='{}', input_tokens=10, output_tokens=5)


class FlakySearch:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    async def search(self, query, max_results=5):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return []


def _settings(**overrides):
    return Settings(llm_provider='mock', retry_base_delay_seconds=0, **overrides)


def _wrap(inner, settings):
    budget = BudgetTracker(max_tool_calls=settings.max_tool_calls_per_run,
                           max_llm_calls=settings.max_llm_calls_per_run,
                           max_wall_clock=settings.max_wall_clock_seconds)
    state = ResearchState()
    return BudgetedLLM(inner, budget, state, settings), budget, state


def _status_error(code):
    request = httpx.Request('GET', 'https://api.example.com')
    return httpx.HTTPStatusError('err', request=request, response=httpx.Response(code, request=request))


def test_transient_classification():
    assert is_transient_error(httpx.ConnectError('down'))
    assert is_transient_error(TimeoutError())
    assert is_transient_error(_status_error(429))
    assert is_transient_error(_status_error(503))
    assert not is_transient_error(_status_error(401))
    assert not is_transient_error(ValueError('bad json'))


@pytest.mark.asyncio
async def test_llm_transient_error_retried_and_charged():
    settings = _settings()
    llm, budget, state = _wrap(FlakyLLM([httpx.ConnectError('down')]), settings)

    await llm.complete([{'role': 'user', 'content': 'hi'}])

    assert llm.inner.calls == 2
    assert budget.llm_calls == 2
    assert state.retry_count == 1
    assert (state.total_input_tokens, state.total_output_tokens) == (10, 5)


@pytest.mark.asyncio
async def test_llm_non_transient_error_not_retried():
    llm, _, state = _wrap(FlakyLLM([ValueError('bad')]), _settings())
    with pytest.raises(ValueError):
        await llm.complete([{'role': 'user', 'content': 'hi'}])
    assert llm.inner.calls == 1
    assert state.retry_count == 0


@pytest.mark.asyncio
async def test_llm_retry_stops_when_budget_exhausted():
    llm, _, _ = _wrap(FlakyLLM([httpx.ConnectError('down')] * 5), _settings(max_llm_calls_per_run=2, max_retries=5))
    with pytest.raises(httpx.ConnectError):
        await llm.complete([{'role': 'user', 'content': 'hi'}])
    assert llm.inner.calls == 2


@pytest.mark.asyncio
async def test_llm_retry_stops_at_max_retries():
    llm, _, _ = _wrap(FlakyLLM([httpx.ConnectError('down')] * 5), _settings(max_retries=1))
    with pytest.raises(httpx.ConnectError):
        await llm.complete([{'role': 'user', 'content': 'hi'}])
    assert llm.inner.calls == 2


@pytest.mark.asyncio
async def test_tokens_accumulate_across_calls():
    llm, _, state = _wrap(FlakyLLM([]), _settings())
    await llm.complete([{'role': 'user', 'content': 'a'}])
    await llm.complete([{'role': 'user', 'content': 'b'}])
    assert (state.total_input_tokens, state.total_output_tokens) == (20, 10)


@pytest.mark.asyncio
async def test_search_rate_limit_retried_and_charged():
    settings = _settings()
    budget = BudgetTracker(max_tool_calls=20, max_llm_calls=10, max_wall_clock=120)
    state = ResearchState()
    search = FlakySearch([_status_error(429)])

    await search_with_retry(search, 'q', budget, state, settings)

    assert search.calls == 2
    assert budget.tool_calls == 2
    assert state.tool_call_count == 2
    assert state.retry_count == 1
