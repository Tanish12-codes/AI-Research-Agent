"""Budget-aware retries for LLM and search calls."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    stop_any,
    wait_exponential_jitter,
    wait_none,
)

from .budgets import BudgetTracker
from .config import Settings
from .llm import LLMProvider, LLMResponse
from .schemas import Evidence
from .state import ResearchState

_OPENAI_TRANSIENT = {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}


def is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return type(exc).__name__ in _OPENAI_TRANSIENT


def _retrying(
    settings: Settings,
    can_retry: Callable[[], bool],
    on_retry: Callable[[RetryCallState], None],
) -> AsyncRetrying:
    delay = settings.retry_base_delay_seconds
    return AsyncRetrying(
        retry=retry_if_exception(is_transient_error),
        stop=stop_any(stop_after_attempt(settings.max_retries + 1), lambda _: not can_retry()),
        wait=wait_exponential_jitter(initial=delay, max=4.0) if delay > 0 else wait_none(),
        before_sleep=on_retry,
        reraise=True,
    )


def _retry_logger(state: ResearchState, tracer: Optional[Any], kind: str) -> Callable[[RetryCallState], None]:
    def on_retry(retry_state: RetryCallState) -> None:
        state.retry_count += 1
        if tracer is not None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            tracer.log_error(f"{kind}_retry", f"attempt {retry_state.attempt_number} failed: {exc!r}")
    return on_retry


class BudgetedLLM(LLMProvider):
    """Wraps a provider with per-attempt budget accounting, timeouts, retries and token totals."""

    def __init__(
        self,
        inner: LLMProvider,
        budget: BudgetTracker,
        state: ResearchState,
        settings: Settings,
        tracer: Optional[Any] = None,
    ) -> None:
        self.inner = inner
        self.budget = budget
        self.state = state
        self.settings = settings
        self.tracer = tracer

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        retrying = _retrying(
            self.settings,
            lambda: self.budget.can_retry_llm(self.settings),
            _retry_logger(self.state, self.tracer, "llm"),
        )
        async for attempt in retrying:
            with attempt:
                self.budget.use_llm_call()
                self.state.increment_llm_calls()
                # A failover chain times out each provider itself; give it room for a full pass.
                timeout = getattr(self.inner, "total_timeout", None) or self.settings.llm_timeout_seconds
                response = await asyncio.wait_for(
                    self.inner.complete(messages, temperature, max_tokens, response_format),
                    timeout=timeout,
                )
        self.state.add_token_usage(response.input_tokens, response.output_tokens)
        self.state.llm_provider_used = self.inner.name
        self.state.llm_model_used = self.inner.model_name
        return response


async def search_with_retry(
    search: Any,
    query: str,
    budget: BudgetTracker,
    state: ResearchState,
    settings: Settings,
    tracer: Optional[Any] = None,
    max_results: int = 5,
) -> list[Evidence]:
    """Run one search, charging a tool call per attempt."""
    retrying = _retrying(
        settings,
        lambda: budget.can_retry_tool(settings),
        _retry_logger(state, tracer, "search"),
    )
    async for attempt in retrying:
        with attempt:
            budget.use_tool_call()
            state.increment_tool_calls()
            results = await asyncio.wait_for(
                search.search(query, max_results=max_results),
                timeout=settings.search_timeout_seconds,
            )
    return results


__all__ = ["BudgetedLLM", "is_transient_error", "search_with_retry"]
