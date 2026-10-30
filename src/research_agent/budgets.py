"""Budget tracking, resource constraints, and retry guardrails for research agent runs."""

from __future__ import annotations

from enum import Enum
import time
from typing import Optional

from .config import Settings


class BudgetStatus(str, Enum):
    """Resource budget utilization status."""

    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"


class BudgetTracker:
    """Tracks and enforces tool calls, LLM calls, and wall-clock execution limits."""

    def __init__(
        self,
        start_time: Optional[float] = None,
        max_tool_calls: Optional[int] = None,
        max_llm_calls: Optional[int] = None,
        max_wall_clock: Optional[float] = None,
    ) -> None:
        """Initialize tracker with zeroed call counters and current monotonic start time."""
        self.tool_calls: int = 0
        self.llm_calls: int = 0
        self.wall_clock_start: float = start_time if start_time is not None else time.monotonic()
        self.max_tool_calls: Optional[int] = max_tool_calls
        self.max_llm_calls: Optional[int] = max_llm_calls
        self.max_wall_clock: Optional[float] = max_wall_clock

    def elapsed_seconds(self) -> float:
        """Return total elapsed wall clock seconds since tracker initialization."""
        return time.monotonic() - self.wall_clock_start

    def remaining_seconds(self, settings: Optional[Settings] = None) -> float:
        """Return remaining wall-clock seconds before budget exhaustion."""
        limit = settings.max_wall_clock_seconds if settings else (self.max_wall_clock or 120.0)
        return max(0.0, limit - self.elapsed_seconds())

    def remaining_tool_calls(self, settings: Optional[Settings] = None) -> int:
        """Return number of allowable tool calls remaining."""
        limit = settings.max_tool_calls_per_run if settings else (self.max_tool_calls or 20)
        return max(0, limit - self.tool_calls)

    def remaining_llm_calls(self, settings: Optional[Settings] = None) -> int:
        """Return number of allowable LLM invocations remaining."""
        limit = settings.max_llm_calls_per_run if settings else (self.max_llm_calls or 10)
        return max(0, limit - self.llm_calls)

    def use_tool_call(self, count: int = 1) -> None:
        """Record usage of one or more tool calls."""
        self.tool_calls += count

    def use_llm_call(self, count: int = 1) -> None:
        """Record usage of one or more LLM invocations."""
        self.llm_calls += count

    def check_budget(self, settings: Optional[Settings] = None, warning_ratio: float = 0.8) -> BudgetStatus:
        """Evaluate the overall budget status against specified settings or configured limits."""
        elapsed = self.elapsed_seconds()
        max_tool = settings.max_tool_calls_per_run if settings else (self.max_tool_calls or 20)
        max_llm = settings.max_llm_calls_per_run if settings else (self.max_llm_calls or 10)
        max_clock = settings.max_wall_clock_seconds if settings else (self.max_wall_clock or 120.0)

        if (
            self.tool_calls >= max_tool
            or self.llm_calls >= max_llm
            or elapsed >= max_clock
        ):
            return BudgetStatus.EXCEEDED

        tool_warn = max_tool * warning_ratio
        llm_warn = max_llm * warning_ratio
        time_warn = max_clock * warning_ratio

        if (
            self.tool_calls >= tool_warn
            or self.llm_calls >= llm_warn
            or elapsed >= time_warn
        ):
            return BudgetStatus.WARNING

        return BudgetStatus.OK

    def can_call_tool(self, settings: Optional[Settings] = None, count: int = 1) -> bool:
        """Check if an additional tool call is permitted within budget."""
        max_tool = settings.max_tool_calls_per_run if settings else (self.max_tool_calls or 20)
        max_clock = settings.max_wall_clock_seconds if settings else (self.max_wall_clock or 120.0)
        if self.tool_calls + count > max_tool:
            return False
        if self.elapsed_seconds() >= max_clock:
            return False
        return True

    def can_call_llm(self, settings: Optional[Settings] = None, count: int = 1) -> bool:
        """Check if an additional LLM call is permitted within budget."""
        max_llm = settings.max_llm_calls_per_run if settings else (self.max_llm_calls or 10)
        max_clock = settings.max_wall_clock_seconds if settings else (self.max_wall_clock or 120.0)
        if self.llm_calls + count > max_llm:
            return False
        if self.elapsed_seconds() >= max_clock:
            return False
        return True

    def can_retry_tool(
        self,
        settings: Settings,
        expected_calls: int = 1,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """Verify whether a tool retry is safe to perform without violating budgets."""
        if not self.can_call_tool(settings, count=expected_calls):
            return False

        effective_timeout = timeout_seconds if timeout_seconds is not None else settings.search_timeout_seconds
        if self.remaining_seconds(settings) < effective_timeout:
            return False

        return True

    def can_retry_llm(
        self,
        settings: Settings,
        expected_calls: int = 1,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """Verify whether an LLM retry is safe to perform without violating budgets."""
        if not self.can_call_llm(settings, count=expected_calls):
            return False

        effective_timeout = timeout_seconds if timeout_seconds is not None else settings.llm_timeout_seconds
        if self.remaining_seconds(settings) < effective_timeout:
            return False

        return True
