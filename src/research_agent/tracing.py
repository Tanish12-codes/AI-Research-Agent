"""Structured observability and JSONL trace logging for research agent runs."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Optional
from pydantic import BaseModel, Field

from .schemas import Claim, Evidence, ResearchPlan


class TraceEvent(BaseModel):
    """An individual event logged in the execution trace."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the event",
    )
    event_type: str = Field(description="Type/category of the event, e.g. 'plan', 'search', 'claim'")
    stage: str = Field(description="Pipeline stage where event occurred, e.g. 'planning', 'retrieval'")
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload data")
    duration_ms: Optional[float] = Field(default=None, description="Optional execution duration in milliseconds")


SENSITIVE_KEYWORDS = ("key", "token", "secret", "password", "auth", "credential")


def redact_sensitive(obj: Any) -> Any:
    """Recursively redacts values of keys containing sensitive keywords."""
    if isinstance(obj, dict):
        redacted_dict: dict[str, Any] = {}
        for k, v in obj.items():
            k_lower = str(k).lower()
            if any(keyword in k_lower for keyword in SENSITIVE_KEYWORDS):
                redacted_dict[k] = "[REDACTED]"
            else:
                redacted_dict[k] = redact_sensitive(v)
        return redacted_dict
    elif isinstance(obj, list):
        return [redact_sensitive(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(redact_sensitive(item) for item in obj)
    return obj


def _to_serializable(data: Any) -> Any:
    """Convert Pydantic models or complex objects into JSON-serializable structures."""
    if hasattr(data, "model_dump"):
        return data.model_dump(mode="json")
    if isinstance(data, datetime):
        return data.isoformat()
    if isinstance(data, (dict, list, str, int, float, bool, type(None))):
        return data
    if isinstance(data, Exception):
        return {"error_type": type(data).__name__, "message": str(data)}
    return str(data)


class RunTracer:
    """Writes execution events into a JSONL trace file with automatic redaction."""

    def __init__(self, run_id: Optional[str] = None, traces_dir: str | Path = "traces") -> None:
        """Initialize tracer for a run."""
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.trace_file = self.traces_dir / f"trace_{self.run_id}_{now_str}.jsonl"
        self._lock = threading.Lock()

    def log_event(
        self,
        event_type: str,
        stage: str,
        data: dict[str, Any],
        duration_ms: Optional[float] = None,
    ) -> TraceEvent:
        """Log a generic event to the trace file with sensitive fields redacted."""
        clean_data = redact_sensitive(_to_serializable(data))
        if not isinstance(clean_data, dict):
            clean_data = {"payload": clean_data}

        event = TraceEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            stage=stage,
            data=clean_data,
            duration_ms=duration_ms,
        )

        line = event.model_dump_json()
        with self._lock:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        return event

    def log_plan(
        self,
        arg1: Any,
        arg2: Any = None,
        duration_ms: Optional[float] = None,
    ) -> TraceEvent:
        """Log a research plan generation or modification event."""
        if arg2 is None:
            stage = "planning"
            plan = arg1
        else:
            stage = str(arg1)
            plan = arg2
        plan_dict = plan.model_dump(mode="json") if hasattr(plan, "model_dump") else plan
        return self.log_event(
            event_type="plan",
            stage=stage,
            data={"plan": plan_dict},
            duration_ms=duration_ms,
        )

    def log_search(
        self,
        stage: str,
        query: str,
        results_count: int,
        metadata: Optional[dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> TraceEvent:
        """Log an external search tool execution."""
        data: dict[str, Any] = {
            "query": query,
            "results_count": results_count,
            "metadata": metadata or {},
        }
        return self.log_event(
            event_type="search",
            stage=stage,
            data=data,
            duration_ms=duration_ms,
        )

    def log_evidence(
        self,
        stage: str,
        evidence: Evidence | dict[str, Any],
        duration_ms: Optional[float] = None,
    ) -> TraceEvent:
        """Log retrieved and processed evidence."""
        ev_dict = evidence.model_dump(mode="json") if hasattr(evidence, "model_dump") else evidence
        return self.log_event(
            event_type="evidence",
            stage=stage,
            data={"evidence": ev_dict},
            duration_ms=duration_ms,
        )

    def log_claim(
        self,
        stage: str,
        claim: Claim | dict[str, Any],
        duration_ms: Optional[float] = None,
    ) -> TraceEvent:
        """Log an extracted candidate claim."""
        cl_dict = claim.model_dump(mode="json") if hasattr(claim, "model_dump") else claim
        return self.log_event(
            event_type="claim",
            stage=stage,
            data={"claim": cl_dict},
            duration_ms=duration_ms,
        )

    def log_verification(
        self,
        stage: str,
        finding_or_rejection: dict[str, Any] | Any,
        duration_ms: Optional[float] = None,
    ) -> TraceEvent:
        """Log claim verification or rejection outcome."""
        verif_dict = (
            finding_or_rejection.model_dump(mode="json")
            if hasattr(finding_or_rejection, "model_dump")
            else _to_serializable(finding_or_rejection)
        )
        return self.log_event(
            event_type="verification",
            stage=stage,
            data={"verification": verif_dict},
            duration_ms=duration_ms,
        )

    def log_error(
        self,
        stage: str,
        error: str | Exception,
        context: Optional[dict[str, Any]] = None,
    ) -> TraceEvent:
        """Log an error occurring in the execution pipeline."""
        data: dict[str, Any] = {
            "error": str(error),
            "context": context or {},
        }
        if isinstance(error, Exception):
            data["exception_class"] = type(error).__name__
        return self.log_event(
            event_type="error",
            stage=stage,
            data=data,
        )

    def log_security(
        self,
        arg1: str,
        arg2: Any,
        passed: Optional[bool] = None,
        details: Optional[dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> TraceEvent:
        """Log an adversarial or security check evaluation."""
        if passed is None and details is None:
            return self.log_event(
                event_type="security",
                stage="security",
                data={
                    "check_name": arg1,
                    "details": arg2 if isinstance(arg2, dict) else {"payload": arg2},
                },
                duration_ms=duration_ms,
            )
        return self.log_event(
            event_type="security",
            stage=arg1,
            data={
                "check_name": str(arg2),
                "passed": passed if passed is not None else False,
                "details": details or {},
            },
            duration_ms=duration_ms,
        )
