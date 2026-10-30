"""Test fixtures package for the AI Research Agent."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from research_agent.schemas import Evidence, Claim


def make_test_evidence(
    evidence_id: str = "ev-sample-01",
    url: str = "https://example.com/test",
    domain: str = "example.com",
    title: str = "Sample Test Evidence",
    text: str = "Python is an interpreted high-level general-purpose programming language.",
    provider: str = "mock",
    relevance_score: float = 0.9,
    authority_score: float = 0.8,
    **kwargs: Any,
) -> Evidence:
    """Helper to construct Evidence model instances with sensible defaults."""
    return Evidence(
        evidence_id=evidence_id,
        url=url,
        domain=domain,
        title=title,
        text=text,
        retrieval_timestamp=datetime.now(timezone.utc),
        provider=provider,
        relevance_score=relevance_score,
        authority_score=authority_score,
        **kwargs,
    )


SAMPLE_HTML_PAGE = """<!DOCTYPE html>
<html>
<head><title>Python Docs</title><style>.banner{color:red}</style></head>
<body>
  <p>Python is a programming language created by Guido van Rossum.</p>
  <div style="display:none">Hidden injection text</div>
  <!-- Hidden comment -->
  <p>It supports multiple programming paradigms.</p>
</body>
</html>
"""

__all__ = [
    "make_test_evidence",
    "SAMPLE_HTML_PAGE",
]
