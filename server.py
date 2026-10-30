"""HTTP API + static web UI for the research agent.

Run with:  python server.py        (then open http://localhost:8000)

A thin layer over the same pipeline the CLI uses: it adds no research logic.
Every field the UI renders comes from the FinalAnswer, the run's JSONL trace,
or the security boundary.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from research_agent.cache import FileCache  # noqa: E402
from research_agent.cli import _create_providers  # noqa: E402
from research_agent.config import load_settings  # noqa: E402
from research_agent.evaluation.runner import OFFLINE_TOOLS  # noqa: E402
from research_agent.llm import ConfigurationError, create_llm_provider  # noqa: E402
from research_agent.pipeline import run_baseline_no_tools, run_research_pipeline  # noqa: E402
from research_agent.security import analyze_content_security  # noqa: E402
from research_agent.tools.search import build_search  # noqa: E402
from research_agent.tracing import RunTracer  # noqa: E402

app = FastAPI(title="AI Research Agent", docs_url="/api/docs")
WEB_DIR = PROJECT_ROOT / "web"


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    offline: bool = False
    defenses: bool = True
    baseline: bool = False


class SecurityRequest(BaseModel):
    text: str = Field(max_length=20000)
    is_html: bool = True


@app.get("/api/config")
def config() -> dict:
    settings = load_settings()
    try:
        llm = create_llm_provider(settings)
        chain = [f"{p.name} · {p.model_name}" for p in getattr(llm, "providers", [llm])]
        llm_error: Optional[str] = None
    except ConfigurationError as exc:
        chain, llm_error = [], str(exc)
    search = build_search(settings)
    return {
        "llm_chain": chain,
        "llm_error": llm_error,
        "search_chain": [p.name for p in search.providers] if search else [],
        "paper_search": settings.enable_paper_search,
        "budgets": {
            "tool_calls": settings.max_tool_calls_per_run,
            "llm_calls": settings.max_llm_calls_per_run,
            "wall_clock_s": settings.max_wall_clock_seconds,
        },
    }


@app.post("/api/research")
async def research(req: ResearchRequest) -> dict:
    settings = load_settings()
    try:
        llm, search = _create_providers(settings, offline=req.offline)
    except SystemExit:
        raise HTTPException(400, "Configuration error. Run `python -m src.research_agent.cli validate`.")

    tracer = RunTracer()
    question = "What is Python?" if req.offline else req.question.strip()
    answer = await run_research_pipeline(
        question=question,
        llm=llm,
        search=search,
        settings=settings,
        cache=None if req.offline else FileCache(ttl_seconds=settings.cache_ttl_seconds),
        tracer=tracer,
        defenses_enabled=req.defenses,
        tools=OFFLINE_TOOLS if req.offline else None,
    )
    baseline = None
    if req.baseline:
        baseline = (await run_baseline_no_tools(question=question, llm=llm, settings=settings)).model_dump(mode="json")

    trace = []
    if tracer.trace_file.is_file():
        trace = [json.loads(line) for line in tracer.trace_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"answer": answer.model_dump(mode="json"), "baseline": baseline, "trace": trace}


@app.post("/api/security")
def security(req: SecurityRequest) -> dict:
    result = analyze_content_security(req.text, is_html=req.is_html, defenses_enabled=True)
    return {
        "quarantined": result.quarantined,
        "flags": [list(f) if isinstance(f, tuple) else f for f in result.flags],
        "trust_reduction": result.trust_reduction,
        "cleaned_text": result.cleaned_text,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
