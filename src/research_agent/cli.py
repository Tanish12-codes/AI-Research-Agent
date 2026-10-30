"""CLI for the AI Research Agent."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

from .cache import FileCache
from .config import Settings, load_settings, validate_settings
from .llm import ConfigurationError, LLMProvider, MockLLMProvider, create_llm_provider
from .pipeline import run_baseline_no_tools, run_research_pipeline
from .schemas import Evidence, FinalAnswer
from .tools.search import MockSearchProvider, SearchWithFallback, build_search
from .tracing import RunTracer
from .evaluation.runner import OFFLINE_TOOLS

# Windows consoles default to cp1252, which cannot print characters models often emit
# (e.g. U+2011 non-breaking hyphen); replace rather than crash after a finished run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

console = Console()


def _create_providers(settings: Settings, offline: bool = False) -> tuple[LLMProvider, SearchWithFallback]:
    """Create LLM and search providers based on settings."""
    if offline:
        mock_plan = {
            "question": "test",
            "question_type": "exploratory",
            "steps": [
                {
                    "step_id": 1,
                    "description": "Search for information",
                    "search_queries": ["test query"],
                    "target_entities": ["Python"],
                    "assumptions": ["offline mode"],
                }
            ],
            "success_criteria": [],
            "assumptions": ["offline mode"],
            "required_entities": ["Python"],
            "required_aspects": [],
        }
        mock_claims = {
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim_text": "Python is a high-level programming language.",
                    "evidence_ids": ["mock-1"],
                    "verbatim_quotes": ["Python is a high-level programming language"],
                    "relevant_entities": ["Python"],
                    "relevant_criteria": [],
                    "numeric_assertions": [],
                }
            ]
        }
        llm = MockLLMProvider(default_response=json.dumps(mock_plan))
        llm.add_response("synthesizer", json.dumps(mock_claims))
        llm.add_response("produce claims", json.dumps(mock_claims))
        llm.add_response("research synthesizer", json.dumps(mock_claims))
        llm.add_response(
            "training data",
            json.dumps({
                "answer": "This is an offline baseline response.",
                "key_findings": ["Offline baseline finding"],
                "confidence": "medium",
                "limitations": ["Generated in offline mock mode"],
            }),
        )

        mock_search_provider = MockSearchProvider()
        mock_search_provider.add_response(
            "",
            [
                Evidence(
                    evidence_id="mock-1",
                    url="https://docs.python.org/3/",
                    domain="docs.python.org",
                    title="Python Documentation",
                    text="Python is a high-level programming language designed for readability.",
                    retrieval_timestamp=datetime.now(timezone.utc),
                    provider="mock",
                    authority_score=0.95,
                    relevance_score=0.9,
                )
            ],
        )
        search = SearchWithFallback(mock_search_provider)
        return llm, search

    try:
        llm = create_llm_provider(settings)
    except (ConfigurationError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    cache = FileCache(ttl_seconds=settings.cache_ttl_seconds)
    search = build_search(settings, cache)
    if search is None:
        console.print(
            "[red]Error: No search provider available. Set TAVILY_API_KEY or BRAVE_API_KEY, "
            "or ENABLE_FREE_SEARCH=true.[/red]"
        )
        sys.exit(1)
    return llm, search


@click.group()
def cli() -> None:
    """AI Research Agent - Production-minded research pipeline."""
    pass


@cli.command()
@click.argument("question")
@click.option("--offline", is_flag=True, help="Use mock providers (no API keys needed)")
@click.option("--no-defenses", is_flag=True, help="Disable security defenses (for red-team testing)")
def research(question: str, offline: bool, no_defenses: bool) -> None:
    """Run a research query."""
    settings = load_settings()
    llm, search = _create_providers(settings, offline)

    tracer = RunTracer()
    cache = FileCache(ttl_seconds=settings.cache_ttl_seconds) if not offline else None

    console.print(Panel(f"[bold]Research Question:[/bold] {question}", title="AI Research Agent"))

    with console.status("Researching..."):
        answer = asyncio.run(
            run_research_pipeline(
                question=question,
                llm=llm,
                search=search,
                settings=settings,
                cache=cache,
                tracer=tracer,
                defenses_enabled=not no_defenses,
                tools=OFFLINE_TOOLS if offline else None,
            )
        )

    _display_answer(answer)

    run_id = answer.run_metadata.run_id if answer.run_metadata else "unknown"
    output_path = Path(f"eval_results/research_{run_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(answer.model_dump(mode="json"), indent=2, default=str), encoding="utf-8")
    console.print(f"\n[dim]Result saved to {output_path}[/dim]")


@cli.command()
@click.argument("question")
@click.option("--offline", is_flag=True, help="Use mock providers (no API keys needed)")
def baseline(question: str, offline: bool) -> None:
    """Run a no-tools baseline comparison."""
    settings = load_settings()
    llm, _ = _create_providers(settings, offline)

    console.print(Panel(f"[bold]Baseline (No Tools):[/bold] {question}", title="Baseline"))

    with console.status("Thinking..."):
        answer = asyncio.run(run_baseline_no_tools(question, llm, settings))

    _display_answer(answer)


@cli.command()
def validate() -> None:
    """Validate configuration."""
    settings = load_settings()
    errors = validate_settings(settings, raise_error=False)

    if errors:
        console.print("[red]Configuration Issues:[/red]")
        for err in errors:
            console.print(f"  [red][X][/red] {err}")
    else:
        console.print("[green][OK] Configuration is valid.[/green]")
        llm = create_llm_provider(settings)
        chain = getattr(llm, "providers", [llm])
        console.print("  LLM chain: " + " > ".join(f"{p.name}/{p.model_name}" for p in chain))
        search = build_search(settings)
        console.print("  Search chain: " + " > ".join(p.name for p in search.providers))


@cli.command()
@click.option("--offline/--live", default=True, help="Run evaluation in offline mock mode (default) or live mode")
@click.option("--max-queries", type=int, default=None, help="Maximum number of queries to run")
def evaluate(offline: bool, max_queries: Optional[int]) -> None:
    """Run the evaluation benchmark suite and generate metrics report."""
    from .evaluation.runner import EvaluationRunner

    settings = load_settings()
    console.print(Panel(f"[bold]Research Agent Evaluation Suite[/bold]\nMode: {'OFFLINE (Mock)' if offline else 'LIVE'}", title="Evaluation"))

    runner = EvaluationRunner(settings=settings, offline=offline)
    with console.status("Running evaluations..."):
        result = asyncio.run(runner.run_evaluation(max_queries=max_queries))

    console.print("\n[bold green][OK] Evaluation completed![/bold green]\n")
    console.print(result.markdown_summary)
    console.print(f"\n[dim]Summary saved to {runner.output_dir}/eval_summary_{'mock' if offline else 'live'}.md[/dim]")


def _display_answer(answer: FinalAnswer) -> None:
    """Display a FinalAnswer using Rich formatting."""
    status_color = {
        "completed": "green",
        "partial": "yellow",
        "abstained": "red",
        "error": "red",
    }.get(answer.execution_status, "white")

    console.print(f"\n[{status_color}]Status: {answer.execution_status}[/{status_color}]")

    if answer.abstention_reason:
        console.print(f"[red]Abstention: {answer.abstention_reason}[/red]")

    console.print(Panel(answer.short_answer, title="Answer"))

    if answer.key_findings:
        console.print("\n[bold]Key Findings:[/bold]")
        for i, finding in enumerate(answer.key_findings, 1):
            console.print(f"  {i}. {finding}")

    if answer.source_links:
        console.print("\n[bold]Sources:[/bold]")
        for link in answer.source_links:
            console.print(f"  * {link}")

    conf_color = (
        "green"
        if answer.confidence_score >= 0.7
        else "yellow"
        if answer.confidence_score >= 0.4
        else "red"
    )
    console.print(f"\n[{conf_color}]Confidence: {answer.confidence_score:.2f}[/{conf_color}]")
    console.print(f"[dim]{answer.confidence_explanation}[/dim]")

    if answer.rejected_claims:
        console.print(f"\n[yellow]Rejected Claims ({len(answer.rejected_claims)}):[/yellow]")
        for rc in answer.rejected_claims:
            console.print(f"  [X] {rc.claim.claim_text}")
            for reason in rc.rejection_reasons:
                console.print(f"    -> {reason}")

    if answer.limitations:
        console.print("\n[dim]Limitations:[/dim]")
        for lim in answer.limitations:
            console.print(f"  * {lim}")

    if answer.run_metadata:
        m = answer.run_metadata
        cost_str = f"${m.estimated_cost_usd:.4f}" if m.estimated_cost_usd is not None else "unknown"
        wall_clock_str = f"{m.wall_clock_seconds:.1f}s" if m.wall_clock_seconds is not None else "unknown"
        console.print(
            f"\n[dim]Run: {m.run_id} | {wall_clock_str} | LLM:{m.llm_calls} Tool:{m.tool_calls} | Cost: {cost_str}[/dim]"
        )


if __name__ == "__main__":
    cli()
