"""Evaluation runner orchestrating benchmark suites and generating comparative reports."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..cache import FileCache
from ..llm import LLMProvider, MockLLMProvider, create_llm_provider
from ..pipeline import ResearchTools, run_baseline_no_tools, run_research_pipeline
from ..schemas import Evidence, FinalAnswer, QuestionType
from ..security import analyze_content_security, validate_url
from ..tools.search import MockSearchProvider, SearchWithFallback, build_search
from .dataset import AttackTestCase, BenignControlCase, EvalDataset, QueryTestCase, load_eval_datasets
from .metrics import EvaluationMetrics, calculate_run_metrics, generate_markdown_summary


async def _no_repos(*_args: Any, **_kwargs: Any) -> list[Evidence]:
    return []


async def _no_page(*_args: Any, **_kwargs: Any) -> Optional[Evidence]:
    return None


OFFLINE_TOOLS = ResearchTools(fetch_repos=_no_repos, fetch_page=_no_page, search_papers=_no_repos)


class EvalRunResult(BaseModel):
    """Aggregate output from an evaluation run."""

    metrics: EvaluationMetrics
    raw_query_results: list[dict[str, Any]] = Field(default_factory=list)
    raw_redteam_results: dict[str, Any] = Field(default_factory=dict)
    raw_baseline_results: list[dict[str, Any]] = Field(default_factory=list)
    markdown_summary: str = ""


class EvaluationRunner:
    """Orchestrates comparative evaluations for agent pipeline vs baseline."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        offline: bool = True,
        output_dir: Optional[str | Path] = None,
    ):
        self.settings = settings or get_settings()
        self.offline = offline
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.output_dir = Path(output_dir) if output_dir else root_dir / "eval_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_mock_providers(self, dataset: EvalDataset) -> tuple[MockLLMProvider, SearchWithFallback]:
        """Configure mock LLM and search with rich fixtures for offline evaluation."""
        mock_search = MockSearchProvider()
        mock_llm = MockLLMProvider()

        for q in dataset.queries:
            search_query = q.question[:60].strip()
            key = search_query.lower()
            if not q.is_answerable:
                mock_search.add_response(key, [])
                continue

            if q.is_low_quality:
                evidence = [
                    Evidence(
                        evidence_id=f"ev-low-{q.id}-1",
                        url="https://shaky-rumors.blogspot.com/post-1",
                        domain="blogspot.com",
                        title="Unverified Rumors",
                        text="Unverified claim without scientific consensus or experimental data.",
                        retrieval_timestamp=datetime.now(timezone.utc),
                        provider="mock",
                        relevance_score=0.4,
                        authority_score=0.2,
                        trust_level=0.5,
                    )
                ]
            else:
                entities_str = " and ".join(q.target_entities) if q.target_entities else "the topic"
                evidence = [
                    Evidence(
                        evidence_id=f"ev-high-{q.id}-1",
                        url=f"https://docs.official-source.org/{q.id}",
                        domain="official-source.org",
                        title=f"Official Documentation on {entities_str}",
                        text=f"Official verified report covering {entities_str}. Extracted facts: {'; '.join(q.expected_findings)} in 2024.",
                        retrieval_timestamp=datetime.now(timezone.utc),
                        provider="mock",
                        relevance_score=0.95,
                        authority_score=0.90,
                        trust_level=1.0,
                    ),
                    Evidence(
                        evidence_id=f"ev-high-{q.id}-2",
                        url=f"https://github.com/org/{q.id}-repo",
                        domain="github.com",
                        title=f"GitHub Repo for {entities_str}",
                        text=f"Repository for {entities_str}. Description: {entities_str} implementations with 15000 stars and Apache-2.0 license.",
                        retrieval_timestamp=datetime.now(timezone.utc),
                        provider="github",
                        relevance_score=0.90,
                        authority_score=0.85,
                        trust_level=1.0,
                    ),
                ]
            mock_search.add_response(key, evidence)

            plan_json = json.dumps({
                "question": q.question,
                "question_type": q.question_type.value,
                "steps": [
                    {
                        "step_id": 1,
                        "description": f"Investigate {q.question}",
                        "search_queries": [search_query],
                        "target_entities": q.target_entities,
                        "assumptions": ["Standard industry context"],
                    }
                ],
                "success_criteria": [
                    {"description": f"Find info on {e}", "required_entities": [e], "measurable": True}
                    for e in q.target_entities
                ],
                "assumptions": ["Benchmark test assumption"],
                "required_entities": q.target_entities,
                "required_aspects": q.required_aspects,
            })

            quotes = [f"Official verified report covering {entities_str}"] if not q.is_low_quality else ["Unverified claim"]
            claims_json = json.dumps({
                "claims": [
                    {
                        "claim_id": f"claim-{q.id}-1",
                        "claim_text": f"Grounded finding regarding {entities_str}.",
                        "evidence_ids": [f"ev-high-{q.id}-1"] if not q.is_low_quality else [f"ev-low-{q.id}-1"],
                        "verbatim_quotes": quotes,
                        "relevant_entities": q.target_entities,
                        "relevant_criteria": [],
                        "numeric_assertions": ["2024"] if not q.is_low_quality else [],
                    }
                ]
            })

            mock_llm.add_response(f"research plan for: {q.question.lower()}", plan_json)
            mock_llm.add_response(f"research question: {q.question.lower()}", claims_json)

        search_fallback = SearchWithFallback(mock_search)
        return mock_llm, search_fallback

    def _setup_live_providers(self) -> tuple[LLMProvider, SearchWithFallback]:
        """Configure live providers (free tiers included) from environment settings."""
        llm = create_llm_provider(self.settings)
        search = build_search(self.settings, FileCache(ttl_seconds=self.settings.cache_ttl_seconds))
        if search is None:
            raise ValueError("No search provider available: set a search key or ENABLE_FREE_SEARCH=true.")
        return llm, search

    async def run_evaluation(self, max_queries: Optional[int] = None) -> EvalRunResult:
        """Run the end-to-end evaluation suite."""
        dataset = load_eval_datasets()
        mode_label = "mock" if self.offline else "live"

        if self.offline:
            llm, search = self._setup_mock_providers(dataset)
            model_name = "mock-model"
            search_name = "mock-search"
        else:
            llm, search = self._setup_live_providers()
            model_name = f"{llm.name}/{llm.model_name}"
            search_name = " > ".join(p.name for p in search.providers)

        queries_to_run = dataset.queries[:max_queries] if max_queries else dataset.queries

        query_results: list[dict[str, Any]] = []
        for q in queries_to_run:
            ans = await run_research_pipeline(
                question=q.question,
                llm=llm,
                search=search,
                settings=self.settings,
                defenses_enabled=True,
                tools=OFFLINE_TOOLS if self.offline else None,
            )
            query_results.append({
                "test_case_id": q.id,
                "question": q.question,
                "is_answerable": q.is_answerable,
                "required_entities": q.target_entities,
                "expected_findings": q.expected_findings,
                "answer": ans,
            })

        baseline_results: list[dict[str, Any]] = []
        for q in queries_to_run:
            base_ans = await run_baseline_no_tools(
                question=q.question,
                llm=llm,
                settings=self.settings,
            )
            baseline_results.append({
                "test_case_id": q.id,
                "question": q.question,
                "answer": base_ans,
            })

        redteam_results = self._run_redteam_suite(dataset)

        metrics = calculate_run_metrics(
            query_results=query_results,
            redteam_results=redteam_results,
            baseline_results=baseline_results,
            mode=mode_label,
            model_name=model_name,
            search_provider=search_name,
        )

        md_summary = generate_markdown_summary(metrics)

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"eval_run_{mode_label}_{timestamp_str}.json"
        summary_path = self.output_dir / f"eval_summary_{mode_label}.md"

        serializable_data = {
            "metrics": metrics.model_dump(mode="json"),
            "redteam": redteam_results,
            "queries": [
                {
                    "id": r["test_case_id"],
                    "question": r["question"],
                    "status": r["answer"].execution_status,
                    "confidence": r["answer"].confidence_score,
                    "verified_count": len(r["answer"].verified_findings),
                    "rejected_count": len(r["answer"].rejected_claims),
                }
                for r in query_results
            ],
            "baseline": [
                {
                    "id": b["test_case_id"],
                    "question": b["question"],
                    "status": b["answer"].execution_status,
                    "confidence": b["answer"].confidence_score,
                }
                for b in baseline_results
            ],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, default=str)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(md_summary)

        return EvalRunResult(
            metrics=metrics,
            raw_query_results=query_results,
            raw_redteam_results=redteam_results,
            raw_baseline_results=baseline_results,
            markdown_summary=md_summary,
        )

    def _run_redteam_suite(self, dataset: EvalDataset) -> dict[str, Any]:
        """Execute red-team attacks across defenses-on vs defenses-off states."""
        total_attacks = len(dataset.attacks)
        blocked_with_defenses = 0
        blocked_without_defenses = 0
        attack_details: list[dict[str, Any]] = []

        for atk in dataset.attacks:
            if atk.category == "ssrf":
                valid_on, reason_on = validate_url(atk.payload)
                is_blocked_on = not valid_on
                is_blocked_off = False
            else:
                sec_on = analyze_content_security(
                    raw_text=atk.payload,
                    is_html=atk.is_html,
                    defenses_enabled=True,
                )
                sec_off = analyze_content_security(
                    raw_text=atk.payload,
                    is_html=atk.is_html,
                    defenses_enabled=False,
                )
                is_blocked_on = sec_on.quarantined or len(sec_on.flags) > 0 or len(sec_on.removed_sentences) > 0
                is_blocked_off = sec_off.quarantined or len(sec_off.flags) > 0

            if is_blocked_on:
                blocked_with_defenses += 1
            if is_blocked_off:
                blocked_without_defenses += 1

            attack_details.append({
                "id": atk.id,
                "name": atk.name,
                "category": atk.category,
                "blocked_defenses_on": is_blocked_on,
                "blocked_defenses_off": is_blocked_off,
            })

        benign_total = len(dataset.benign_controls)
        false_positives = 0
        for ctl in dataset.benign_controls:
            sec_ctl = analyze_content_security(
                raw_text=ctl.payload,
                is_html=ctl.is_html,
                defenses_enabled=True,
            )
            if sec_ctl.quarantined:
                false_positives += 1

        return {
            "total_attacks": total_attacks,
            "blocked_attacks": blocked_with_defenses,
            "blocked_without_defenses": blocked_without_defenses,
            "benign_controls": benign_total,
            "false_positives": false_positives,
            "attacks": attack_details,
        }
