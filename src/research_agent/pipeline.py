"""Main research pipeline orchestrator."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from .budgets import BudgetStatus, BudgetTracker
from .cache import FileCache
from .confidence import compute_confidence, should_abstain
from .config import Settings
from .filtering import check_coverage, filter_and_rank_evidence
from .llm import LLMProvider
from .planner import create_plan, create_replan, validate_question
from .resilience import BudgetedLLM, search_with_retry
from .schemas import (
    Evidence,
    FinalAnswer,
    RejectedClaim,
    ResearchPlan,
    RunMetadata,
    VerifiedFinding,
)
from .security import analyze_content_security
from .state import ResearchState
from .synthesizer import synthesize_claims
from .tools.fetch import fetch_page
from .tools.github import extract_github_repos, fetch_multiple_repos
from .tools.papers import is_academic_question, search_papers
from .tools.search import SearchWithFallback
from .verifier import verify_all_claims


@dataclass
class ResearchTools:
    """Network tools besides search; replaceable so offline runs stay hermetic."""

    fetch_repos: Callable[..., Awaitable[list[Evidence]]] = fetch_multiple_repos
    fetch_page: Callable[..., Awaitable[Optional[Evidence]]] = fetch_page
    search_papers: Callable[..., Awaitable[list[Evidence]]] = search_papers


async def run_research_pipeline(
    question: str,
    llm: LLMProvider,
    search: SearchWithFallback,
    settings: Settings,
    cache: Optional[FileCache] = None,
    tracer: Optional[RunTracer] = None,
    defenses_enabled: bool = True,
    tools: Optional[ResearchTools] = None,
) -> FinalAnswer:
    """Execute the complete research pipeline."""
    from .tracing import RunTracer

    run_id = str(uuid.uuid4())[:8]
    start_time = datetime.now(timezone.utc)
    state = ResearchState(run_id=run_id)
    budget = BudgetTracker(
        max_tool_calls=settings.max_tool_calls_per_run,
        max_llm_calls=settings.max_llm_calls_per_run,
        max_wall_clock=settings.max_wall_clock_seconds,
    )

    if tracer is None:
        tracer = RunTracer(run_id=run_id)

    llm = BudgetedLLM(llm, budget, state, settings, tracer)
    tools = tools or ResearchTools()

    tracer.log_event(
        "pipeline_start",
        "init",
        {
            "question": question,
            "run_id": run_id,
            "defenses_enabled": defenses_enabled,
        },
    )

    search_provider_name = getattr(search, "provider_used", "") or getattr(
        getattr(search, "primary", None), "name", "unknown"
    )

    try:
        is_valid, cleaned_question = validate_question(question)
        if not is_valid:
            return _create_error_answer(
                question, cleaned_question, start_time, run_id, state, settings, search_provider_name
            )

        tracer.log_event("question_validated", "validation", {"cleaned": cleaned_question})

        if budget.check_budget() == BudgetStatus.EXCEEDED:
            return _create_budget_exceeded_answer(
                question, start_time, run_id, state, settings, search_provider_name
            )

        plan = await create_plan(cleaned_question, llm)
        state.plan = plan
        tracer.log_plan(plan)

        all_evidence = await _execute_research(
            plan, search, settings, cache, budget, tracer, state, defenses_enabled, tools
        )

        search_provider_name = (
            getattr(search, "provider_used", "")
            or getattr(getattr(search, "primary", None), "name", "unknown")
        )

        if not all_evidence:
            tracer.log_event("no_evidence", "search", {"reason": "No evidence found"})
            return _create_abstention_answer(
                question,
                "No evidence was found for this research question.",
                start_time,
                run_id,
                state,
                settings,
                plan=plan,
                search_provider_name=search_provider_name,
            )

        secure_evidence = _apply_security(
            all_evidence, tracer, defenses_enabled
        )

        if not secure_evidence:
            return _create_abstention_answer(
                question,
                "All evidence was quarantined due to security concerns.",
                start_time,
                run_id,
                state,
                settings,
                plan=plan,
                search_provider_name=search_provider_name,
            )

        target_entities = plan.required_entities
        filtered_evidence = filter_and_rank_evidence(
            secure_evidence, cleaned_question, target_entities
        )

        for ev in filtered_evidence:
            state.add_evidence(ev)

        tracer.log_event(
            "evidence_filtered",
            "filtering",
            {
                "input_count": len(secure_evidence),
                "output_count": len(filtered_evidence),
            },
        )

        coverage_ratio, covered, missing = check_coverage(
            filtered_evidence, plan.required_entities, plan.required_aspects
        )

        tracer.log_event(
            "coverage_check",
            "coverage",
            {
                "ratio": coverage_ratio,
                "covered": covered,
                "missing": missing,
            },
        )

        if (
            missing
            and coverage_ratio < 0.7
            and budget.can_call_llm(count=2)
            and budget.can_call_tool()
        ):
            tracer.log_event("replan_triggered", "replan", {"missing": missing})

            replan = await create_replan(
                cleaned_question, missing, [], llm
            )

            additional_evidence = await _execute_research(
                replan, search, settings, cache, budget, tracer, state, defenses_enabled, tools
            )

            if additional_evidence:
                additional_secure = _apply_security(additional_evidence, tracer, defenses_enabled)
                additional_filtered = filter_and_rank_evidence(
                    additional_secure,
                    cleaned_question,
                    target_entities,
                    max_evidence=5,
                )
                filtered_evidence.extend(additional_filtered)
                for ev in additional_filtered:
                    state.add_evidence(ev)

                coverage_ratio, covered, missing = check_coverage(
                    filtered_evidence, plan.required_entities, plan.required_aspects
                )

        if not budget.can_call_llm():
            state.budget_truncations.append(
                "Synthesis skipped: LLM call or wall-clock budget exhausted"
            )
            tracer.log_event("budget_exhausted", "synthesis", {"evidence": len(filtered_evidence)})
            return _create_partial_answer(
                question,
                filtered_evidence,
                start_time,
                run_id,
                state,
                settings,
                plan=plan,
                search_provider_name=search_provider_name,
            )

        claims = await synthesize_claims(
            filtered_evidence,
            cleaned_question,
            plan.question_type,
            plan.required_entities,
            llm,
        )

        tracer.log_event(
            "claims_synthesized",
            "synthesis",
            {
                "claim_count": len(claims),
            },
        )

        evidence_map = {ev.evidence_id: ev for ev in filtered_evidence}
        verified_findings, rejected_claims = verify_all_claims(claims, evidence_map) if claims else ([], [])

        # One corrective retry when the checker struck (nearly) everything: the model
        # gets the rejection reasons and rewrites. Bounded to a single extra LLM call.
        if len(verified_findings) < 2 and budget.can_call_llm():
            first_reasons = [reason for r in rejected_claims for reason in r.rejection_reasons]
            feedback = list(dict.fromkeys(first_reasons))[:8] or [
                "No claims were produced. Produce claims that quote the evidence exactly."
            ]
            tracer.log_event(
                "synthesis_retry",
                "synthesis",
                {"first_verified": len(verified_findings), "first_rejected": len(rejected_claims)},
            )
            retry_claims = await synthesize_claims(
                filtered_evidence,
                cleaned_question,
                plan.question_type,
                plan.required_entities,
                llm,
                feedback=feedback,
            )
            if retry_claims:
                v2, r2 = verify_all_claims(retry_claims, evidence_map)
                if len(v2) > len(verified_findings):
                    seen = {f.claim.claim_text for f in v2}
                    verified_findings = v2 + [f for f in verified_findings if f.claim.claim_text not in seen]
                    rejected_claims = r2
                    claims = retry_claims
                    state.budget_truncations.append(
                        "Claims were rewritten once after the checker rejected the first attempt"
                    )

        if not claims:
            return _create_abstention_answer(
                question,
                "Could not synthesize any claims from the evidence.",
                start_time,
                run_id,
                state,
                settings,
                plan=plan,
                search_provider_name=search_provider_name,
            )

        for f in verified_findings:
            state.mark_verified(f)
        for r in rejected_claims:
            state.mark_rejected(r)

        tracer.log_event(
            "claims_verified",
            "verification",
            {
                "verified": len(verified_findings),
                "rejected": len(rejected_claims),
            },
        )

        confidence_score, confidence_explanation = compute_confidence(
            verified_findings,
            rejected_claims,
            filtered_evidence,
            coverage_ratio,
            plan.question_type,
            plan.required_entities,
        )

        should_abstain_flag, abstention_reason = should_abstain(
            confidence_score,
            settings.confidence_threshold,
            len(verified_findings),
            len(filtered_evidence),
        )

        if should_abstain_flag:
            return _create_abstention_answer(
                question,
                abstention_reason,
                start_time,
                run_id,
                state,
                settings,
                plan=plan,
                verified_findings=verified_findings,
                rejected_claims=rejected_claims,
                confidence_score=confidence_score,
                confidence_explanation=confidence_explanation,
                search_provider_name=search_provider_name,
            )

        answer = _assemble_answer(
            question=question,
            plan=plan,
            verified_findings=verified_findings,
            rejected_claims=rejected_claims,
            evidence=filtered_evidence,
            confidence_score=confidence_score,
            confidence_explanation=confidence_explanation,
            coverage_ratio=coverage_ratio,
            missing_items=missing,
            start_time=start_time,
            run_id=run_id,
            state=state,
            settings=settings,
            search_provider_name=search_provider_name,
        )

        tracer.log_event(
            "pipeline_complete",
            "finish",
            {
                "status": answer.execution_status,
                "confidence": answer.confidence_score,
                "findings": len(answer.verified_findings),
            },
        )

        return answer

    except Exception as e:
        tracer.log_error("pipeline_error", str(e))
        state.errors.append(str(e))
        return _create_error_answer(
            question,
            f"Pipeline error: {str(e) or type(e).__name__}",
            start_time,
            run_id,
            state,
            settings,
            search_provider_name=search_provider_name,
        )


async def _execute_research(
    plan: ResearchPlan,
    search: SearchWithFallback,
    settings: Settings,
    cache: Optional[FileCache],
    budget: BudgetTracker,
    tracer: RunTracer,
    state: ResearchState,
    defenses_enabled: bool,
    tools: ResearchTools,
) -> list[Evidence]:
    """Execute search queries from plan with bounded concurrency."""
    semaphore = asyncio.Semaphore(settings.max_concurrent_searches)
    all_evidence: list[Evidence] = []

    search_tasks: list[tuple[str, list[str]]] = []
    for step in plan.steps:
        for query in step.search_queries:
            search_tasks.append((query, step.target_entities))

    skipped_queries: list[str] = []

    async def bounded_search(query: str) -> list[Evidence]:
        async with semaphore:
            if not budget.can_call_tool():
                skipped_queries.append(query)
                return []
            try:
                return await search_with_retry(search, query, budget, state, settings, tracer)
            except asyncio.TimeoutError:
                tracer.log_error("search_timeout", f"Query timed out: {query}")
                return []
            except Exception as e:
                tracer.log_error("search_error", f"Query failed: {query} - {e}")
                return []

    tasks = [bounded_search(query) for query, _ in search_tasks]
    results = await asyncio.gather(*tasks)

    for result_list in results:
        all_evidence.extend(result_list)

    if skipped_queries:
        state.budget_truncations.append(
            f"{len(skipped_queries)} of {len(search_tasks)} planned searches skipped "
            "(tool-call or wall-clock budget exhausted)"
        )

    tracer.log_event(
        "searches_complete",
        "search",
        {
            "total_queries": len(search_tasks),
            "skipped_queries": len(skipped_queries),
            "total_results": len(all_evidence),
        },
    )

    urls = [ev.url for ev in all_evidence]
    github_repos = extract_github_repos(urls)

    repos_to_fetch = github_repos[: min(3, budget.remaining_tool_calls())]
    if repos_to_fetch and budget.can_call_tool(count=len(repos_to_fetch)):
        budget.use_tool_call(len(repos_to_fetch))
        state.increment_tool_calls(len(repos_to_fetch))
        github_evidence = await tools.fetch_repos(
            repos_to_fetch,
            token=settings.github_token,
            cache=cache,
            max_concurrent=settings.max_concurrent_searches,
        )
        all_evidence.extend(github_evidence)

        tracer.log_event(
            "github_fetch",
            "search",
            {
                "repos_found": len(github_repos),
                "repos_fetched": len(github_evidence),
            },
        )

    # Academic questions also query arXiv + Semantic Scholar. Routing is a deterministic
    # keyword rule on the user's question, never on retrieved text.
    if settings.enable_paper_search and is_academic_question(plan.question):
        paper_queries = [step.search_queries[0] for step in plan.steps if step.search_queries][:2]
        calls_needed = 2 * len(paper_queries)  # one arXiv + one Semantic Scholar call per query
        if paper_queries and budget.can_call_tool(count=calls_needed):
            budget.use_tool_call(calls_needed)
            state.increment_tool_calls(calls_needed)
            paper_errors: list[str] = []
            paper_evidence = await tools.search_papers(
                paper_queries,
                max_results=3,
                timeout=settings.search_timeout_seconds,
                cache=cache,
                s2_api_key=settings.semantic_scholar_api_key,
                errors=paper_errors,
            )
            all_evidence.extend(paper_evidence)
            for err in paper_errors:
                tracer.log_error("paper_search_error", err)
            tracer.log_event(
                "paper_search",
                "search",
                {"queries": paper_queries, "papers_found": len(paper_evidence), "source_errors": len(paper_errors)},
            )

    pages_to_fetch = [
        ev.url
        for ev in all_evidence
        if len(ev.text.strip()) < 100 and ev.provider != "github"
    ][: min(3, budget.remaining_tool_calls())]

    if pages_to_fetch and budget.can_call_tool(count=len(pages_to_fetch)):
        budget.use_tool_call(len(pages_to_fetch))
        state.increment_tool_calls(len(pages_to_fetch))
        fetch_tasks = [
            tools.fetch_page(url, timeout=settings.search_timeout_seconds, cache=cache)
            for url in pages_to_fetch
        ]
        fetch_results = await asyncio.gather(*fetch_tasks)

        for result in fetch_results:
            if result is not None:
                all_evidence.append(result)

    return all_evidence


def _apply_security(
    evidence_list: list[Evidence],
    tracer: RunTracer,
    defenses_enabled: bool,
) -> list[Evidence]:
    """Apply security controls to evidence. Returns non-quarantined evidence."""
    safe_evidence: list[Evidence] = []

    for ev in evidence_list:
        result = analyze_content_security(
            ev.text,
            is_html=False,
            defenses_enabled=defenses_enabled,
        )

        if result.quarantined:
            ev.is_quarantined = True
            ev.quarantine_reason = result.quarantine_reason
            ev.security_flags = result.flags
            tracer.log_security(
                "quarantined",
                {
                    "evidence_id": ev.evidence_id,
                    "reason": result.quarantine_reason,
                },
            )
            continue

        ev.text = result.cleaned_text
        ev.trust_level = max(0.0, ev.trust_level - result.trust_reduction)
        ev.security_flags = result.flags

        if result.flags:
            tracer.log_security(
                "flags_detected",
                {
                    "evidence_id": ev.evidence_id,
                    "flags": result.flags,
                    "trust_level": ev.trust_level,
                    "removed_sentences": result.removed_sentences,
                },
            )

        safe_evidence.append(ev)

    return safe_evidence


def _summarize_findings(verified_findings: list[VerifiedFinding], max_sentences: int = 3) -> str:
    """Build the short answer from verified claims only, preferring one claim per entity.

    Picking by entity keeps a comparison answer from describing only the first
    product. Claim text is used as written (no case changes), so proper nouns survive.
    """
    if not verified_findings:
        return "Insufficient evidence to provide a definitive answer."

    chosen: list[str] = []
    seen_entities: set[str] = set()
    for f in verified_findings:
        entities = {e.lower() for e in f.claim.relevant_entities}
        if entities and entities <= seen_entities:
            continue
        seen_entities |= entities
        chosen.append(f.claim.claim_text)
        if len(chosen) == max_sentences:
            break
    for f in verified_findings:
        if len(chosen) >= min(2, len(verified_findings)):
            break
        if f.claim.claim_text not in chosen:
            chosen.append(f.claim.claim_text)

    sentences = [c.strip() if c.strip().endswith((".", "!", "?")) else c.strip() + "." for c in chosen]
    return " ".join(sentences)


def _assemble_answer(
    question: str,
    plan: ResearchPlan,
    verified_findings: list[VerifiedFinding],
    rejected_claims: list[RejectedClaim],
    evidence: list[Evidence],
    confidence_score: float,
    confidence_explanation: str,
    coverage_ratio: float,
    missing_items: list[str],
    start_time: datetime,
    run_id: str,
    state: ResearchState,
    settings: Settings,
    search_provider_name: str = "unknown",
) -> FinalAnswer:
    """Assemble the final validated answer."""
    end_time = datetime.now(timezone.utc)
    wall_clock = (end_time - start_time).total_seconds()

    key_findings = [f.claim.claim_text for f in verified_findings]
    short_answer = _summarize_findings(verified_findings)

    # dict.fromkeys keeps first-seen order so output is deterministic across runs.
    source_links = list(
        dict.fromkeys(
            ev.url
            for f in verified_findings
            for ev in f.supporting_evidence
            if ev.url
        )
    )

    limitations: list[str] = []
    if missing_items:
        limitations.append(f"Missing coverage for: {', '.join(missing_items)}")
    if coverage_ratio < 1.0:
        limitations.append(f"Evidence coverage: {coverage_ratio:.0%}")
    if rejected_claims:
        limitations.append(f"{len(rejected_claims)} claim(s) rejected during verification")
    limitations.extend(state.budget_truncations)
    limitations.append("Confidence scores are heuristic estimates, not calibrated probabilities")
    limitations.append("Lexical matching does not prove semantic entailment")

    next_steps: list[str] = []
    if missing_items:
        next_steps.append(f"Search specifically for: {', '.join(missing_items[:3])}")
    if confidence_score < 0.7:
        next_steps.append("Consider additional authoritative sources")
    next_steps.append("Verify critical claims against primary sources")

    metadata = RunMetadata(
        run_id=run_id,
        start_time=start_time,
        end_time=end_time,
        wall_clock_seconds=wall_clock,
        llm_calls=state.llm_call_count,
        tool_calls=state.tool_call_count,
        total_input_tokens=state.total_input_tokens,
        total_output_tokens=state.total_output_tokens,
        retries=state.retry_count,
        estimated_cost_usd=None,
        search_provider=search_provider_name,
        llm_provider=state.llm_provider_used or settings.llm_provider,
        llm_model=state.llm_model_used or settings.llm_model,
        errors=list(state.errors),
        warnings=list(state.budget_truncations),
    )

    return FinalAnswer(
        question=question,
        short_answer=short_answer,
        key_findings=key_findings,
        source_links=source_links,
        verified_findings=verified_findings,
        rejected_claims=rejected_claims,
        confidence_score=confidence_score,
        confidence_explanation=confidence_explanation,
        assumptions=plan.assumptions,
        limitations=limitations,
        suggested_next_steps=next_steps,
        execution_status="partial" if state.budget_truncations else "completed",
        run_metadata=metadata,
    )


def _create_error_answer(
    question: str,
    error_msg: str,
    start_time: datetime,
    run_id: str,
    state: ResearchState,
    settings: Settings,
    search_provider_name: str = "unknown",
) -> FinalAnswer:
    end_time = datetime.now(timezone.utc)
    return FinalAnswer(
        question=question,
        short_answer=f"Error: {error_msg}",
        key_findings=[],
        source_links=[],
        confidence_score=0.0,
        confidence_explanation="Error occurred during research.",
        execution_status="error",
        run_metadata=RunMetadata(
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            wall_clock_seconds=(end_time - start_time).total_seconds(),
            llm_calls=state.llm_call_count,
            tool_calls=state.tool_call_count,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            retries=state.retry_count,
            errors=state.errors + [error_msg],
            search_provider=search_provider_name,
            llm_provider=state.llm_provider_used or settings.llm_provider,
            llm_model=state.llm_model_used or settings.llm_model,
        ),
    )


def _create_budget_exceeded_answer(
    question: str,
    start_time: datetime,
    run_id: str,
    state: ResearchState,
    settings: Settings,
    search_provider_name: str = "unknown",
) -> FinalAnswer:
    return _create_error_answer(
        question,
        "Budget exceeded (tool calls, LLM calls, or wall clock)",
        start_time,
        run_id,
        state,
        settings,
        search_provider_name=search_provider_name,
    )


def _create_abstention_answer(
    question: str,
    reason: str,
    start_time: datetime,
    run_id: str,
    state: ResearchState,
    settings: Settings,
    plan: Optional[ResearchPlan] = None,
    verified_findings: Optional[list[VerifiedFinding]] = None,
    rejected_claims: Optional[list[RejectedClaim]] = None,
    confidence_score: float = 0.0,
    confidence_explanation: str = "Insufficient evidence.",
    search_provider_name: str = "unknown",
) -> FinalAnswer:
    end_time = datetime.now(timezone.utc)
    return FinalAnswer(
        question=question,
        short_answer=f"Unable to provide a reliable answer: {reason}",
        key_findings=[f.claim.claim_text for f in (verified_findings or [])],
        source_links=[],
        verified_findings=verified_findings or [],
        rejected_claims=rejected_claims or [],
        confidence_score=confidence_score,
        confidence_explanation=confidence_explanation,
        assumptions=plan.assumptions if plan else [],
        limitations=[reason] + state.budget_truncations,
        suggested_next_steps=["Try rephrasing the question", "Check with additional sources"],
        execution_status="abstained",
        abstention_reason=reason,
        run_metadata=RunMetadata(
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            wall_clock_seconds=(end_time - start_time).total_seconds(),
            llm_calls=state.llm_call_count,
            tool_calls=state.tool_call_count,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            retries=state.retry_count,
            search_provider=search_provider_name,
            llm_provider=state.llm_provider_used or settings.llm_provider,
            llm_model=state.llm_model_used or settings.llm_model,
            errors=list(state.errors),
        ),
    )


def _create_partial_answer(
    question: str,
    evidence: list[Evidence],
    start_time: datetime,
    run_id: str,
    state: ResearchState,
    settings: Settings,
    plan: Optional[ResearchPlan] = None,
    search_provider_name: str = "unknown",
) -> FinalAnswer:
    """Return the evidence gathered so far when the budget ran out before synthesis."""
    end_time = datetime.now(timezone.utc)
    source_links = list(dict.fromkeys(ev.url for ev in evidence if ev.url))
    return FinalAnswer(
        question=question,
        short_answer=(
            f"Partial result: the run budget ran out before synthesis. "
            f"{len(source_links)} relevant source(s) were collected but not verified."
        ),
        key_findings=[],
        source_links=source_links,
        confidence_score=0.0,
        confidence_explanation="No claims were synthesized or verified.",
        assumptions=plan.assumptions if plan else [],
        limitations=state.budget_truncations
        + ["Sources are unverified leads, not verified findings"],
        suggested_next_steps=[
            "Re-run with a larger MAX_LLM_CALLS_PER_RUN or MAX_WALL_CLOCK_SECONDS",
            "Review the collected sources directly",
        ],
        execution_status="partial",
        run_metadata=RunMetadata(
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            wall_clock_seconds=(end_time - start_time).total_seconds(),
            llm_calls=state.llm_call_count,
            tool_calls=state.tool_call_count,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            retries=state.retry_count,
            search_provider=search_provider_name,
            llm_provider=state.llm_provider_used or settings.llm_provider,
            llm_model=state.llm_model_used or settings.llm_model,
            errors=list(state.errors),
            warnings=list(state.budget_truncations),
        ),
    )


async def run_baseline_no_tools(
    question: str,
    llm: LLMProvider,
    settings: Settings,
) -> FinalAnswer:
    """Answer a question with a single LLM call, no tools."""
    start_time = datetime.now(timezone.utc)
    run_id = f"baseline-{str(uuid.uuid4())[:8]}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer the question based on your training data. "
                "Be honest about uncertainty. Respond with JSON: "
                '{"answer": "your answer", "key_findings": ["finding1"], '
                '"confidence": "high/medium/low", "limitations": ["limitation1"]}'
            ),
        },
        {"role": "user", "content": question},
    ]

    try:
        response = await llm.complete(
            messages=messages,
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            data = {
                "answer": response.content,
                "key_findings": [],
                "confidence": "low",
                "limitations": ["Failed to parse structured response"],
            }

        end_time = datetime.now(timezone.utc)

        confidence_map = {"high": 0.8, "medium": 0.5, "low": 0.2}

        return FinalAnswer(
            question=question,
            short_answer=str(data.get("answer", response.content)),
            key_findings=[str(k) for k in data.get("key_findings", [])]
            if isinstance(data.get("key_findings"), list)
            else [],
            source_links=[],
            confidence_score=confidence_map.get(str(data.get("confidence", "low")).lower(), 0.3),
            confidence_explanation=(
                "Baseline: single LLM call, no search tools. "
                "Confidence is model self-reported, not evidence-based."
            ),
            assumptions=["Based solely on model training data"],
            limitations=(
                ([str(item) for item in data.get("limitations", [])]
                 if isinstance(data.get("limitations"), list) else [])
                + [
                    "No web search or verification performed",
                    "No evidence citations available",
                    "Based on training data cutoff",
                ]
            ),
            suggested_next_steps=["Run the full research pipeline for source-backed claims"],
            execution_status="completed",
            run_metadata=RunMetadata(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                wall_clock_seconds=(end_time - start_time).total_seconds(),
                llm_calls=1,
                tool_calls=0,
                total_input_tokens=response.input_tokens,
                total_output_tokens=response.output_tokens,
                llm_provider=llm.name,
                llm_model=llm.model_name,
            ),
        )
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        return FinalAnswer(
            question=question,
            short_answer=f"Baseline error: {e}",
            key_findings=[],
            source_links=[],
            confidence_score=0.0,
            confidence_explanation="Error in baseline.",
            execution_status="error",
            run_metadata=RunMetadata(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                wall_clock_seconds=(end_time - start_time).total_seconds(),
                llm_calls=1,
                tool_calls=0,
                errors=[str(e)],
                llm_provider=llm.name,
                llm_model=llm.model_name,
            ),
        )


__all__ = [
    "ResearchTools",
    "run_research_pipeline",
    "run_baseline_no_tools",
]
