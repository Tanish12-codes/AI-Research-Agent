"""Research planner that creates structured research plans via LLM."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Optional

from .llm import LLMProvider, LLMResponse
from .schemas import PlanStep, QuestionType, ResearchPlan, SuccessCriterion

PLAN_SYSTEM_PROMPT = """You are a research planning assistant. Given a question, create a structured research plan.

IMPORTANT:
- You must ONLY create a plan. You do NOT have access to any search tools or evidence.
- Create 1-4 research steps, each with 1-4 search queries.
- Identify the question type: comparison, discovery, pros_cons, how_to, factual, or exploratory.
- List explicit assumptions about the question.
- Define measurable success criteria.
- Identify required entities and aspects that must be covered.

Respond with a JSON object matching this schema exactly:
{
  "question": "the original question",
  "question_type": "one of: comparison, discovery, pros_cons, how_to, factual, exploratory",
  "steps": [
    {
      "step_id": 1,
      "description": "what this step investigates",
      "search_queries": ["query 1", "query 2"],
      "target_entities": ["entity1"],
      "assumptions": ["assumption"]
    }
  ],
  "success_criteria": [
    {
      "description": "what must be found",
      "required_entities": ["entity1"],
      "required_aspects": ["aspect1"],
      "measurable": true
    }
  ],
  "assumptions": ["global assumption"],
  "required_entities": ["entity1", "entity2"],
  "required_aspects": ["aspect1", "aspect2"]
}

Be specific with search queries. Make them concrete enough to find relevant results."""

REPLAN_SYSTEM_PROMPT = """You are a research planning assistant. A previous research plan had coverage gaps.
Create a focused follow-up plan to address the missing items.

The original question, the gaps found, and any verification issues are provided.
Create 1-2 additional steps with 1-3 queries each, focused specifically on the gaps.

Respond with the same JSON schema as before."""


async def create_plan(
    question: str,
    llm: LLMProvider,
    temperature: float = 0.0,
) -> ResearchPlan:
    """Create a structured research plan for the given question."""
    messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": f"Create a research plan for: {question}"},
    ]

    response = await llm.complete(
        messages=messages,
        temperature=temperature,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    return _parse_plan_response(response.content, question)


async def create_replan(
    question: str,
    missing_items: list[str],
    verification_issues: list[str],
    llm: LLMProvider,
    temperature: float = 0.0,
) -> ResearchPlan:
    """Create a focused follow-up plan to address coverage gaps."""
    gap_info = (
        f"Original question: {question}\n"
        f"Missing coverage items: {', '.join(missing_items)}\n"
        f"Verification issues: {', '.join(verification_issues) if verification_issues else 'None'}\n"
        f"Create a focused follow-up plan to address these gaps."
    )

    messages = [
        {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
        {"role": "user", "content": gap_info},
    ]

    response = await llm.complete(
        messages=messages,
        temperature=temperature,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )

    return _parse_plan_response(response.content, question)


def _parse_plan_response(content: str, question: str) -> ResearchPlan:
    """Parse LLM response into a ResearchPlan, with fallback handling."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        json_match = re.search(r"\{[\s\S]*\}", content or "")
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return _fallback_plan(question)
        else:
            return _fallback_plan(question)

    if not isinstance(data, dict):
        return _fallback_plan(question)

    try:
        qt = data.get("question_type", "exploratory")
        try:
            question_type = QuestionType(qt)
        except ValueError:
            question_type = QuestionType.EXPLORATORY

        steps = []
        raw_steps = data.get("steps", [])
        if isinstance(raw_steps, list):
            for i, step_data in enumerate(raw_steps[:4]):
                if not isinstance(step_data, dict):
                    continue
                raw_queries = step_data.get("search_queries", [])
                queries = [
                    str(q).strip() for q in raw_queries if str(q).strip()
                ][:4]
                if not queries:
                    queries = [question.strip() or "general research"]
                steps.append(
                    PlanStep(
                        step_id=int(step_data.get("step_id", i + 1)),
                        description=str(step_data.get("description", f"Research step {i+1}")),
                        search_queries=queries,
                        target_entities=[
                            str(e) for e in step_data.get("target_entities", []) if str(e).strip()
                        ],
                        assumptions=[
                            str(a) for a in step_data.get("assumptions", []) if str(a).strip()
                        ],
                    )
                )

        if not steps:
            return _fallback_plan(question)

        criteria = []
        raw_criteria = data.get("success_criteria", [])
        if isinstance(raw_criteria, list):
            for c in raw_criteria:
                if isinstance(c, dict):
                    criteria.append(
                        SuccessCriterion(
                            description=str(c.get("description", "")),
                            required_entities=[
                                str(e) for e in c.get("required_entities", []) if str(e).strip()
                            ],
                            required_aspects=[
                                str(a) for a in c.get("required_aspects", []) if str(a).strip()
                            ],
                            measurable=bool(c.get("measurable", True)),
                        )
                    )

        return ResearchPlan(
            question=str(data.get("question", question)),
            question_type=question_type,
            steps=steps,
            success_criteria=criteria,
            assumptions=[str(a) for a in data.get("assumptions", []) if str(a).strip()],
            required_entities=[
                str(e) for e in data.get("required_entities", []) if str(e).strip()
            ],
            required_aspects=[
                str(a) for a in data.get("required_aspects", []) if str(a).strip()
            ],
        )
    except Exception:
        return _fallback_plan(question)


def _fallback_plan(question: str) -> ResearchPlan:
    """Create a minimal fallback plan when parsing fails."""
    clean_q = question.strip() if question else "General research"
    return ResearchPlan(
        question=clean_q,
        question_type=QuestionType.EXPLORATORY,
        steps=[
            PlanStep(
                step_id=1,
                description="General research on the topic",
                search_queries=[clean_q[:200] if clean_q else "general research"],
                target_entities=[],
                assumptions=["Fallback plan due to LLM parsing failure"],
            )
        ],
        success_criteria=[],
        assumptions=["Using fallback plan - LLM response could not be parsed"],
        required_entities=[],
        required_aspects=[],
    )


def validate_question(question: str) -> tuple[bool, str]:
    """Validate and sanitize user question."""
    if not question or not question.strip():
        return False, "Question cannot be empty."

    cleaned = question.strip()

    if len(cleaned) > 1000:
        return False, "Question exceeds maximum length of 1000 characters."

    if len(cleaned) < 5:
        return False, "Question is too short. Please provide a more detailed research question."

    cleaned = "".join(
        c for c in cleaned
        if not unicodedata.category(c).startswith("C") or c in "\n\t "
    )

    return True, cleaned


__all__ = [
    "PLAN_SYSTEM_PROMPT",
    "REPLAN_SYSTEM_PROMPT",
    "create_plan",
    "create_replan",
    "validate_question",
]
