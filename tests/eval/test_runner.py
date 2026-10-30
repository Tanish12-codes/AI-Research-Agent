"""Tests for the offline evaluation harness."""
import pytest

from research_agent.config import Settings
from research_agent.evaluation.dataset import load_eval_datasets
from research_agent.evaluation.runner import EvaluationRunner
from research_agent.planner import create_plan


@pytest.mark.asyncio
async def test_mock_search_matches_every_planned_query(tmp_path):
    """Every answerable query's planned searches must hit its mock fixture."""
    dataset = load_eval_datasets()
    runner = EvaluationRunner(settings=Settings(llm_provider='mock'), offline=True, output_dir=tmp_path)
    llm, search = runner._setup_mock_providers(dataset)

    for q in dataset.queries:
        if not q.is_answerable:
            continue
        plan = await create_plan(q.question, llm)
        for step in plan.steps:
            for query in step.search_queries:
                assert await search.search(query), f'No mock evidence for {q.id}: {query!r}'


@pytest.mark.asyncio
async def test_offline_eval_makes_no_network_calls(tmp_path, monkeypatch):
    """Offline evaluation must be hermetic: any real HTTP request fails the test."""
    attempted = []

    async def no_network(self, request, *args, **kwargs):
        attempted.append(str(request.url))
        raise AssertionError(f'network call in offline eval: {request.url}')

    monkeypatch.setattr('httpx.AsyncClient.send', no_network)
    monkeypatch.chdir(tmp_path)

    runner = EvaluationRunner(settings=Settings(llm_provider='mock'), offline=True, output_dir=tmp_path)
    result = await runner.run_evaluation()

    assert attempted == []
    assert result.metrics.abstention_rate_on_answerable == 0.0
