"""arXiv / Semantic Scholar parsing, query building, routing and pipeline integration (offline)."""

import pytest

from research_agent.tools.papers import (
    _dedupe,
    arxiv_query,
    is_academic_question,
    parse_arxiv_feed,
    parse_semantic_scholar,
    search_papers,
)

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <published>2020-05-22T17:33:37Z</published>
    <title>Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks</title>
    <summary>  Large pre-trained language models have been shown to store factual
      knowledge in their parameters.  </summary>
    <author><name>Patrick Lewis</name></author>
    <author><name>Ethan Perez</name></author>
    <author><name>Aleksandra Piktus</name></author>
    <author><name>Fabio Petroni</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/0000.00000v1</id>
    <title>No abstract</title>
    <summary></summary>
  </entry>
</feed>"""


class TestArxivParsing:
    def test_parses_entry_and_normalizes_whitespace(self):
        [ev] = parse_arxiv_feed(ARXIV_FEED, "rag")
        assert ev.url == "https://arxiv.org/abs/2005.11401v4"
        assert ev.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
        assert ev.provider == "arxiv"
        assert ev.domain == "arxiv.org"
        assert "Patrick Lewis, Ethan Perez, Aleksandra Piktus et al. (2020)" in ev.text
        assert "store factual knowledge in their parameters." in ev.text
        assert ev.publication_date.year == 2020

    def test_malformed_xml_returns_empty(self):
        assert parse_arxiv_feed("<feed><entry>", "q") == []

    def test_query_ands_specific_terms(self):
        assert arxiv_query("Compare approaches to memory in LLM agents") == "all:memory AND all:LLM AND all:agents"
        assert arxiv_query("the of") == "all:the of"


class TestSemanticScholarParsing:
    def test_parses_and_skips_papers_without_abstracts(self):
        data = {"data": [
            {"paperId": "p1", "title": "ReAct: Synergizing Reasoning and Acting", "abstract": "We explore LLMs.",
             "year": 2022, "citationCount": 3000, "venue": "ICLR", "url": "https://www.semanticscholar.org/paper/p1",
             "authors": [{"name": "Shunyu Yao"}], "externalIds": {"ArXiv": "2210.03629"}},
            {"paperId": "p2", "title": "No abstract", "abstract": None, "url": "https://x"},
        ]}
        [ev] = parse_semantic_scholar(data, "react")
        assert ev.provider == "semantic_scholar"
        assert ev.domain == "semanticscholar.org"
        assert "Citations: 3000." in ev.text  # quotable, so numeric claims can be verified
        assert ev.retrieval_metadata["arxiv_id"] == "2210.03629"

    def test_empty(self):
        assert parse_semantic_scholar({}, "q") == []


class TestRouting:
    @pytest.mark.parametrize("q", [
        "Compare different approaches to adding memory in an AI support agent.",
        "Research the pros and cons of a multi-agent architecture for customer support.",
        "What are state-of-the-art RAG retrieval methods?",
    ])
    def test_academic_questions(self, q):
        assert is_academic_question(q)

    @pytest.mark.parametrize("q", ["What is Python?", "Find 5 Indian B2B SaaS startups in HR tech."])
    def test_non_academic_questions(self, q):
        assert not is_academic_question(q)


class TestSearchPapers:
    def test_dedupe_keeps_first_copy_across_sources(self):
        a = parse_arxiv_feed(ARXIV_FEED, "q")[0]
        b = a.model_copy(update={"provider": "semantic_scholar", "evidence_id": "s2-x"})
        assert [e.provider for e in _dedupe([a, b])] == ["arxiv"]

    @pytest.mark.asyncio
    async def test_one_failing_source_is_recorded_not_fatal(self, monkeypatch):
        import research_agent.tools.papers as papers

        async def ok(q, *a, **k):
            return parse_arxiv_feed(ARXIV_FEED, q)

        async def boom(q, *a, **k):
            raise RuntimeError("429")

        monkeypatch.setattr(papers, "search_arxiv", ok)
        monkeypatch.setattr(papers, "search_semantic_scholar", boom)
        errors: list[str] = []
        results = await search_papers(["rag"], errors=errors)
        assert len(results) == 1
        assert errors and errors[0].startswith("semantic_scholar: RuntimeError")


class TestPipelineRouting:
    """The pipeline calls the papers tool only for academic questions, within budget."""

    async def _run(self, question, enable=True):
        from tests.integration.test_pipeline import _create_mock_providers
        from research_agent.evaluation.runner import OFFLINE_TOOLS
        from research_agent.pipeline import run_research_pipeline

        llm, search, settings = _create_mock_providers()
        settings = settings.model_copy(update={"enable_paper_search": enable})
        calls: list[list[str]] = []

        async def fake_papers(queries, **kwargs):
            calls.append(queries)
            return []

        tools = OFFLINE_TOOLS.__class__(
            fetch_repos=OFFLINE_TOOLS.fetch_repos, fetch_page=OFFLINE_TOOLS.fetch_page, search_papers=fake_papers,
        )
        answer = await run_research_pipeline(question=question, llm=llm, search=search, settings=settings, tools=tools)
        return answer, calls

    @pytest.mark.asyncio
    async def test_academic_question_queries_papers(self):
        _, calls = await self._run("What are the latest research approaches to Python memory management?")
        assert calls and 1 <= len(calls[0]) <= 2

    @pytest.mark.asyncio
    async def test_plain_question_skips_papers(self):
        _, calls = await self._run("What is Python?")
        assert calls == []

    @pytest.mark.asyncio
    async def test_disabled_setting_skips_papers(self):
        _, calls = await self._run("Research approaches to Python memory?", enable=False)
        assert calls == []
