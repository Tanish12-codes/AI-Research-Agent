"""Tests for the free / keyless providers: OpenAI-compatible LLMs, failover, DuckDuckGo, Wikipedia."""

import pytest

from research_agent.config import Settings, load_settings
from research_agent.llm import (
    FallbackLLMProvider,
    GeminiProvider,
    LLMResponse,
    MockLLMProvider,
    OpenAIProvider,
    RateLimitError,
    BadRequestError,
    ConfigurationError,
    create_llm_provider,
    _strip_code_fences,
)
from research_agent.schemas import Evidence
from research_agent.tools.search import (
    DuckDuckGoSearchProvider,
    MockSearchProvider,
    SearchWithFallback,
    TavilySearchProvider,
    WikipediaSearchProvider,
    build_search,
)


class _Failing(MockLLMProvider):
    def __init__(self, exc: Exception, label: str):
        super().__init__()
        self._exc = exc
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    async def complete(self, messages, temperature=0.0, max_tokens=2000, response_format=None):
        raise self._exc


class TestLLMFactory:
    def test_groq_uses_openai_compatible_endpoint(self):
        llm = create_llm_provider(Settings(llm_provider="groq", llm_model="", groq_api_key="k"))
        assert isinstance(llm, FallbackLLMProvider)
        assert [p.model_name for p in llm.providers] == [
            "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b",
        ]
        first = llm.providers[0]
        assert isinstance(first, OpenAIProvider)
        assert first.name == "groq"
        assert first.base_url == "https://api.groq.com/openai/v1"
        assert first.extra_body == {"reasoning_effort": "low"}
        assert llm.providers[2].extra_body == {}

    def test_openai_compatible_404_is_failover_able(self):
        from research_agent.llm import ModelUnavailableError

        class NotFoundError(Exception):
            status_code = 404

        p = OpenAIProvider(api_key="k", model="gone", provider_name="groq")
        assert isinstance(p._normalize_error(NotFoundError("model_not_found")), ModelUnavailableError)

    def test_openrouter_disables_native_json_mode(self):
        llm = create_llm_provider(Settings(llm_provider="openrouter", llm_model="", openrouter_api_key="k"))
        assert llm.name == "openrouter"
        assert llm.supports_json_mode is False
        assert llm.model_name.endswith(":free")

    def test_ollama_needs_no_key(self):
        llm = create_llm_provider(Settings(llm_provider="ollama", llm_model=""))
        assert llm.name == "ollama"
        assert llm.base_url == "http://localhost:11434/v1"

    def test_missing_groq_key_names_signup_url(self):
        with pytest.raises(ConfigurationError, match="console.groq.com"):
            create_llm_provider(Settings(llm_provider="groq", llm_model=""))

    def test_openai_default_model_does_not_leak_to_gemini(self):
        llm = create_llm_provider(Settings(llm_provider="gemini", llm_model="gpt-4o-mini", gemini_api_key="k"))
        assert isinstance(llm, FallbackLLMProvider)
        assert all(isinstance(p, GeminiProvider) for p in llm.providers)
        assert llm.model_name == "gemini-2.5-flash"

    def test_explicit_gemini_model_is_not_chained(self):
        llm = create_llm_provider(Settings(llm_provider="gemini", llm_model="gemini-2.5-flash", gemini_api_key="k"))
        assert isinstance(llm, GeminiProvider)
        assert llm.model_name == "gemini-2.5-flash"

    def test_gemini_models_setting_controls_chain(self):
        llm = create_llm_provider(
            Settings(llm_provider="gemini", llm_model="", gemini_api_key="k", gemini_models=["a", "b"])
        )
        assert [p.model_name for p in llm.providers] == ["a", "b"]

    def test_auto_orders_free_providers_first(self):
        llm = create_llm_provider(
            Settings(llm_provider="auto", openai_api_key="o", groq_api_key="g", gemini_api_key="m")
        )
        assert isinstance(llm, FallbackLLMProvider)
        names = [p.name for p in llm.providers]
        # Groq first (largest free daily quota), then every Gemini model, then paid.
        assert names == ["groq"] * 3 + ["gemini"] * 3 + ["openai"]

    def test_auto_with_single_provider_is_unwrapped(self):
        llm = create_llm_provider(Settings(llm_provider="auto", openai_api_key="o"))
        assert isinstance(llm, OpenAIProvider)

    def test_auto_with_nothing_configured_errors(self):
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            create_llm_provider(Settings(llm_provider="auto"))

    def test_load_settings_defaults_to_auto(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        s = load_settings(tmp_path / "missing.env")
        assert s.llm_provider == "auto"
        assert s.enable_free_search is True


class TestFallbackLLM:
    @pytest.mark.asyncio
    async def test_fails_over_on_rate_limit(self):
        ok = MockLLMProvider(default_response='{"ok": true}')
        llm = FallbackLLMProvider([_Failing(RateLimitError("quota"), "gemini"), ok])
        resp = await llm.complete([{"role": "user", "content": "hi"}])
        assert isinstance(resp, LLMResponse)
        assert resp.content == '{"ok": true}'
        assert llm.name == "mock"
        assert llm.failovers == ["gemini: RateLimitError"]

    @pytest.mark.asyncio
    async def test_fails_over_on_retired_model(self):
        from research_agent.llm import ModelUnavailableError
        llm = FallbackLLMProvider([_Failing(ModelUnavailableError("404"), "gemini"), MockLLMProvider()])
        assert (await llm.complete([{"role": "user", "content": "hi"}])).content

    @pytest.mark.asyncio
    async def test_bad_request_is_not_failed_over(self):
        llm = FallbackLLMProvider([_Failing(BadRequestError("bad"), "gemini"), MockLLMProvider()])
        with pytest.raises(BadRequestError):
            await llm.complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_all_providers_failing_raises_last_error(self):
        llm = FallbackLLMProvider([
            _Failing(RateLimitError("a"), "gemini"),
            _Failing(RateLimitError("b"), "groq"),
        ])
        with pytest.raises(RateLimitError, match="b"):
            await llm.complete([{"role": "user", "content": "hi"}])

    def test_strip_code_fences(self):
        assert _strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


DDG_HTML = """
<div class="result results_links result--ad"><a class="result__a" href="https://ads.example.com">Ad</a>
  <a class="result__snippet">Buy now</a></div>
<div class="result results_links">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fqdrant.tech%2Fdocs%2F&rut=x">Qdrant Docs</a>
  <a class="result__snippet">Qdrant is an open-source vector database written in Rust.</a>
</div>
<div class="result results_links">
  <a class="result__a" href="https://milvus.io/">Milvus</a>
  <a class="result__snippet">Milvus is a cloud-native vector database.</a>
</div>
<div class="result results_links"><a class="result__a" href="https://empty.example">No snippet</a></div>
"""


class TestDuckDuckGoParsing:
    def test_parses_results_and_unwraps_redirects(self):
        results = DuckDuckGoSearchProvider.parse_html(DDG_HTML, "vector db")
        assert [r.url for r in results] == ["https://qdrant.tech/docs/", "https://milvus.io/"]
        assert results[0].domain == "qdrant.tech"
        assert results[0].provider == "duckduckgo"
        assert "Rust" in results[0].text

    def test_skips_ads_and_respects_max_results(self):
        results = DuckDuckGoSearchProvider.parse_html(DDG_HTML, "q", max_results=1)
        assert len(results) == 1
        assert "ads.example.com" not in results[0].url

    def test_challenge_page_yields_no_results(self):
        assert DuckDuckGoSearchProvider.parse_html("<html><body>Please verify</body></html>", "q") == []


class TestWikipediaParsing:
    def test_parses_generator_search_response(self):
        data = {"query": {"pages": [
            {"index": 2, "title": "Milvus", "fullurl": "https://en.wikipedia.org/wiki/Milvus", "extract": "Milvus..."},
            {"index": 1, "title": "Vector database", "fullurl": "https://en.wikipedia.org/wiki/Vector_database",
             "extract": "A vector database stores embeddings."},
            {"index": 3, "title": "Empty", "fullurl": "https://en.wikipedia.org/wiki/Empty", "extract": ""},
        ]}}
        results = WikipediaSearchProvider.parse_response(data, "vector database")
        assert [r.title for r in results] == ["Vector database", "Milvus"]
        assert results[0].provider == "wikipedia"
        assert results[0].domain == "en.wikipedia.org"

    def test_empty_response(self):
        assert WikipediaSearchProvider.parse_response({}, "q") == []


def _ev(eid: str) -> Evidence:
    from datetime import datetime, timezone
    return Evidence(
        evidence_id=eid, url=f"https://example.com/{eid}", domain="example.com", title=eid,
        text="text", retrieval_timestamp=datetime.now(timezone.utc), provider="mock",
    )


class _Raising(MockSearchProvider):
    @property
    def name(self) -> str:
        return "broken"

    async def search(self, query, max_results=5):
        raise RuntimeError("down")


class TestSearchChain:
    @pytest.mark.asyncio
    async def test_empty_results_fall_through(self):
        second = MockSearchProvider({"q": [_ev("a")]})
        chain = SearchWithFallback(MockSearchProvider(), second)
        results = await chain.search("q")
        assert [r.evidence_id for r in results] == ["a"]

    @pytest.mark.asyncio
    async def test_errors_fall_through_to_extra_fallbacks(self):
        third = MockSearchProvider({"q": [_ev("c")]})
        chain = SearchWithFallback(_Raising(), _Raising(), [third])
        assert (await chain.search("q"))[0].evidence_id == "c"

    @pytest.mark.asyncio
    async def test_all_raising_reraises(self):
        with pytest.raises(RuntimeError):
            await SearchWithFallback(_Raising(), _Raising()).search("q")

    @pytest.mark.asyncio
    async def test_all_empty_returns_empty(self):
        assert await SearchWithFallback(MockSearchProvider(), MockSearchProvider()).search("q") == []

    def test_build_search_free_only(self):
        chain = build_search(Settings())
        assert [p.name for p in chain.providers] == ["duckduckgo", "wikipedia"]

    def test_build_search_keyed_first(self):
        chain = build_search(Settings(tavily_api_key="t"))
        assert isinstance(chain.primary, TavilySearchProvider)
        assert [p.name for p in chain.providers] == ["tavily", "duckduckgo", "wikipedia"]

    def test_build_search_nothing_available(self):
        assert build_search(Settings(enable_free_search=False)) is None


class _Hanging(MockLLMProvider):
    @property
    def name(self) -> str:
        return "slow"

    async def complete(self, messages, temperature=0.0, max_tokens=2000, response_format=None):
        import asyncio
        await asyncio.sleep(10)


class TestFallbackTimeouts:
    @pytest.mark.asyncio
    async def test_hung_provider_times_out_and_fails_over(self):
        llm = FallbackLLMProvider([_Hanging(), MockLLMProvider(default_response="ok")], per_provider_timeout=0.05)
        resp = await llm.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "ok"
        assert llm.failovers == ["slow: APITimeoutError"]
        assert llm.total_timeout == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_sticks_with_last_working_provider(self):
        first = _Failing(RateLimitError("quota"), "gemini")
        llm = FallbackLLMProvider([first, MockLLMProvider(default_response="ok")])
        await llm.complete([{"role": "user", "content": "a"}])
        await llm.complete([{"role": "user", "content": "b"}])
        assert llm.failovers == ["gemini: RateLimitError"]


class TestGroqJsonValidateRetry:
    @pytest.mark.asyncio
    async def test_json_validate_failed_retries_without_native_json_mode(self):
        from types import SimpleNamespace

        calls = []

        class FakeCompletions:
            async def create(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise Exception("Error code: 400 - {'code': 'json_validate_failed'}")
                msg = SimpleNamespace(content='```json\n{"claims": []}\n```')
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5), model="m",
                )

        p = OpenAIProvider(api_key="k", model="m", provider_name="groq")
        p._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        resp = await p.complete([{"role": "user", "content": "x"}], response_format={"type": "json_object"})
        assert resp.content == '{"claims": []}'
        assert "response_format" in calls[0] and "response_format" not in calls[1]
        assert calls[1]["messages"][0]["role"] == "system"
