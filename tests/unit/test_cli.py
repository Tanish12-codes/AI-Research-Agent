"""Tests for CLI provider initialization and commands."""

import pytest
from click.testing import CliRunner

from research_agent.cli import _create_providers, cli
from research_agent.config import Settings
from research_agent.llm import (
    AnthropicProvider,
    GeminiProvider,
    MockLLMProvider,
    OpenAIProvider,
)
from research_agent.tools.search import BraveSearchProvider, MockSearchProvider, TavilySearchProvider


class TestCliProviderInitialization:
    """Test _create_providers() under different provider configurations."""

    def test_cli_provider_openai(self):
        settings = Settings(
            llm_provider="openai",
            openai_api_key="sk-test-key",
            llm_model="gpt-4o-mini",
            tavily_api_key="tvly-test",
        )
        llm, search = _create_providers(settings, offline=False)
        assert isinstance(llm, OpenAIProvider)
        assert llm.name == "openai"
        assert llm.model_name == "gpt-4o-mini"
        assert isinstance(search.primary, TavilySearchProvider)

    def test_cli_provider_gemini(self):
        settings = Settings(
            llm_provider="gemini",
            gemini_api_key="gemini-test-key",
            llm_model="gemini-1.5-flash",
            tavily_api_key="tvly-test",
        )
        llm, search = _create_providers(settings, offline=False)
        assert isinstance(llm, GeminiProvider)
        assert llm.name == "gemini"
        assert llm.model_name == "gemini-1.5-flash"
        assert isinstance(search.primary, TavilySearchProvider)

    def test_cli_provider_anthropic(self):
        settings = Settings(
            llm_provider="anthropic",
            anthropic_api_key="anthropic-test-key",
            llm_model="claude-3-5-sonnet-20241022",
            tavily_api_key="tvly-test",
        )
        llm, search = _create_providers(settings, offline=False)
        assert isinstance(llm, AnthropicProvider)
        assert llm.name == "anthropic"
        assert llm.model_name == "claude-3-5-sonnet-20241022"
        assert isinstance(search.primary, TavilySearchProvider)

    def test_cli_provider_offline_mode(self):
        # Offline mode should succeed with no API keys configured at all
        settings = Settings(
            llm_provider="openai",
            openai_api_key=None,
            tavily_api_key=None,
        )
        llm, search = _create_providers(settings, offline=True)
        assert isinstance(llm, MockLLMProvider)
        assert llm.name == "mock"
        assert isinstance(search.primary, MockSearchProvider)

    def test_cli_provider_missing_openai_key(self, capsys):
        settings = Settings(
            llm_provider="openai",
            openai_api_key=None,
            tavily_api_key="tvly-test",
        )
        with pytest.raises(SystemExit) as exc_info:
            _create_providers(settings, offline=False)
        assert exc_info.value.code == 1

    def test_cli_provider_missing_gemini_key(self, capsys):
        settings = Settings(
            llm_provider="gemini",
            gemini_api_key=None,
            tavily_api_key="tvly-test",
        )
        with pytest.raises(SystemExit) as exc_info:
            _create_providers(settings, offline=False)
        assert exc_info.value.code == 1

    def test_cli_provider_missing_anthropic_key(self, capsys):
        settings = Settings(
            llm_provider="anthropic",
            anthropic_api_key=None,
            tavily_api_key="tvly-test",
        )
        with pytest.raises(SystemExit) as exc_info:
            _create_providers(settings, offline=False)
        assert exc_info.value.code == 1

    def test_cli_provider_unsupported(self, capsys):
        settings = Settings(
            llm_provider="unsupported_xyz",
            tavily_api_key="tvly-test",
        )
        with pytest.raises(SystemExit) as exc_info:
            _create_providers(settings, offline=False)
        assert exc_info.value.code == 1


class TestCliSearchProviderInitialization:
    """Test search provider initialization logic in _create_providers()."""

    def test_tavily_only(self):
        settings = Settings(
            llm_provider="mock",
            tavily_api_key="tavily-key",
            brave_api_key=None,
            enable_free_search=False,
        )
        _, search = _create_providers(settings, offline=False)
        assert isinstance(search.primary, TavilySearchProvider)
        assert search.fallback is None

    def test_brave_only(self):
        settings = Settings(
            llm_provider="mock",
            tavily_api_key=None,
            brave_api_key="brave-key",
            enable_free_search=False,
        )
        _, search = _create_providers(settings, offline=False)
        assert isinstance(search.primary, BraveSearchProvider)
        assert search.fallback is None

    def test_tavily_primary_brave_fallback(self):
        settings = Settings(
            llm_provider="mock",
            tavily_api_key="tavily-key",
            brave_api_key="brave-key",
            enable_free_search=False,
        )
        _, search = _create_providers(settings, offline=False)
        assert isinstance(search.primary, TavilySearchProvider)
        assert isinstance(search.fallback, BraveSearchProvider)

    def test_no_search_keys_exits(self):
        settings = Settings(
            llm_provider="mock",
            tavily_api_key=None,
            brave_api_key=None,
            enable_free_search=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _create_providers(settings, offline=False)
        assert exc_info.value.code == 1


class TestCliCommands:
    """Test that existing CLI commands work."""

    def test_help_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "research" in result.output
        assert "baseline" in result.output
        assert "validate" in result.output
        assert "evaluate" in result.output

    def test_validate_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["validate"])
        # Should run and display configuration output
        assert result.exit_code == 0

    def test_baseline_offline_command(self, monkeypatch):
        from research_agent.schemas import FinalAnswer
        fake_answer = FinalAnswer(
            question="What is Python?",
            short_answer="Python is a programming language.",
            key_findings=["Finding 1"],
            source_links=[],
            confidence_score=0.8,
            confidence_explanation="High confidence",
            execution_status="completed",
        )
        def fake_run(coro):
            coro.close()
            return fake_answer
        monkeypatch.setattr("research_agent.cli.asyncio.run", fake_run)

        runner = CliRunner()
        result = runner.invoke(cli, ["baseline", "What is Python?", "--offline"])
        assert result.exit_code == 0
        assert "Baseline" in result.output
        assert "Status: completed" in result.output
        assert "Python is a programming language." in result.output

    def test_research_offline_command(self, monkeypatch):
        from research_agent.schemas import FinalAnswer
        fake_answer = FinalAnswer(
            question="What is Python?",
            short_answer="Python is a programming language.",
            key_findings=["Finding 1"],
            source_links=[],
            confidence_score=0.8,
            confidence_explanation="High confidence",
            execution_status="completed",
        )
        def fake_run(coro):
            coro.close()
            return fake_answer
        monkeypatch.setattr("research_agent.cli.asyncio.run", fake_run)

        runner = CliRunner()
        result = runner.invoke(cli, ["research", "What is Python?", "--offline"])
        assert result.exit_code == 0
        assert "Status: completed" in result.output
        assert "Result saved to" in result.output
