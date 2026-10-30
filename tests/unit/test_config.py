"""Tests for configuration module."""
import pytest
import os

from research_agent.config import Settings, load_settings, validate_settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.llm_provider == 'openai'
        assert s.max_concurrent_searches == 3
        assert s.confidence_threshold == 0.3
    
    def test_custom_values(self):
        s = Settings(llm_model='gpt-4', max_tool_calls_per_run=50)
        assert s.llm_model == 'gpt-4'
        assert s.max_tool_calls_per_run == 50


class TestValidation:
    def test_missing_openai_key(self):
        s = Settings(llm_provider='openai', openai_api_key=None)
        errors = validate_settings(s, raise_error=False)
        assert any('OPENAI_API_KEY' in e for e in errors)
    
    def test_missing_search_keys_ok_with_free_search(self):
        s = Settings(tavily_api_key=None, brave_api_key=None)
        errors = validate_settings(s, raise_error=False)
        assert not any('search' in e.lower() for e in errors)

    def test_missing_search_keys_and_free_search_disabled(self):
        s = Settings(tavily_api_key=None, brave_api_key=None, enable_free_search=False)
        errors = validate_settings(s, raise_error=False)
        assert any('search' in e.lower() for e in errors)
    
    def test_valid_config(self):
        s = Settings(
            openai_api_key='test',
            tavily_api_key='test',
        )
        errors = validate_settings(s, raise_error=False)
        assert len(errors) == 0
