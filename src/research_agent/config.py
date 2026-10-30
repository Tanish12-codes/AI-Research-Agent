"""Configuration module for the AI Research Agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application configuration settings."""

    llm_provider: str = Field(default="openai", description="LLM provider name (e.g. openai)")
    llm_model: str = Field(default="gpt-4o-mini", description="Model identifier")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    gemini_models: list[str] = Field(
        default_factory=list, description="Gemini models to fail over across (empty = built-in free list)"
    )
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key (free tier)")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key (free models)")
    ollama_base_url: Optional[str] = Field(default=None, description="Ollama OpenAI-compatible URL (local)")

    tavily_api_key: Optional[str] = Field(default=None, description="Tavily search API key")
    brave_api_key: Optional[str] = Field(default=None, description="Brave search API key")
    enable_free_search: bool = Field(
        default=True, description="Fall back to keyless DuckDuckGo and Wikipedia search"
    )

    github_token: Optional[str] = Field(default=None, description="GitHub personal access token")
    enable_paper_search: bool = Field(default=True, description="Query arXiv/Semantic Scholar for academic questions")
    semantic_scholar_api_key: Optional[str] = Field(default=None, description="Optional free S2 key (higher limits)")

    max_concurrent_searches: int = Field(default=3, description="Max parallel search requests")
    search_timeout_seconds: int = Field(default=15, description="Timeout for search requests")
    llm_timeout_seconds: int = Field(default=30, description="Timeout for LLM API calls")
    cache_ttl_seconds: int = Field(default=3600, description="Default TTL for cache entries")
    max_tool_calls_per_run: int = Field(default=20, description="Budget cap for tool calls")
    max_llm_calls_per_run: int = Field(default=10, description="Budget cap for LLM invocations")
    max_wall_clock_seconds: int = Field(default=120, description="Max execution time in seconds")
    confidence_threshold: float = Field(default=0.3, description="Minimum confidence score threshold")
    max_retries: int = Field(default=2, description="Retries per LLM/search call on transient errors")
    retry_base_delay_seconds: float = Field(default=0.5, description="Initial exponential backoff delay")


def _read_env_file(filepath: Path) -> dict[str, str]:
    """Parse a simple .env file into a dictionary."""
    env_vars: dict[str, str] = {}
    if not filepath.is_file():
        return env_vars

    try:
        content = filepath.read_text(encoding="utf-8")
    except OSError:
        return env_vars

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()

        if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
            val = val[1:-1]
        elif val.startswith("#"):
            val = ""
        elif " #" in val or "\t#" in val:
            val = val.split("#", 1)[0].rstrip()

        env_vars[key] = val

    return env_vars


def load_settings(env_file: Optional[str | Path] = None) -> Settings:
    """Load settings by merging .env file values with system environment variables."""
    env_file_path = Path(env_file) if env_file else Path(".env")
    file_vars = _read_env_file(env_file_path)

    def get_val(key: str, default: Optional[str] = None) -> Optional[str]:
        if key in os.environ and os.environ[key] != "":
            return os.environ[key]
        return file_vars.get(key, default)

    def get_int(key: str, default: int) -> int:
        val = get_val(key)
        if val is None or val == "":
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def get_float(key: str, default: float) -> float:
        val = get_val(key)
        if val is None or val == "":
            return default
        try:
            return float(val)
        except ValueError:
            return default

    # Empty LLM_MODEL means "use the provider's default" (see llm.DEFAULT_MODELS).
    provider = (get_val("LLM_PROVIDER", "auto") or "auto").strip()
    free_search = (get_val("ENABLE_FREE_SEARCH", "true") or "true").strip().lower()

    return Settings(
        llm_provider=provider,
        llm_model=get_val("LLM_MODEL", "") or "",
        openai_api_key=get_val("OPENAI_API_KEY"),
        gemini_api_key=get_val("GEMINI_API_KEY"),
        anthropic_api_key=get_val("ANTHROPIC_API_KEY"),
        gemini_models=[m.strip() for m in (get_val("GEMINI_MODELS", "") or "").split(",") if m.strip()],
        groq_api_key=get_val("GROQ_API_KEY"),
        openrouter_api_key=get_val("OPENROUTER_API_KEY"),
        ollama_base_url=get_val("OLLAMA_BASE_URL"),
        tavily_api_key=get_val("TAVILY_API_KEY"),
        brave_api_key=get_val("BRAVE_API_KEY"),
        enable_free_search=free_search not in ("0", "false", "no", "off"),
        github_token=get_val("GITHUB_TOKEN"),
        enable_paper_search=(get_val("ENABLE_PAPER_SEARCH", "true") or "true").strip().lower()
        not in ("0", "false", "no", "off"),
        semantic_scholar_api_key=get_val("SEMANTIC_SCHOLAR_API_KEY"),
        max_concurrent_searches=get_int("MAX_CONCURRENT_SEARCHES", 3),
        search_timeout_seconds=get_int("SEARCH_TIMEOUT_SECONDS", 15),
        llm_timeout_seconds=get_int("LLM_TIMEOUT_SECONDS", 30),
        cache_ttl_seconds=get_int("CACHE_TTL_SECONDS", 3600),
        max_tool_calls_per_run=get_int("MAX_TOOL_CALLS_PER_RUN", 20),
        max_llm_calls_per_run=get_int("MAX_LLM_CALLS_PER_RUN", 10),
        max_wall_clock_seconds=get_int("MAX_WALL_CLOCK_SECONDS", 120),
        confidence_threshold=get_float("CONFIDENCE_THRESHOLD", 0.3),
        max_retries=get_int("MAX_RETRIES", 2),
        retry_base_delay_seconds=get_float("RETRY_BASE_DELAY_SECONDS", 0.5),
    )


def validate_settings(settings: Optional[Settings] = None, raise_error: bool = True) -> list[str]:
    """Validate that required configuration options and keys are present."""
    if settings is None:
        settings = get_settings()

    errors: list[str] = []

    from .llm import SUPPORTED_PROVIDERS, ConfigurationError, create_llm_provider, normalize_provider

    provider = normalize_provider(settings.llm_provider)
    if provider not in SUPPORTED_PROVIDERS:
        errors.append(
            f"Unsupported LLM provider '{settings.llm_provider}'. "
            f"Supported providers are: {', '.join(repr(p) for p in SUPPORTED_PROVIDERS)}."
        )
    else:
        try:
            create_llm_provider(settings)
        except ConfigurationError as exc:
            errors.append(str(exc))

    if not settings.tavily_api_key and not settings.brave_api_key and not settings.enable_free_search:
        errors.append(
            "No search provider available. Configure TAVILY_API_KEY or BRAVE_API_KEY, "
            "or set ENABLE_FREE_SEARCH=true to use keyless DuckDuckGo and Wikipedia search."
        )

    if settings.max_concurrent_searches <= 0:
        errors.append("MAX_CONCURRENT_SEARCHES must be greater than 0.")
    if settings.search_timeout_seconds <= 0:
        errors.append("SEARCH_TIMEOUT_SECONDS must be greater than 0.")
    if settings.llm_timeout_seconds <= 0:
        errors.append("LLM_TIMEOUT_SECONDS must be greater than 0.")
    if settings.cache_ttl_seconds < 0:
        errors.append("CACHE_TTL_SECONDS must be non-negative.")
    if settings.max_tool_calls_per_run <= 0:
        errors.append("MAX_TOOL_CALLS_PER_RUN must be greater than 0.")
    if settings.max_llm_calls_per_run <= 0:
        errors.append("MAX_LLM_CALLS_PER_RUN must be greater than 0.")
    if settings.max_wall_clock_seconds <= 0:
        errors.append("MAX_WALL_CLOCK_SECONDS must be greater than 0.")
    if settings.max_retries < 0:
        errors.append("MAX_RETRIES must be non-negative.")
    if settings.retry_base_delay_seconds < 0:
        errors.append("RETRY_BASE_DELAY_SECONDS must be non-negative.")
    if not (0.0 <= settings.confidence_threshold <= 1.0):
        errors.append("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0.")

    if errors and raise_error:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {err}" for err in errors))

    return errors


_settings_instance: Optional[Settings] = None


def get_settings(force_reload: bool = False) -> Settings:
    """Return a singleton instance of Settings."""
    global _settings_instance
    if _settings_instance is None or force_reload:
        _settings_instance = load_settings()
    return _settings_instance
