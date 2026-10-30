"""LLM provider adapter with dependency injection."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    content: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model: str = "unknown"
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


class LLMError(Exception):
    """Base exception for LLM provider errors."""

    pass


class ConfigurationError(ValueError, LLMError):
    """Raised when LLM configuration is invalid or missing required values."""

    pass


class InvalidJSONError(ValueError, LLMError):
    """Raised when an LLM provider fails to return valid JSON when requested."""

    pass


class RateLimitError(LLMError):
    """Raised when an LLM provider rate limit is exceeded."""

    pass


class APITimeoutError(LLMError):
    """Raised when an LLM provider API call times out."""

    pass


class APIConnectionError(LLMError):
    """Raised when connection to an LLM provider fails."""

    pass


class InternalServerError(LLMError):
    """Raised when an LLM provider returns a 5xx server error."""

    pass


class AuthenticationError(LLMError):
    """Raised when authentication with an LLM provider fails."""

    pass


class BadRequestError(LLMError):
    """Raised when an invalid request is made to an LLM provider."""

    pass


class ModelUnavailableError(LLMError):
    """Raised when the requested model does not exist or was retired (HTTP 404)."""

    pass


class ContentFilterError(LLMError):
    """Raised when a provider stops output early for safety or recitation reasons.

    Gemini's recitation filter can cut a response that quotes web text verbatim,
    which this pipeline asks for. Another provider usually completes the same prompt.
    """

    pass


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Complete prompt messages and return standard LLMResponse."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Underlying model name identifier."""
        ...


class OpenAIProvider(LLMProvider):
    """OpenAI API provider.

    Also drives any OpenAI-compatible endpoint (Groq, OpenRouter, Ollama) via
    ``base_url``; ``supports_json_mode`` is False for endpoints that reject
    ``response_format``, so JSON is requested by prompt and validated instead.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: int = 30,
        base_url: Optional[str] = None,
        provider_name: str = "openai",
        supports_json_mode: bool = True,
        extra_headers: Optional[dict[str, str]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = base_url
        self.provider_name = provider_name
        self.supports_json_mode = supports_json_mode
        self.extra_headers = extra_headers or {}
        self.extra_body = extra_body or {}
        self._client: Optional[Any] = None

    @property
    def name(self) -> str:
        return self.provider_name

    @property
    def model_name(self) -> str:
        return self.model

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as err:
                raise ImportError(
                    "OpenAI SDK is required for OpenAIProvider. "
                    "Install it using: pip install openai"
                ) from err
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
                base_url=self.base_url,
                default_headers=self.extra_headers or None,
            )
        return self._client

    def _normalize_error(self, exc: Exception) -> Exception:
        err_name = type(exc).__name__
        err_msg = str(exc)
        status_code = getattr(exc, "status_code", None)
        label = "OpenAI" if self.provider_name == "openai" else self.provider_name

        if err_name == "RateLimitError" or status_code == 429:
            return RateLimitError(f"{label} rate limit exceeded: {err_msg}")
        if err_name == "APITimeoutError" or "timeout" in err_msg.lower():
            return APITimeoutError(f"{label} timeout: {err_msg}")
        if err_name == "APIConnectionError" or "connection" in err_msg.lower():
            return APIConnectionError(f"{label} connection error: {err_msg}")
        if err_name == "InternalServerError" or (isinstance(status_code, int) and status_code >= 500):
            return InternalServerError(f"{label} server error: {err_msg}")
        if err_name == "AuthenticationError" or status_code in (401, 403):
            return AuthenticationError(f"{label} authentication error: {err_msg}")
        if err_name == "NotFoundError" or status_code == 404:
            return ModelUnavailableError(f"{label} model '{self.model}' unavailable: {err_msg}")
        if err_name == "BadRequestError" or status_code == 400:
            return BadRequestError(f"{label} bad request: {err_msg}")
        return exc

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Execute chat completion request with OpenAI client."""
        client = await self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        wants_json = bool(response_format and response_format.get("type") == "json_object")
        if response_format and self.supports_json_mode:
            kwargs["response_format"] = response_format
        elif wants_json:
            kwargs["messages"] = _with_json_instruction(messages)

        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Groq rejects a generation that breaks its own JSON mode with
            # 400 json_validate_failed. Retry once with prompt-instructed JSON,
            # which this method validates itself below.
            if wants_json and "response_format" in kwargs and "json_validate_failed" in str(exc):
                kwargs.pop("response_format")
                kwargs["messages"] = _with_json_instruction(messages)
                try:
                    response = await client.chat.completions.create(**kwargs)
                except Exception as retry_exc:
                    normalized = self._normalize_error(retry_exc)
                    if normalized is not retry_exc:
                        raise normalized from retry_exc
                    raise
            else:
                normalized = self._normalize_error(exc)
                if normalized is not exc:
                    raise normalized from exc
                raise

        if not response.choices:
            raise InternalServerError(f"{self.provider_name} returned no choices")
        choice = response.choices[0]
        usage = response.usage
        content = choice.message.content or ""

        if wants_json:
            content = _strip_code_fences(content)
            try:
                json.loads(content)
            except (json.JSONDecodeError, TypeError) as err:
                raise InvalidJSONError(
                    f"{self.provider_name} provider failed to return valid JSON when response_format was "
                    f"json_object: {content[:200]!r}"
                ) from err

        return LLMResponse(
            content=content,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            model=response.model,
            finish_reason=choice.finish_reason,
            raw_response=response,
        )


def _strip_code_fences(content: str) -> str:
    """Remove a surrounding ```json fence that some models add despite instructions."""
    clean = content.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()
    return clean


def _with_json_instruction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prepend a JSON-only instruction for endpoints without native JSON mode."""
    instruction = {
        "role": "system",
        "content": "Output must be a single valid JSON object only. No markdown fences or commentary.",
    }
    return [instruction, *messages]


class GeminiProvider(LLMProvider):
    """Google Gemini API provider using official Google GenAI SDK."""

    def __init__(self, api_key: str, model: str = "gemini-flash-latest", timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: Optional[Any] = None

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self.model

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as err:
                raise ImportError(
                    "Google GenAI SDK is required for GeminiProvider. "
                    "Install it using: pip install google-genai"
                ) from err

            # Google GenAI SDK HttpOptions timeout is configured in milliseconds (int)
            timeout_ms = int(self.timeout * 1000)
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    def _normalize_error(self, exc: Exception) -> Exception:
        err_name = type(exc).__name__
        err_msg = str(exc)
        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

        if status_code == 429 or "resource has been exhausted" in err_msg.lower() or "quota" in err_msg.lower():
            return RateLimitError(f"Gemini rate limit exceeded: {err_msg}")
        if (isinstance(status_code, int) and status_code >= 500) or err_name == "ServerError":
            return InternalServerError(f"Gemini server error: {err_msg}")
        if status_code in (401, 403) or "api_key" in err_msg.lower() or "permission" in err_msg.lower():
            return AuthenticationError(f"Gemini authentication error: {err_msg}")
        if "timeout" in err_msg.lower() or "deadline" in err_msg.lower():
            return APITimeoutError(f"Gemini timeout: {err_msg}")
        if "connection" in err_msg.lower():
            return APIConnectionError(f"Gemini connection error: {err_msg}")
        if status_code == 404:
            return ModelUnavailableError(f"Gemini model '{self.model}' unavailable: {err_msg}")
        if err_name == "ClientError" and status_code == 400:
            return BadRequestError(f"Gemini bad request: {err_msg}")
        return exc

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Execute content generation request with Gemini client."""
        from google.genai import types

        client = await self._get_client()

        # Separate system instruction from conversational messages
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        system_instruction = "\n\n".join(filter(None, system_parts)) if system_parts else None

        non_system = [m for m in messages if m.get("role") != "system"]
        if not non_system:
            non_system = [{"role": "user", "content": "Hello"}]

        # Format and alternate contents for Gemini
        gemini_contents: list[Any] = []
        for msg in non_system:
            role = "model" if msg.get("role") == "assistant" else "user"
            content = msg.get("content", "")
            if gemini_contents and gemini_contents[-1].role == role:
                prev_text = gemini_contents[-1].parts[0].text or ""
                gemini_contents[-1] = types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=f"{prev_text}\n\n{content}")],
                )
            else:
                gemini_contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=content)],
                    )
                )

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        # Thinking tokens count against max_output_tokens on 2.5 models and truncate
        # structured JSON; the pipeline does its own planning, so Flash runs without it.
        if "2.5-flash" in self.model:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        if response_format:
            rf_type = response_format.get("type")
            if rf_type == "json_object":
                config_kwargs["response_mime_type"] = "application/json"
            else:
                raise ValueError(f"Unsupported response_format type for Gemini: '{rf_type}'")

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=gemini_contents,
                config=config,
            )
        except Exception as exc:
            normalized = self._normalize_error(exc)
            if normalized is not exc:
                raise normalized from exc
            raise

        content = ""
        finish_reason = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                fr_str = str(candidate.finish_reason).upper()
                if "STOP" in fr_str:
                    finish_reason = "stop"
                elif "MAX_TOKENS" in fr_str or "LENGTH" in fr_str:
                    finish_reason = "length"
                elif "SAFETY" in fr_str or "RECITATION" in fr_str or "BLOCK" in fr_str:
                    finish_reason = "content_filter"
                else:
                    finish_reason = fr_str.lower()
            if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                content = "".join(
                    getattr(p, "text", "") or "" for p in candidate.content.parts if hasattr(p, "text")
                )
        elif hasattr(response, "text") and response.text:
            content = response.text

        if finish_reason == "content_filter":
            raise ContentFilterError(
                f"Gemini stopped generation early (safety/recitation filter) after {len(content)} chars"
            )

        # If json_object was requested, strip potential markdown fences and validate JSON
        if response_format and response_format.get("type") == "json_object":
            clean = content.strip()
            if clean.startswith("```"):
                lines = clean.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                clean = "\n".join(lines).strip()
            try:
                json.loads(clean)
                content = clean
            except (json.JSONDecodeError, TypeError) as err:
                raise InvalidJSONError(
                    f"Gemini provider failed to return valid JSON when response_format was json_object "
                    f"(finish_reason={finish_reason}): {content[:200]!r}"
                ) from err

        input_tokens = None
        output_tokens = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

        model_version = getattr(response, "model_version", None) or self.model

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_version,
            finish_reason=finish_reason,
            raw_response=response,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider using official Anthropic SDK.

    Note: Anthropic does not provide a native schema-constrained json_object
    response format. Prompting guides the model toward JSON output, but does not guarantee
    valid structure. When json_object is requested, responses are strictly validated as
    JSON, and an InvalidJSONError is raised if the output cannot be decoded.
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5", timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: Optional[Any] = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self.model

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as err:
                raise ImportError(
                    "Anthropic SDK is required for AnthropicProvider. "
                    "Install it using: pip install anthropic"
                ) from err
            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                timeout=float(self.timeout),
                max_retries=0,
            )
        return self._client

    def _normalize_error(self, exc: Exception) -> Exception:
        err_name = type(exc).__name__
        err_msg = str(exc)
        status_code = getattr(exc, "status_code", None)

        if err_name == "RateLimitError" or status_code == 429:
            return RateLimitError(f"Anthropic rate limit exceeded: {err_msg}")
        if err_name == "APITimeoutError" or "timeout" in err_msg.lower():
            return APITimeoutError(f"Anthropic timeout: {err_msg}")
        if err_name == "APIConnectionError" or "connection" in err_msg.lower():
            return APIConnectionError(f"Anthropic connection error: {err_msg}")
        if err_name == "InternalServerError" or (isinstance(status_code, int) and status_code >= 500):
            return InternalServerError(f"Anthropic server error: {err_msg}")
        if err_name == "AuthenticationError" or status_code in (401, 403):
            return AuthenticationError(f"Anthropic authentication error: {err_msg}")
        if err_name == "BadRequestError" or status_code == 400:
            return BadRequestError(f"Anthropic bad request: {err_msg}")
        return exc

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Execute chat completion request with Anthropic client."""
        client = await self._get_client()

        # System message handling
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        system_str = "\n\n".join(filter(None, system_parts))

        if response_format:
            rf_type = response_format.get("type")
            if rf_type == "json_object":
                # Guide model toward JSON output; valid JSON will be strictly validated below
                json_prompt = (
                    "CRITICAL: Output must be valid JSON only. Do not include markdown code fences, "
                    "preamble, or trailing commentary."
                )
                system_str = f"{system_str}\n\n{json_prompt}".strip() if system_str else json_prompt
            else:
                raise ValueError(f"Unsupported response_format type for Anthropic: '{rf_type}'")

        # Conversational message formatting
        non_system = [m for m in messages if m.get("role") != "system"]
        if not non_system:
            non_system = [{"role": "user", "content": "Hello"}]

        formatted_messages: list[dict[str, str]] = []
        for msg in non_system:
            role = "assistant" if msg.get("role") == "assistant" else "user"
            content = msg.get("content", "")
            if formatted_messages and formatted_messages[-1]["role"] == role:
                formatted_messages[-1]["content"] += f"\n\n{content}"
            else:
                formatted_messages.append({"role": role, "content": content})

        if formatted_messages[0]["role"] != "user":
            formatted_messages.insert(0, {"role": "user", "content": "(continue)"})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_str:
            kwargs["system"] = system_str

        try:
            response = await client.messages.create(**kwargs)
        except Exception as exc:
            normalized = self._normalize_error(exc)
            if normalized is not exc:
                raise normalized from exc
            raise

        content_parts = []
        if hasattr(response, "content") and response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content_parts.append(block.text or "")
                elif isinstance(block, dict) and "text" in block:
                    content_parts.append(block["text"] or "")
                elif isinstance(block, str):
                    content_parts.append(block)
        content = "".join(content_parts)

        # If json_object was requested, strip potential markdown fences and validate JSON
        if response_format and response_format.get("type") == "json_object":
            clean = content.strip()
            if clean.startswith("```"):
                lines = clean.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                clean = "\n".join(lines).strip()
            try:
                json.loads(clean)
                content = clean
            except (json.JSONDecodeError, TypeError) as err:
                raise InvalidJSONError(
                    f"Anthropic provider failed to return valid JSON when response_format was json_object: "
                    f"{content[:200]!r}"
                ) from err

        finish_reason = None
        if hasattr(response, "stop_reason") and response.stop_reason:
            sr = str(response.stop_reason).lower()
            if sr in ("end_turn", "stop_sequence"):
                finish_reason = "stop"
            elif sr == "max_tokens":
                finish_reason = "length"
            elif sr == "tool_use":
                finish_reason = "tool_calls"
            else:
                finish_reason = sr

        input_tokens = None
        output_tokens = None
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "input_tokens", None)
            output_tokens = getattr(response.usage, "output_tokens", None)

        model = getattr(response, "model", None) or self.model

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            finish_reason=finish_reason,
            raw_response=response,
        )


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for offline testing."""

    def __init__(
        self,
        responses: Optional[dict[str, str]] = None,
        default_response: Optional[str] = None,
    ):
        self._responses: dict[str, str] = responses or {}
        self._default_response = default_response or '{"error": "no mock response configured"}'
        self._calls: list[dict[str, Any]] = []
        self._call_count: int = 0
        self._sequential: list[str] = []
        self._seq_index: int = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model"

    def add_response(self, keyword: str, response: str) -> None:
        """Add keyword-triggered mock response."""
        self._responses[keyword.lower()] = response

    def set_sequential_responses(self, responses: list[str]) -> None:
        """Set responses that will be returned in order."""
        self._sequential = responses
        self._seq_index = 0

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """Generate response matching sequential or keyword mock rules."""
        self._calls.append({"messages": messages, "temperature": temperature})
        self._call_count += 1

        if self._sequential and self._seq_index < len(self._sequential):
            content = self._sequential[self._seq_index]
            self._seq_index += 1
            return LLMResponse(
                content=content,
                input_tokens=sum(len(m.get("content", "")) // 4 for m in messages),
                output_tokens=len(content) // 4,
                model="mock-model",
                finish_reason="stop",
            )

        all_content = " ".join(m.get("content", "") for m in messages).lower()

        for keyword, response in self._responses.items():
            if keyword in all_content:
                return LLMResponse(
                    content=response,
                    input_tokens=sum(len(m.get("content", "")) // 4 for m in messages),
                    output_tokens=len(response) // 4,
                    model="mock-model",
                    finish_reason="stop",
                )

        return LLMResponse(
            content=self._default_response,
            input_tokens=sum(len(m.get("content", "")) // 4 for m in messages),
            output_tokens=len(self._default_response) // 4,
            model="mock-model",
            finish_reason="stop",
        )


class FallbackLLMProvider(LLMProvider):
    """Try providers in order, moving on when one is rate-limited, down or misconfigured.

    Free tiers hit quota limits often; failing over to another free provider keeps
    a demo or eval run alive instead of aborting. Bad requests and invalid JSON are
    not failed over: those are prompt problems another model would likely repeat.
    """

    _FAILOVER = (
        RateLimitError, AuthenticationError, APIConnectionError, APITimeoutError,
        InternalServerError, ContentFilterError, ModelUnavailableError,
    )

    def __init__(self, providers: list[LLMProvider], per_provider_timeout: Optional[float] = None):
        if not providers:
            raise ConfigurationError("FallbackLLMProvider needs at least one provider")
        self.providers = providers
        self.per_provider_timeout = per_provider_timeout
        self._active = providers[0]
        self.failovers: list[str] = []

    @property
    def name(self) -> str:
        return self._active.name

    @property
    def model_name(self) -> str:
        return self._active.model_name

    @property
    def total_timeout(self) -> Optional[float]:
        """Upper bound for a full pass over the chain, used by the caller's own timeout."""
        if self.per_provider_timeout is None:
            return None
        return self.per_provider_timeout * len(self.providers)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        # Start from the provider that last succeeded: one that just hit its quota
        # will keep failing, and retrying it first wastes a request on every call.
        start = self.providers.index(self._active)
        ordered = self.providers[start:] + self.providers[:start]
        last_exc: Optional[Exception] = None
        for provider in ordered:
            try:
                call = provider.complete(messages, temperature, max_tokens, response_format)
                if self.per_provider_timeout:
                    response = await asyncio.wait_for(call, timeout=self.per_provider_timeout)
                else:
                    response = await call
                self._active = provider
                return response
            except asyncio.TimeoutError:
                # A hung (overloaded) model must not consume the whole call budget.
                exc = APITimeoutError(f"{provider.name}/{provider.model_name} timed out")
                self.failovers.append(f"{provider.name}: APITimeoutError")
                last_exc = exc
            except (*self._FAILOVER, ImportError) as exc:
                self.failovers.append(f"{provider.name}: {type(exc).__name__}")
                last_exc = exc
        assert last_exc is not None
        raise last_exc


DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "ollama": "llama3.1",
    "mock": "mock-model",
}

# Groq retires models often (llama-3.3-70b-versatile went 404), so chain several.
GROQ_FREE_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
GEMINI_FREE_MODELS = ("gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-flash-latest")

PROVIDER_ALIASES: dict[str, str] = {
    "google": "gemini",
    "google-gemini": "gemini",
    "claude": "anthropic",
}

# Free-first order used by LLM_PROVIDER=auto.
AUTO_ORDER = ("groq", "gemini", "openrouter", "ollama", "openai", "anthropic")
SUPPORTED_PROVIDERS = ("auto", "openai", "gemini", "anthropic", "groq", "openrouter", "ollama", "mock")


def normalize_provider(raw: Optional[str]) -> str:
    p = (raw or "openai").strip().lower()
    return PROVIDER_ALIASES.get(p, p)


def available_providers(settings: Any) -> list[str]:
    """Providers in AUTO_ORDER that have credentials configured."""
    keys = {
        "gemini": getattr(settings, "gemini_api_key", None),
        "groq": getattr(settings, "groq_api_key", None),
        "openrouter": getattr(settings, "openrouter_api_key", None),
        "ollama": getattr(settings, "ollama_base_url", None),
        "openai": getattr(settings, "openai_api_key", None),
        "anthropic": getattr(settings, "anthropic_api_key", None),
    }
    return [p for p in AUTO_ORDER if keys.get(p)]


def _require(settings: Any, attr: str, message: str) -> str:
    value = getattr(settings, attr, None)
    if not value:
        raise ConfigurationError(message)
    return value


def _groq_provider(key: str, model: str, timeout: int) -> OpenAIProvider:
    # gpt-oss models reason before answering; reasoning tokens count against
    # max_tokens and the tokens-per-minute limit, so keep the effort low.
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else None
    return OpenAIProvider(
        api_key=key, model=model, timeout=timeout,
        base_url="https://api.groq.com/openai/v1", provider_name="groq", extra_body=extra,
    )


def _build_single(provider: str, settings: Any, model: Optional[str], timeout: int) -> LLMProvider:
    explicit_model = bool(model)
    model = model or DEFAULT_MODELS.get(provider, "")
    if provider == "openai":
        key = _require(settings, "openai_api_key", "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'. "
                       "Set OPENAI_API_KEY in your .env file or environment.")
        return OpenAIProvider(api_key=key, model=model, timeout=timeout)
    if provider == "gemini":
        key = _require(settings, "gemini_api_key", "GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'. "
                       "Get a free key at https://aistudio.google.com/apikey.")
        # Free-tier quotas are per model (e.g. 20 requests/day), so with no explicit
        # model we chain several; '-latest' aliases survive model retirements.
        if explicit_model:
            return GeminiProvider(api_key=key, model=model, timeout=timeout)
        models = getattr(settings, "gemini_models", None) or list(GEMINI_FREE_MODELS)
        chain = [GeminiProvider(api_key=key, model=m, timeout=timeout) for m in models]
        return chain[0] if len(chain) == 1 else FallbackLLMProvider(chain, per_provider_timeout=timeout)
    if provider == "anthropic":
        key = _require(settings, "anthropic_api_key", "ANTHROPIC_API_KEY is required when LLM_PROVIDER is "
                       "'anthropic'. Set ANTHROPIC_API_KEY in your .env file or environment.")
        return AnthropicProvider(api_key=key, model=model, timeout=timeout)
    if provider == "groq":
        key = _require(settings, "groq_api_key", "GROQ_API_KEY is required when LLM_PROVIDER is 'groq'. "
                       "Get a free key at https://console.groq.com/keys.")
        models = [model] if explicit_model else list(GROQ_FREE_MODELS)
        chain = [_groq_provider(key, m, timeout) for m in models]
        return chain[0] if len(chain) == 1 else FallbackLLMProvider(chain, per_provider_timeout=timeout)
    if provider == "openrouter":
        key = _require(settings, "openrouter_api_key", "OPENROUTER_API_KEY is required when LLM_PROVIDER is "
                       "'openrouter'. Get a free key at https://openrouter.ai/keys.")
        return OpenAIProvider(
            api_key=key, model=model, timeout=timeout,
            base_url="https://openrouter.ai/api/v1", provider_name="openrouter",
            supports_json_mode=False, extra_headers={"X-Title": "AI Research Agent"},
        )
    if provider == "ollama":
        base_url = getattr(settings, "ollama_base_url", None) or "http://localhost:11434/v1"
        return OpenAIProvider(
            api_key="ollama", model=model, timeout=timeout, base_url=base_url, provider_name="ollama",
        )
    if provider == "mock":
        return MockLLMProvider()
    raise ConfigurationError(
        f"Unsupported LLM provider '{provider}'. "
        f"Supported providers are: {', '.join(repr(p) for p in SUPPORTED_PROVIDERS)}."
    )


def create_llm_provider(settings: Any) -> LLMProvider:
    """Create an LLMProvider from settings.

    ``LLM_PROVIDER=auto`` chains every configured provider, free tiers first,
    behind a FallbackLLMProvider. ``LLM_MODEL`` applies only to a single named
    provider, since model ids are not portable across providers.

    Raises:
        ConfigurationError: If the provider is unsupported or the required API key is missing.
    """
    provider = normalize_provider(getattr(settings, "llm_provider", None))
    timeout = int(getattr(settings, "llm_timeout_seconds", 30))
    raw_model = getattr(settings, "llm_model", None)

    if provider == "auto":
        names = available_providers(settings)
        if not names:
            raise ConfigurationError(
                "LLM_PROVIDER=auto found no configured provider. Set at least one free key: "
                "GEMINI_API_KEY, GROQ_API_KEY or OPENROUTER_API_KEY (or OLLAMA_BASE_URL)."
            )
        built: list[LLMProvider] = []
        for n in names:
            single = _build_single(n, settings, None, timeout)
            built.extend(single.providers if isinstance(single, FallbackLLMProvider) else [single])
        return built[0] if len(built) == 1 else FallbackLLMProvider(built, per_provider_timeout=timeout)

    # A stale OpenAI default model must not leak into another provider.
    model = raw_model
    if provider != "openai" and raw_model in (DEFAULT_MODELS["openai"], "auto", ""):
        model = None
    return _build_single(provider, settings, model, timeout)


__all__ = [
    "LLMResponse",
    "LLMError",
    "ConfigurationError",
    "InvalidJSONError",
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "AuthenticationError",
    "BadRequestError",
    "ContentFilterError",
    "ModelUnavailableError",
    "LLMProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "AnthropicProvider",
    "MockLLMProvider",
    "FallbackLLMProvider",
    "DEFAULT_MODELS",
    "SUPPORTED_PROVIDERS",
    "available_providers",
    "normalize_provider",
    "create_llm_provider",
]
