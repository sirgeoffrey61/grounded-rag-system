"""
LLM provider abstraction for grounded RAG generation.

Why provider abstraction:
    Retrieval, reranking, citations, and confidence stay stable while the inference
    backend changes (Groq today, OpenAI / Azure / local Ollama tomorrow).

Why Ollama fails in cloud (Render, Railway, Streamlit Cloud):
    PaaS containers have no long-running local daemon, no host.docker.internal to a
    laptop, and no GPU for large local models. A hosted OpenAI-compatible API
    (Groq) removes that ops burden and scales with request volume.

Why hosted inference (Groq):
    Low-latency Llama/Mixtral-class models via HTTPS, API keys in env/secrets,
    and health checks that work from any region without sidecars.

TODO: Provider fallback chain (Groq -> OpenAI -> Ollama) with circuit breaker.
TODO: Response caching (Redis / disk) for repeated benchmark queries.
TODO: Streaming tokens (SSE) for /ask and Streamlit UI.
TODO: Multi-model routing (fast 8B vs larger model by query length / confidence).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL_NAME = "llama-3.1-8b-instant"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2


class LLMClientError(RuntimeError):
    """Raised when generation fails after retries or health is bad."""


@dataclass(frozen=True)
class LLMHealthStatus:
    status: str  # ok | degraded | unavailable
    detail: str = ""
    latency_ms: float | None = None
    provider: str = ""
    model: str = ""


@dataclass(frozen=True)
class LLMGenerateResult:
    text: str
    latency_ms: float
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def _env_api_key() -> str:
    return (os.environ.get("GROQ_API_KEY") or os.environ.get("RAG_GROQ_API_KEY") or "").strip()


def _env_provider() -> str:
    return (os.environ.get("RAG_LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()


def _env_model() -> str:
    return (os.environ.get("RAG_MODEL_NAME") or DEFAULT_MODEL_NAME).strip()


def _env_timeout() -> float:
    raw = os.environ.get("RAG_LLM_TIMEOUT_SECONDS") or os.environ.get("RAG_OLLAMA_TIMEOUT_SECONDS")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_TIMEOUT_SECONDS


def _extract_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _extract_token_usage(response: Any) -> dict[str, int | None]:
    meta = getattr(response, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if not isinstance(usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "401" in msg or "invalid api key" in msg or "incorrect api key" in msg:
        return "invalid_api_key"
    if "429" in msg or "rate limit" in msg:
        return "rate_limit"
    if "timeout" in msg or isinstance(exc, TimeoutError):
        return "timeout"
    return "unknown"


class LLMClient:
    """OpenAI-compatible chat client (Groq by default)."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float = 0.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.provider = (provider or _env_provider()).lower()
        self.model = model or _env_model()
        self.api_key = (api_key if api_key is not None else _env_api_key()).strip()
        self.base_url = (base_url or GROQ_OPENAI_BASE_URL).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else _env_timeout()
        )
        self.temperature = temperature
        self.max_retries = max(0, max_retries)
        self._chat: ChatOpenAI | None = None

    def _ensure_chat(self) -> ChatOpenAI:
        if self._chat is not None:
            return self._chat
        if self.provider != "groq":
            raise LLMClientError(
                f"Unsupported RAG_LLM_PROVIDER={self.provider!r} (only 'groq' is implemented)"
            )
        if not self.api_key:
            raise LLMClientError(
                "GROQ_API_KEY is not set. Add it to .env or your cloud provider secrets."
            )
        self._chat = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=self.temperature,
            timeout=self.timeout_seconds,
            max_retries=0,
        )
        return self._chat

    def generate(self, messages: list[BaseMessage | Any]) -> str:
        """Run chat completion; returns assistant text."""
        return self.generate_with_metadata(messages).text

    def generate_with_metadata(self, messages: list[BaseMessage | Any]) -> LLMGenerateResult:
        """Run chat completion with latency and token usage logging."""
        last_exc: Exception | None = None
        attempts = self.max_retries + 1

        for attempt in range(1, attempts + 1):
            t0 = time.perf_counter()
            try:
                chat = self._ensure_chat()
                response = chat.invoke(messages)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                text = _extract_content(response)
                if not text:
                    raise LLMClientError("Empty response from LLM")

                usage = _extract_token_usage(response)
                logger.info(
                    "LLM generate ok provider=%s model=%s latency_ms=%.1f "
                    "prompt_tokens=%s completion_tokens=%s attempt=%d/%d",
                    self.provider,
                    self.model,
                    latency_ms,
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                    attempt,
                    attempts,
                )
                return LLMGenerateResult(
                    text=text,
                    latency_ms=latency_ms,
                    provider=self.provider,
                    model=self.model,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                )
            except LLMClientError:
                raise
            except Exception as exc:
                last_exc = exc
                kind = _classify_error(exc)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                logger.warning(
                    "LLM generate failed provider=%s model=%s kind=%s latency_ms=%.1f "
                    "attempt=%d/%d error=%s",
                    self.provider,
                    self.model,
                    kind,
                    latency_ms,
                    attempt,
                    attempts,
                    exc,
                )
                if kind == "invalid_api_key":
                    raise LLMClientError(
                        "Invalid GROQ_API_KEY — check your Groq console key."
                    ) from exc
                if attempt >= attempts:
                    break
                if kind == "rate_limit":
                    time.sleep(min(2.0 ** (attempt - 1), 8.0))
                elif kind == "timeout":
                    time.sleep(0.5)

        assert last_exc is not None
        kind = _classify_error(last_exc)
        if kind == "timeout":
            raise LLMClientError(
                f"LLM timed out after {self.timeout_seconds}s (model={self.model})"
            ) from last_exc
        if kind == "rate_limit":
            raise LLMClientError(
                f"Groq rate limit exceeded (model={self.model}): {last_exc}"
            ) from last_exc
        raise LLMClientError(
            f"LLM generation failed ({self.provider}/{self.model}): {last_exc}"
        ) from last_exc

    def health_check(self) -> LLMHealthStatus:
        """Probe Groq OpenAI-compatible /models endpoint."""
        if self.provider != "groq":
            return LLMHealthStatus(
                status="unavailable",
                detail=f"unsupported provider={self.provider}",
                provider=self.provider,
                model=self.model,
            )
        if not self.api_key:
            return LLMHealthStatus(
                status="unavailable",
                detail="GROQ_API_KEY not set",
                provider=self.provider,
                model=self.model,
            )
        url = f"{self.base_url}/models"
        try:
            t0 = time.perf_counter()
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 401:
                return LLMHealthStatus(
                    status="unavailable",
                    detail="invalid GROQ_API_KEY (HTTP 401)",
                    latency_ms=round(latency_ms, 2),
                    provider=self.provider,
                    model=self.model,
                )
            if resp.status_code == 429:
                return LLMHealthStatus(
                    status="degraded",
                    detail="Groq rate limited on health probe (HTTP 429)",
                    latency_ms=round(latency_ms, 2),
                    provider=self.provider,
                    model=self.model,
                )
            if resp.status_code != 200:
                return LLMHealthStatus(
                    status="unavailable",
                    detail=f"HTTP {resp.status_code}",
                    latency_ms=round(latency_ms, 2),
                    provider=self.provider,
                    model=self.model,
                )
            data = resp.json()
            ids = [
                m.get("id", "")
                for m in data.get("data", [])
                if isinstance(m, dict)
            ]
            has_model = any(self.model in mid for mid in ids)
            return LLMHealthStatus(
                status="ok" if has_model else "degraded",
                detail=f"model={self.model} listed={has_model}",
                latency_ms=round(latency_ms, 2),
                provider=self.provider,
                model=self.model,
            )
        except Exception as exc:
            return LLMHealthStatus(
                status="unavailable",
                detail=str(exc),
                provider=self.provider,
                model=self.model,
            )


_client: LLMClient | None = None


def get_llm_client(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
) -> LLMClient:
    """Return a process-wide default client (lazy singleton)."""
    global _client
    if _client is None:
        _client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
    return _client


def reset_llm_client() -> None:
    """Test helper to clear the singleton."""
    global _client
    _client = None
