from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import requests


class LLMProviderError(Exception):
    """Raised when an upstream LLM provider returns an error."""


def _resolve_provider_timeout_seconds() -> int:
    raw_value = os.getenv("LLM_PROVIDER_TIMEOUT_SECONDS", "120").strip()
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return 120
    return max(5, min(parsed, 300))


def _resolve_ollama_timeout_seconds() -> int:
    raw_value = os.getenv("LLM_PROVIDER_TIMEOUT_SECONDS_OLLAMA", "").strip()
    if not raw_value:
        raw_value = os.getenv("LLM_PROVIDER_TIMEOUT_SECONDS", "26").strip()
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return 26
    return max(5, min(parsed, 28))


def _resolve_ollama_retry_count() -> int:
    raw_value = os.getenv("LLM_PROVIDER_RETRY_COUNT_OLLAMA", "1").strip()
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(parsed, 2))


@dataclass
class LLMResult:
    text: str
    duration_ms: float


@dataclass(frozen=True)
class LLMModelOption:
    value: str
    label: str


@dataclass(frozen=True)
class LLMProviderDefinition:
    key: str
    label: str
    models: Tuple[LLMModelOption, ...]


_LLM_PROVIDER_DEFINITIONS: Tuple[LLMProviderDefinition, ...] = (
    LLMProviderDefinition(
        key="openai",
        label="OpenAI",
        models=(
            LLMModelOption(value="gpt-5", label="gpt-5"),
            LLMModelOption(value="gpt-5-mini", label="gpt-5-mini"),
            LLMModelOption(value="gpt-5-nano", label="gpt-5-nano"),
            LLMModelOption(value="gpt-4.1", label="gpt-4.1"),
            LLMModelOption(value="other", label="Other"),
        ),
    ),
    LLMProviderDefinition(
        key="anthropic",
        label="Anthropic",
        models=(
            LLMModelOption(
                value="claude-sonnet-4-5-20250929", label="claude-sonnet-4-5-20250929"
            ),
            LLMModelOption(
                value="claude-opus-4-1-20250805", label="claude-opus-4-1-20250805"
            ),
            LLMModelOption(
                value="claude-haiku-4-5-20251001", label="claude-haiku-4-5-20251001"
            ),
            LLMModelOption(value="other", label="Other"),
        ),
    ),
    LLMProviderDefinition(
        key="gemini",
        label="Gemini",
        models=(
            LLMModelOption(value="gemini-2.5-pro", label="gemini-2.5-pro"),
            LLMModelOption(value="gemini-flash-latest", label="gemini-flash-latest"),
            LLMModelOption(
                value="gemini-flash-lite-latest", label="gemini-flash-lite-latest"
            ),
            LLMModelOption(value="other", label="Other"),
        ),
    ),
    LLMProviderDefinition(
        key="groq",
        label="Groq",
        models=(
            LLMModelOption(value="qwen/qwen3-32b", label="qwen/qwen3-32b"),
            LLMModelOption(
                value="meta-llama/llama-4-maverick-17b-128e-instruct",
                label="meta-llama/llama-4-maverick-17b-128e-instruct",
            ),
            LLMModelOption(
                value="meta-llama/llama-4-scout-17b-16e-instruct",
                label="meta-llama/llama-4-scout-17b-16e-instruct",
            ),
            LLMModelOption(value="openai/gpt-oss-120b", label="openai/gpt-oss-120b"),
            LLMModelOption(value="openai/gpt-oss-20b", label="openai/gpt-oss-20b"),
            LLMModelOption(value="other", label="Other"),
        ),
    ),
    LLMProviderDefinition(
        key="ollama",
        label="Ollama (Local)",
        models=(
            LLMModelOption(value="llama3", label="llama3"),
            LLMModelOption(value="mistral", label="mistral"),
            LLMModelOption(value="gemma", label="gemma"),
            LLMModelOption(value="qwen", label="qwen"),
            LLMModelOption(value="other", label="Other"),
        ),
    ),
)


def get_llm_providers() -> List[Dict[str, Any]]:
    return [
        {
            "key": provider.key,
            "label": provider.label,
            "models": [
                {"value": model.value, "label": model.label}
                for model in provider.models
            ],
        }
        for provider in _LLM_PROVIDER_DEFINITIONS
    ]


def _post_json(
    url: str,
    *,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    params: Dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> Tuple[Dict[str, Any] | None, float]:
    started_at = time.perf_counter()
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            params=params,
            timeout=timeout_seconds or _resolve_provider_timeout_seconds(),
        )
    except requests.RequestException as exc:  # pragma: no cover - network failures
        raise LLMProviderError("Network request to provider failed") from exc

    duration_ms = (time.perf_counter() - started_at) * 1000

    data: Dict[str, Any] | None
    try:
        data = response.json() if response.text else None
    except ValueError:
        data = None

    if not response.ok:
        provider_message = None
        if isinstance(data, dict):
            provider_message = (
                data.get("error", {}).get("message")
                or data.get("message")
                or data.get("error")
            )
        raise LLMProviderError(
            provider_message
            or f"Provider request failed with status {response.status_code}"
        )

    if data is None and response.status_code != 204:
        raise LLMProviderError("Provider returned an unreadable response")

    return data, duration_ms


def _call_openai(model: str, prompt: str, api_key: str) -> LLMResult:
    data, duration_ms = _post_json(
        "https://api.openai.com/v1/responses",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        payload={
            "model": model,
            "input": prompt,
        },
    )
    chunks: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
            elif isinstance(item.get("text"), str):
                chunks.append(item["text"])
    text = "\n".join(chunk.strip() for chunk in chunks if chunk).strip()
    if not text and isinstance(data.get("output_text"), str):
        text = data["output_text"].strip()
    if not text:
        raise LLMProviderError("Provider returned an empty response")
    return LLMResult(text=text, duration_ms=duration_ms)


def _call_anthropic(model: str, prompt: str, api_key: str) -> LLMResult:
    data, duration_ms = _post_json(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        payload={
            "model": model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    content = data.get("content")
    chunks: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
            elif isinstance(part, dict) and isinstance(part.get("content"), list):
                inner_text = [
                    inner.get("text")
                    for inner in part["content"]
                    if isinstance(inner, dict) and isinstance(inner.get("text"), str)
                ]
                if inner_text:
                    chunks.append("\n".join(inner_text))
    text = "\n\n".join(chunk.strip() for chunk in chunks if chunk).strip()
    if not text:
        raise LLMProviderError("Provider returned an empty response")
    return LLMResult(text=text, duration_ms=duration_ms)


def _call_gemini(model: str, prompt: str, api_key: str) -> LLMResult:
    data, duration_ms = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"Content-Type": "application/json"},
        payload={"contents": [{"parts": [{"text": prompt}]}]},
        params={"key": api_key},
    )

    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LLMProviderError("Provider returned an empty response")
    parts = (
        candidates[0].get("content", {}).get("parts")
        if isinstance(candidates[0], dict)
        else None
    )
    text = ""
    if isinstance(parts, list):
        collected = [
            part.get("text")
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        text = "\n".join(chunk.strip() for chunk in collected if chunk).strip()
    if not text:
        raise LLMProviderError("Provider returned an empty response")
    return LLMResult(text=text, duration_ms=duration_ms)


def _call_groq(model: str, prompt: str, api_key: str) -> LLMResult:
    data, duration_ms = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        payload={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not text:
        raise LLMProviderError("Provider returned an empty response")
    return LLMResult(text=text, duration_ms=duration_ms)


def _resolve_ollama_base_url(api_key: str) -> str:
    override = api_key.strip()
    if override.startswith("http"):
        return override.rstrip("/")
    env_value = os.getenv("OLLAMA_BASE_URL", "").strip()
    if env_value:
        return env_value.rstrip("/")
    return "http://localhost:11434"


def _call_ollama(model: str, prompt: str, api_key: str) -> LLMResult:
    # API key field doubles as endpoint override in the UI.
    base_url = _resolve_ollama_base_url(api_key)
    url = f"{base_url}/api/chat"
    timeout_seconds = _resolve_ollama_timeout_seconds()
    max_attempts = _resolve_ollama_retry_count()

    last_error: LLMProviderError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            data, duration_ms = _post_json(
                url,
                headers={"Content-Type": "application/json"},
                payload={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout_seconds=timeout_seconds,
            )
            break
        except LLMProviderError as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise
            time.sleep(1.0)
    else:  # pragma: no cover - defensive branch
        if last_error is not None:
            raise last_error
        raise LLMProviderError("Provider request failed")

    # Ollama /api/chat response format: {"model":..., "message": {"role": "assistant", "content": "..."}}
    text = data.get("message", {}).get("content", "").strip()

    if not text:
        # Fallback to /api/generate format: {"response": "..."}
        text = data.get("response", "").strip()

    if not text:
        raise LLMProviderError("Provider returned an empty response")

    return LLMResult(text=text, duration_ms=duration_ms)


_PROVIDER_HANDLERS: Dict[str, Callable[[str, str, str], LLMResult]] = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "groq": _call_groq,
    "ollama": _call_ollama,
}


def call_llm(provider: str, model: str, prompt: str, api_key: str) -> LLMResult:
    provider_key = provider.lower().strip()
    if not provider_key:
        raise ValueError("Provider is required")
    if not model.strip():
        raise ValueError("Model is required")
    if not prompt.strip():
        raise ValueError("Prompt is required")

    # Require API key for all except Ollama (unless used for custom URL override)
    if provider_key != "ollama" and not api_key.strip():
        raise ValueError("API key is required")

    try:
        handler = _PROVIDER_HANDLERS[provider_key]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError("Unsupported provider selected") from exc

    return handler(model, prompt, api_key)
