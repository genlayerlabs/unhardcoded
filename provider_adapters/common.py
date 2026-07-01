"""Shared provider-adapter primitives."""
from __future__ import annotations

import json
import asyncio
import time
from typing import Awaitable, Callable, Any

CallProviderHook = Callable[[dict], dict]
AsyncCallProviderHook = Callable[[dict], Awaitable[dict]]
TokenProvider = Callable[[], "str | None"]


def cached_tokens(usage: dict) -> "int | None":
    """Prompt-cache read tokens across provider response shapes."""
    if not isinstance(usage, dict):
        return None
    for parent in ("prompt_tokens_details", "input_tokens_details"):
        d = usage.get(parent)
        if isinstance(d, dict) and d.get("cached_tokens") is not None:
            return d.get("cached_tokens")
    return usage.get("cache_read_input_tokens")


def auth_token(
    request: dict,
    env_get: Callable[[str], str | None],
    default_env: str,
) -> tuple[str | None, dict | None]:
    auth_env = request.get("auth_env") or default_env
    token = env_get(auth_env)
    if not token:
        return None, err("auth_error", 0, 0, f"env var {auth_env!r} unset")
    return token, None


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    return ""


def json_args(arguments: Any) -> dict:
    """OpenAI tool_call arguments (a JSON *string*) -> the object the native
    provider APIs want. Malformed/non-object args degrade to ``{}`` rather
    than aborting the turn."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# Substrings that genuinely mean "the model's context window was exceeded",
# across OpenAI ("maximum context length is N tokens ... reduce the length of
# the messages"), OpenRouter ("This endpoint's maximum context length is N")
# and the OpenAI error code. Deliberately narrow: matching "token"/"maximum"
# alone misreads max_tokens/parameter errors as overflow (see classify_status).
_OVERFLOW_400_SIGNATURES = (
    "context length",            # "(maximum) context length is N tokens"
    "context window",
    "context_length_exceeded",   # OpenAI error code
    "reduce the length of the messages",
)


def provider_error_message(err_body: Any) -> str:
    """The most specific human-readable reason from a parsed provider error body.

    OpenAI-compatible providers put the reason in ``error.message``. OpenRouter
    relays an upstream failure as a generic envelope —
    ``{"error": {"message": "Provider returned error",
    "metadata": {"raw": "<json string of the UPSTREAM error>"}}}`` — where the
    real reason (e.g. "Invalid 'max_output_tokens': ... >= 16") lives in
    ``metadata.raw``, not the outer message. Dig it out so both classification
    and the caller-facing error carry the truth instead of "Provider returned
    error". Falls back to the stringified body for unknown shapes."""
    if not isinstance(err_body, dict):
        return str(err_body)
    error = err_body.get("error")
    if isinstance(error, str):
        return error
    if not isinstance(error, dict):
        return str(err_body)
    inner = _openrouter_relayed_message(error.get("metadata"))
    if inner:
        return inner
    outer = str(error.get("message") or "").strip()
    return outer or str(err_body)


def _openrouter_relayed_message(metadata: Any) -> "str | None":
    """The upstream ``error.message`` OpenRouter stashes in ``metadata.raw``."""
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("raw")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return raw.strip() or None
    if isinstance(raw, dict):
        inner = raw.get("error")
        if isinstance(inner, dict):
            return str(inner.get("message") or "").strip() or None
        if isinstance(inner, str):
            return inner.strip() or None
    return None


def classify_status(status: int, err_msg: str) -> str:
    if status in (401, 403):
        return "auth_error"
    if status == 429:
        return "rate_limit"
    if status == 402:
        return "payment_required"
    if status in (408, 504):
        return "timeout"
    if status == 404:
        return "model_unavailable"
    if status == 400:
        m = (err_msg or "").lower()
        # Only a genuine "the context window was exceeded" signature is a
        # context_overflow. The previous catch-all (any of "context"/"token"/
        # "length"/"maximum" anywhere in the body) mislabelled unrelated 400s as
        # overflow — most perversely "Invalid 'max_output_tokens': integer below
        # minimum value. Expected a value >= 16, but got 4" (max_tokens too
        # SMALL) tripped on the "token" inside "max_output_tokens" and was
        # reported as the context being too BIG, then aborted. Everything that
        # is not an overflow is a bad_request: it falls through to the next
        # candidate and carries the real provider message to the caller.
        if any(sig in m for sig in _OVERFLOW_400_SIGNATURES):
            return "context_overflow"
        return "bad_request"
    if 500 <= status < 600:
        return "server_error"
    return "unknown"


def err(kind: str, status: int, latency_ms: int, message: str) -> dict:
    return {
        "ok":            False,
        "error_kind":    kind,
        "http_status":   status,
        "latency_ms":    latency_ms,
        "error_message": message,
    }


def elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def first_token_timeout_s(request: dict) -> "float | None":
    raw = request.get("first_token_timeout_ms")
    try:
        seconds = float(raw) / 1000.0
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


async def before_first_output(
    awaitable,
    timeout_s: "float | None",
    t0: float,
    saw_output: Callable[[], bool],
):
    if timeout_s is None or saw_output():
        return await awaitable
    remaining = timeout_s - (time.monotonic() - t0)
    if remaining <= 0:
        raise asyncio.TimeoutError
    return await asyncio.wait_for(awaitable, timeout=remaining)


def first_token_timeout_err(timeout_s: float, latency_ms: int) -> dict:
    return err("timeout", 0, latency_ms,
               f"first token timed out after {int(timeout_s * 1000)}ms")


# Back-compat names used by older tests/scripts that import through
# llm_router_host's re-export layer.
_cached_tokens = cached_tokens
_classify_status = classify_status
_err = err
_elapsed_ms = elapsed_ms
_first_token_timeout_s = first_token_timeout_s
_before_first_output = before_first_output
_first_token_timeout_err = first_token_timeout_err
_provider_error_message = provider_error_message
