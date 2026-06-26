"""Shared provider-adapter primitives."""
from __future__ import annotations

import json
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
        if "context" in m or "token" in m or "length" in m or "maximum" in m:
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


# Back-compat names used by older tests/scripts that import through
# llm_router_host's re-export layer.
_cached_tokens = cached_tokens
_classify_status = classify_status
_err = err
_elapsed_ms = elapsed_ms
