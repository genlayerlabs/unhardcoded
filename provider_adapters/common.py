"""Shared provider-adapter primitives."""
from __future__ import annotations

import json
import asyncio
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
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


CACHE_CONTROL = {"type": "ephemeral"}


def anthropic_cache_family(request: dict) -> bool:
    """Whether the chosen route serves an Anthropic-class model — the only
    family whose prompt cache is OPT-IN per request (`cache_control`
    breakpoints). Only the router can inject them: policy-driven clients are
    model-agnostic by design and never know the winner (#74)."""
    fam = str(request.get("model_family") or request.get("served_model_id") or "")
    return "claude" in fam.lower()


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


async def ignore_delta(_delta: str) -> None:
    """A no-op emit for driving a streaming backend as a NON-streaming call
    (the first-token deadline path): the deltas are discarded, only the
    aggregated final response is returned."""
    return None


@dataclass
class StreamAcc:
    """Mutable accumulator an SSE event handler folds a native stream into, then
    the adapter reads to build its complete-response dict. `tool_calls` is left
    handler-owned (a dict keyed by content index, or a list) so each provider
    accumulates in its own shape."""
    t0: float
    status: int = 0
    text_parts: list = field(default_factory=list)
    tool_calls: Any = None
    finish_reason: Any = None
    usage: dict = field(default_factory=dict)
    raw_model: Any = None
    saw_output: bool = False
    emitted: bool = False

    def latency(self) -> int:
        return elapsed_ms(self.t0)


async def drive_http_sse(
    *,
    client: Any,
    url: str,
    body: dict,
    headers: dict,
    timeout: float,
    request: dict,
    on_event: Callable[[dict, "StreamAcc"], Awaitable["dict | None"]],
) -> "tuple[StreamAcc | None, dict | None]":
    """Drive an httpx Server-Sent-Events stream with the first-token deadline,
    shared by the native httpx adapters (Anthropic, Gemini). Opens the stream
    (guarded by the first-token timeout), maps a non-2xx to a classified error,
    then feeds each decoded `data:` JSON object to `on_event(ev, acc)` — which
    mutates `acc` and returns an error dict to abort or None to continue. Returns
    `(acc, None)` on a clean end or `(None, error_dict)`; the caller finalizes the
    accumulator into its provider-specific response shape. Bedrock is NOT a
    consumer (its transport is the boto3 event stream, not httpx SSE)."""
    acc = StreamAcc(t0=time.monotonic())
    timeout_s = first_token_timeout_s(request)

    def _saw() -> bool:
        return acc.saw_output

    try:
        async with AsyncExitStack() as stack:
            try:
                resp = await before_first_output(
                    stack.enter_async_context(
                        client.stream("POST", url, json=body, headers=headers,
                                      timeout=timeout)),
                    timeout_s, acc.t0, _saw)
            except (asyncio.TimeoutError, TimeoutError):
                return None, first_token_timeout_err(timeout_s, acc.latency())
            acc.status = resp.status_code
            if not (200 <= resp.status_code < 300):
                raw = (await resp.aread()).decode("utf-8", "replace")[:500]
                return None, err(classify_status(resp.status_code, raw),
                                 resp.status_code, acc.latency(), raw)

            lines = resp.aiter_lines().__aiter__()
            while True:
                try:
                    line = await before_first_output(
                        lines.__anext__(), timeout_s, acc.t0, _saw)
                except StopAsyncIteration:
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    if not acc.saw_output:
                        return None, first_token_timeout_err(timeout_s, acc.latency())
                    raise
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                aborted = await on_event(ev, acc)
                if aborted is not None:
                    return None, aborted
    except Exception as exc:  # noqa: BLE001
        if acc.emitted:
            partial = "".join(acc.text_parts)
            return None, err("stream_interrupted", 0, acc.latency(),
                             f"{type(exc).__name__}: {exc} (partial: {partial[:200]!r})")
        return None, err("network_error", 0, acc.latency(),
                         f"{type(exc).__name__}: {exc}")
    return acc, None


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
