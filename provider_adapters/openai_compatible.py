"""OpenAI-compatible /chat/completions provider adapter."""
from __future__ import annotations

import asyncio
import os
from typing import Callable, Any

from provider_adapters.common import (
    CallProviderHook,
    AsyncCallProviderHook,
    TokenProvider,
    _cached_tokens,
    _classify_status,
    _elapsed_ms,
    _err,
    _provider_error_message,
)


def _resolve_auth_headers(
    request: dict,
    env_get: Callable[[str], str | None],
    token_providers: dict[str, TokenProvider] | None = None,
) -> tuple[dict | None, dict | None]:
    """Map a provider's auth descriptor to request headers."""
    auth = request.get("auth")
    auth = auth if isinstance(auth, dict) else None
    kind = auth.get("kind") if auth else None
    if kind is None and request.get("auth_env"):
        kind, auth = "bearer", {"kind": "bearer", "env": request.get("auth_env")}

    if kind in (None, "none"):
        return {}, None
    if kind == "bearer":
        env = (auth or {}).get("env") or request.get("auth_env")
        token = env_get(env) if env else None
        if not token:
            return None, _err("auth_error", 0, 0, f"env var {env!r} unset")
        return {"Authorization": f"Bearer {token}"}, None
    if kind == "oauth":
        provider = (auth or {}).get("provider")
        getter = (token_providers or {}).get(provider)
        if getter is None:
            return None, _err("auth_error", 0, 0,
                              f"no oauth token provider for {provider!r}")
        token = getter()
        if not token:
            return None, _err("auth_error", 0, 0,
                              f"oauth token provider {provider!r} returned nothing")
        return {"Authorization": f"Bearer {token}"}, None
    return None, _err("auth_error", 0, 0, f"unknown auth kind {kind!r}")


def _resolve_ollama_cloud_auth(
    env_get: Callable[[str], str | None],
    url: str,
    method: str = "POST",
    body: bytes = b"",
) -> dict | None:
    """Resolve auth headers for Ollama Cloud via OLLAMA_API_KEY."""
    api_key = env_get("OLLAMA_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return None


def _prepare_openai_call(
    request: dict,
    env_get: Callable[[str], str | None],
    extra: dict[str, str],
    timeout_s: float,
    token_providers: dict[str, TokenProvider] | None = None,
) -> tuple[tuple | None, dict | None]:
    """Build (url, body, headers, timeout_s) for an OpenAI-compatible call."""
    auth_headers, err = _resolve_auth_headers(request, env_get, token_providers)
    if err is not None:
        return None, err

    offer = request.get("offer") or {}
    body: dict = {
        "model": offer.get("wire_model_id") or request["served_model_id"],
        "messages": request.get("messages") or [],
    }
    for field in ("tools", "response_format", "temperature", "seed", "max_tokens"):
        v = request.get(field)
        if v is not None:
            body[field] = v

    url = (request.get("base_url") or "").rstrip("/") + "/chat/completions"
    base_url = request.get("base_url") or ""
    provider_id = request.get("provider_id") or ""
    seller_endpoint = offer.get("seller_endpoint") or ""

    is_ollama = (
        provider_id == "ollama" or
        "ollama.com" in base_url or
        "ollama.com" in seller_endpoint or
        "localhost:11434" in base_url or
        "127.0.0.1:11434" in base_url or
        base_url.rstrip("/").endswith(":11434/v1")
    )

    if is_ollama:
        endpoint = seller_endpoint or base_url
        if endpoint.startswith("https://ollama.com"):
            auth_headers = _resolve_ollama_cloud_auth(
                env_get, url, method="POST", body=b""
            )
            if auth_headers is None:
                return None, _err("auth_error", 0, 0,
                                  "Ollama Cloud requires OLLAMA_API_KEY")
        else:
            auth_headers = {}

    headers = {"Content-Type": "application/json", **auth_headers, **extra}
    peer_id = offer.get("peer_id")
    if peer_id:
        headers["x-antseed-pin-peer"] = peer_id
    timeout = (request.get("timeout_ms") or int(timeout_s * 1000)) / 1000.0
    return (url, body, headers, timeout), None


def make_http_call_provider(
    env_get: Callable[[str], str | None] | None = None,
    timeout_s: float = 30.0,
    extra_headers: dict[str, str] | None = None,
    token_providers: dict[str, TokenProvider] | None = None,
    provider_rules: dict[str, dict] | None = None,
) -> CallProviderHook:
    """Synchronous OpenAI-compatible provider backend."""
    import httpx

    _env_get = env_get or os.environ.get
    _extra = dict(extra_headers or {})

    def call(request: dict) -> dict:
        api_kind = request.get("api_kind", "openai_compatible")
        if api_kind != "openai_compatible":
            return _err("unsupported_api_kind", 0, 0,
                        f"api_kind={api_kind!r} not supported by HTTP backend")

        prep, err = _prepare_openai_call(
            request, _env_get, _extra, timeout_s, token_providers)
        if err is not None:
            return err
        url, body, headers, timeout = prep

        import time as _time
        t0 = _time.monotonic()
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
        except httpx.TimeoutException:
            return _err("timeout", 0, _elapsed_ms(t0), f"POST {url} timed out")
        except (httpx.NetworkError, httpx.RequestError) as e:
            return _err("network_error", 0, _elapsed_ms(t0), str(e))

        rules = (provider_rules or {}).get(request.get("provider_id")) or {}
        return _parse_openai_response(
            resp, _elapsed_ms(t0), error_map=rules.get("error_map"))

    return call


_PEER_GATES: dict[str, asyncio.Semaphore] = {}


def _peer_gate(peer_id: str, cap: int) -> asyncio.Semaphore:
    sem = _PEER_GATES.get(peer_id)
    if sem is None:
        sem = asyncio.Semaphore(cap)
        _PEER_GATES[peer_id] = sem
    return sem


def make_async_call_provider(
    env_get: Callable[[str], str | None] | None = None,
    timeout_s: float = 30.0,
    extra_headers: dict[str, str] | None = None,
    client: Any = None,
    token_providers: dict[str, TokenProvider] | None = None,
    provider_rules: dict[str, dict] | None = None,
) -> AsyncCallProviderHook:
    """Async OpenAI-compatible provider backend."""
    import httpx

    _env_get = env_get or os.environ.get
    _extra = dict(extra_headers or {})

    async def call(request: dict) -> dict:
        api_kind = request.get("api_kind", "openai_compatible")
        if api_kind != "openai_compatible":
            return _err("unsupported_api_kind", 0, 0,
                        f"api_kind={api_kind!r} not supported by HTTP backend")

        prep, err = _prepare_openai_call(
            request, _env_get, _extra, timeout_s, token_providers)
        if err is not None:
            return err
        url, body, headers, timeout = prep

        import time as _time
        t0 = _time.monotonic()
        offer = request.get("offer") or {}
        peer_id = offer.get("peer_id")
        cap = offer.get("max_concurrency")
        gate = _peer_gate(peer_id, cap) if (
            peer_id and isinstance(cap, int) and cap > 0) else None
        if gate is not None:
            try:
                await asyncio.wait_for(gate.acquire(), timeout=timeout)
            except (asyncio.TimeoutError, TimeoutError):
                return _err("rate_limit", 0, _elapsed_ms(t0),
                            f"antseed peer {peer_id[:10]} in-flight cap {cap} saturated")
        try:
            try:
                if client is not None:
                    resp = await client.post(
                        url, json=body, headers=headers, timeout=timeout)
                else:
                    async with httpx.AsyncClient() as c:
                        resp = await c.post(
                            url, json=body, headers=headers, timeout=timeout)
            except httpx.TimeoutException:
                result = _err("timeout", 0, _elapsed_ms(t0),
                              f"POST {url} timed out")
            except (httpx.NetworkError, httpx.RequestError) as e:
                result = _err("network_error", 0, _elapsed_ms(t0), str(e))
            else:
                rules = (provider_rules or {}).get(request.get("provider_id")) or {}
                result = _parse_openai_response(
                    resp, _elapsed_ms(t0), error_map=rules.get("error_map"))
            return result
        finally:
            if gate is not None:
                gate.release()

    return call


def _classify_from_map(err_msg: str, error_map: dict | None) -> str | None:
    """Provider-declared body-substring -> canonical kind."""
    if not error_map:
        return None
    msg = (err_msg or "").lower()
    for needle, kind in error_map.items():
        if needle.lower() in msg:
            return str(kind)
    return None


def _parse_openai_response(
    resp: Any,
    latency: int,
    error_map: dict | None = None,
) -> dict:
    """Translate an OpenAI-compatible response into the router response shape."""
    status = resp.status_code
    if 200 <= status < 300:
        try:
            data = resp.json()
        except Exception as e:
            return _err("bad_response", status, latency, f"json parse: {e}")

        choices = data.get("choices") or []
        if not choices:
            return _err("bad_response", status, latency, "no choices in response")

        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish == "content_filter":
            return _err("content_filter", status, latency,
                        "blocked by provider filter")

        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        text = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")
        if not str(text).strip() and not tool_calls:
            return _err("bad_response", status, latency, "empty assistant content")
        return {
            "ok": True,
            "latency_ms": latency,
            "response": {
                "text": text,
                "tool_calls": tool_calls,
                "finish_reason": finish,
                "tokens_in": usage.get("prompt_tokens"),
                "tokens_out": usage.get("completion_tokens"),
                "tokens_total": usage.get("total_tokens"),
                "tokens_cached": _cached_tokens(usage),
                "cost_reported": usage.get("cost"),
                "raw_model": data.get("model"),
            },
        }

    try:
        err_body = resp.json()
        err_msg = _provider_error_message(err_body)
    except Exception:
        err_msg = (resp.text or "")[:500]
    kind = _classify_from_map(err_msg, error_map) or _classify_status(status, err_msg)
    return _err(kind, status, latency, err_msg[:500])
