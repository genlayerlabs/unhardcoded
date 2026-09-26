"""OpenAI-compatible /chat/completions provider adapter."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import AsyncExitStack, suppress
from typing import Awaitable, Callable, Any

from provider_adapters.common import (
    CACHE_CONTROL,
    CallProviderHook,
    AsyncCallProviderHook,
    TokenProvider,
    anthropic_cache_family,
    before_first_output,
    first_token_timeout_err,
    first_token_timeout_s,
    _cached_tokens,
    _classify_status,
    _elapsed_ms,
    _err,
    _provider_error_message,
)

from provider_adapters.diagnostics import ProviderTiming, upstream_metadata

Emit = Callable[[str], Awaitable[None]]


def _has_reasoning_content(message: dict) -> bool:
    """Empty deltas and detail metadata alone are not generated output."""
    def nonblank(value):
        return isinstance(value, str) and bool(value.strip())

    if any(nonblank(message.get(key)) for key in ("reasoning", "reasoning_content")):
        return True
    details = message.get("reasoning_details")
    return isinstance(details, list) and any(
        isinstance(part, dict) and any(nonblank(part.get(key))
                                      for key in ("text", "summary", "data", "encrypted_content"))
        for part in details)


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


def _with_cache_breakpoints(messages: list[dict]) -> list[dict]:
    """COPY of `messages` with cache_control on the first system message and
    on the last message (rolling breakpoint — the next call re-reads its
    whole history from cache). Never mutates the caller's list: the request
    dict is shared with retries and other rank candidates."""

    def _tag(msg: dict) -> dict:
        tagged = dict(msg)
        content = tagged.get("content")
        if isinstance(content, str):
            tagged["content"] = [{"type": "text", "text": content,
                                  "cache_control": dict(CACHE_CONTROL)}]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            parts = [dict(p) for p in content]
            parts[-1]["cache_control"] = dict(CACHE_CONTROL)
            tagged["content"] = parts
        return tagged

    out = list(messages or [])
    for i, msg in enumerate(out):
        if isinstance(msg, dict) and msg.get("role") == "system":
            out[i] = _tag(msg)
            break
    if out and isinstance(out[-1], dict):
        out[-1] = _tag(out[-1])
    return out


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
    for field in ("tools", "response_format", "temperature", "seed", "max_tokens",
                  "reasoning", "reasoning_effort"):
        v = request.get(field)
        if v is not None:
            body[field] = v

    # Prompt-cache breakpoints for anthropic-class routes (#74), on surfaces
    # known to RELAY cache_control in OpenAI-format content parts (openrouter
    # documents this). Anthropic caching is opt-in per request: without the
    # markers an agentic session re-buys its whole prefix at full input price
    # on every call. Gated fail-safe: unknown surfaces are left untouched.
    if anthropic_cache_family(request) and (request.get("provider_id") or "") in (
        "openrouter",
        "openrouter_market",
    ):
        body["messages"] = _with_cache_breakpoints(body["messages"])

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
    if provider_id.startswith("antseed") and os.getenv("ANTSEED_PROXY_TOKEN"):
        # The operator's funded buyer proxy (antseed/public-proxy.js) requires it.
        headers["Authorization"] = "Bearer " + os.environ["ANTSEED_PROXY_TOKEN"]
    from byo_http import is_byo_buyer
    if is_byo_buyer(request, env_get):
        # Never trust an offer's endpoint to choose where a tenant secret goes.
        url = env_get('ANTSEED_BYO_URL').rstrip('/') + '/v1/chat/completions'
        headers['Authorization'] = 'Bearer ' + env_get('ANTSEED_BYO_TOKEN')
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
        if request.get('protocol') == 'decisions':
            return _err('unsupported_api_kind', 0, 0, 'Decision requests require the async decision transport')
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
_PEER_GATE_LOCK: asyncio.Lock | None = None

# Maximum idle time (seconds) before a peer gate is evicted to prevent
# unbounded memory growth when peers are dynamically discovered and retired.
_PEER_GATE_TTL: float = 600.0
_PEER_GATE_TIMESTAMPS: dict[str, float] = {}


def _peer_gate_lock() -> asyncio.Lock:
    """Lazy-initialised lock to avoid creating asyncio primitives at import time
    (which may happen outside an event loop in test/synchronous contexts)."""
    global _PEER_GATE_LOCK
    if _PEER_GATE_LOCK is None:
        _PEER_GATE_LOCK = asyncio.Lock()
    return _PEER_GATE_LOCK


async def _peer_gate(peer_id: str, cap: int) -> asyncio.Semaphore:
    """Return a per-peer concurrency semaphore, creating one if needed.

    Uses a lock to prevent a race where two concurrent coroutines both see
    ``None`` and create separate semaphores for the same peer_id — the
    second would overwrite the first, silently dropping any coroutines
    already waiting on it.

    Old entries are evicted after ``_PEER_GATE_TTL`` seconds of inactivity
    so that the dict does not grow without bound when AntSeed peers are
    dynamically discovered and later retired.
    """
    import time as _time

    lock = _peer_gate_lock()
    async with lock:
        # Evict stale entries first (best-effort, not on every call to
        # avoid quadratic sweep cost — only when the lock is already held
        # and the dict is getting large).
        if len(_PEER_GATES) > 256:
            now = _time.monotonic()
            stale = [k for k, ts in _PEER_GATE_TIMESTAMPS.items()
                     if now - ts > _PEER_GATE_TTL]
            for k in stale:
                _PEER_GATES.pop(k, None)
                _PEER_GATE_TIMESTAMPS.pop(k, None)

        sem = _PEER_GATES.get(peer_id)
        if sem is None:
            sem = asyncio.Semaphore(cap)
            _PEER_GATES[peer_id] = sem
        _PEER_GATE_TIMESTAMPS[peer_id] = _time.monotonic()
        return sem


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _peer_gate_wait_s(call_timeout_s: float) -> float:
    raw = os.getenv("PEER_GATE_WAIT_S", "").strip()
    if not raw:
        return max(0.01, float(call_timeout_s))
    try:
        return max(0.01, min(float(call_timeout_s), float(raw)))
    except (TypeError, ValueError):
        return max(0.01, float(call_timeout_s))


def _peer_lease_ttl_s() -> float:
    try:
        return max(15.0, min(3600.0, float(os.getenv(
            "PEER_LEASE_TTL_S", "120"))))
    except (TypeError, ValueError):
        return 120.0


class _PeerCapacitySlot:
    """One local semaphore permit or one renewable cross-replica DB lease."""

    def __init__(self, *, semaphore: asyncio.Semaphore | None = None,
                 peer_id: str | None = None, lease_id: str | None = None,
                 ttl_s: float = 120.0):
        self._semaphore = semaphore
        self._peer_id = peer_id
        self._lease_id = lease_id
        self._ttl_s = ttl_s
        self._heartbeat: asyncio.Task[None] | None = None
        self._released = False
        if peer_id and lease_id:
            self._heartbeat = asyncio.create_task(self._renew_loop())

    async def _renew_loop(self) -> None:
        import host_store

        while True:
            await asyncio.sleep(max(5.0, self._ttl_s / 3.0))
            ok = await asyncio.to_thread(
                host_store.renew_peer_lease,
                self._peer_id, self._lease_id, ttl_s=self._ttl_s)
            if not ok:
                return

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat
        if self._peer_id and self._lease_id:
            import host_store
            await asyncio.to_thread(
                host_store.release_peer_lease,
                self._peer_id, self._lease_id)
        elif self._semaphore is not None:
            self._semaphore.release()


async def _acquire_peer_capacity(
        request: dict, call_timeout_s: float, env_get=None) \
        -> tuple[_PeerCapacitySlot | None, str | None]:
    """Acquire the selected seller's cap locally or across all replicas.

    The distributed path is enabled only by Kubernetes. Local development keeps
    the in-process semaphore and does not require PostgreSQL.
    """
    offer = request.get("offer") or {}
    peer_id = offer.get("peer_id")
    cap = offer.get("max_concurrency")
    if not peer_id or not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        return None, None
    peer_id = str(peer_id).strip().lower()
    from byo_http import is_byo_buyer
    if env_get is not None and is_byo_buyer(request, env_get):
        # A BYO offer's peer_id/max_concurrency come from the tenant's gateway;
        # never let them size or starve the operator's shared gate for that peer.
        peer_id = f"tenant:{env_get('SAAS_TENANT_SCOPE')}:{peer_id}"
    wait_s = _peer_gate_wait_s(call_timeout_s)
    if not _enabled("DISTRIBUTED_PEER_GATES"):
        gate = await _peer_gate(peer_id, cap)
        try:
            await asyncio.wait_for(gate.acquire(), timeout=wait_s)
        except (asyncio.TimeoutError, TimeoutError):
            return None, "saturated"
        return _PeerCapacitySlot(semaphore=gate), None

    import host_store

    lease_id = uuid.uuid4().hex
    ttl_s = _peer_lease_ttl_s()
    deadline = time.monotonic() + wait_s
    while True:
        acquired, store_ok = await asyncio.to_thread(
            host_store.try_acquire_peer_lease,
            peer_id, lease_id, cap, ttl_s=ttl_s)
        if not store_ok:
            return None, "unavailable"
        if acquired:
            return _PeerCapacitySlot(
                peer_id=peer_id, lease_id=lease_id, ttl_s=ttl_s), None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, "saturated"
        await asyncio.sleep(min(0.1, remaining))


def _peer_capacity_error(peer_id: str, cap: int, reason: str,
                         started: float) -> dict:
    kind = "network_error" if reason == "unavailable" else "rate_limit"
    detail = ("shared peer gate unavailable" if reason == "unavailable" else
              f"in-flight cap {cap} saturated")
    return _err(kind, 0, _elapsed_ms(started),
                f"antseed peer {peer_id[:10]} {detail}")


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
        if request.get("protocol") == "decisions":
            from provider_adapters.decisions import call_decisions
            return await call_decisions(request, env_get=_env_get, client=client,
                timeout_s=timeout_s, extra_headers=_extra, token_providers=token_providers)
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
        # The stream-under-the-hood branch acquires inside the streaming
        # adapter. Acquiring here as well would deadlock a cap=1 peer against
        # this same request.
        uses_streaming_backend = request.get("first_token_timeout_ms") is not None
        timing = ProviderTiming(request, timeout, "buffered_sse" if uses_streaming_backend else "buffered")
        slot = None
        if not uses_streaming_backend:
            slot, gate_error = await _acquire_peer_capacity(request, timeout, _env_get)
            if gate_error:
                return timing.attach(_peer_capacity_error(
                    str(peer_id or ""), int(cap or 0), gate_error, t0))
        try:
            try:
                # HTTPX limits inactivity between reads; a trickling response can
                # exceed it indefinitely. Bound the complete buffered call, including
                # time already spent waiting for peer capacity.
                remaining = max(0.0, timeout - (_time.monotonic() - t0))
                deadline = asyncio.timeout(remaining)
                async with deadline:
                    if uses_streaming_backend:
                        # Reuse the streaming backend (defined below in this module) to
                        # get a first-token bound, discarding deltas — a non-stream call.
                        async def _ignore_delta(_delta: str) -> None:
                            return None

                        result = await stream_openai_compatible(
                            request,
                            _ignore_delta,
                            client=client,
                            env_get=_env_get,
                            extra_headers=_extra,
                            timeout_s=timeout_s,
                            token_providers=token_providers,
                            provider_rules=provider_rules, _timing=timing,
                        )
                    else:
                        from byo_http import buyer_client, is_byo_buyer
                        if is_byo_buyer(request, _env_get):
                            async with buyer_client() as buyer:
                                timing.data["phase"] = "connection_pool"
                                resp = await _post_bounded(buyer, url, json=body, headers=headers, timeout=timeout, **timing.http_options(buyer))
                        elif client is not None:
                            timing.data["phase"] = "connection_pool"
                            resp = await client.post(
                                url, json=body, headers=headers, timeout=timeout, **timing.http_options(client))
                        else:
                            async with httpx.AsyncClient() as c:
                                timing.data["phase"] = "connection_pool"
                                resp = await c.post(
                                    url, json=body, headers=headers, timeout=timeout, **timing.http_options(c))
                        rules = (provider_rules or {}).get(request.get("provider_id")) or {}
                        result = _parse_openai_response(
                            resp, _elapsed_ms(t0), error_map=rules.get("error_map"))
            except (TimeoutError, httpx.TimeoutException) as exc:
                timing.data["timeout_source"] = "attempt_deadline" if deadline.expired() else type(exc).__name__
                result = _err("timeout", 0, _elapsed_ms(t0),
                              f"POST {url} timed out")
            except (httpx.NetworkError, httpx.RequestError) as e:
                result = _err("network_error", 0, _elapsed_ms(t0), str(e))
            except ResponseTooLarge as e:
                result = _err("bad_response", 200, _elapsed_ms(t0), str(e))
            return timing.attach(result)
        finally:
            if slot is not None:
                await slot.release()

    return call


class ResponseTooLarge(Exception):
    """A tenant (BYO) gateway sent more than the configured byte caps."""


def _byo_caps() -> tuple[int, int]:
    return (int(os.getenv("BYO_MAX_RESPONSE_BYTES", str(32 << 20))),
            int(os.getenv("BYO_MAX_SSE_LINE_BYTES", str(1 << 20))))


async def _post_bounded(http, url: str, **kwargs):
    """POST and buffer at most BYO_MAX_RESPONSE_BYTES of the body."""
    import httpx
    cap = _byo_caps()[0]
    async with http.stream("POST", url, **kwargs) as resp:
        raw = bytearray()
        async for chunk in resp.aiter_bytes():
            raw.extend(chunk)
            if len(raw) > cap:
                raise ResponseTooLarge(f"response exceeds {cap} bytes")
        return httpx.Response(resp.status_code, headers=resp.headers,
                              content=bytes(raw), request=resp.request)


async def _read_head(resp, limit: int = 4096) -> bytes:
    raw = b""
    async for chunk in resp.aiter_bytes():
        raw += chunk
        if len(raw) >= limit:
            break
    return raw[:limit]


async def _bounded_lines(resp):
    """aiter_lines with a per-line and a total byte cap."""
    total_cap, line_cap = _byo_caps()
    buf, total = b"", 0
    async for chunk in resp.aiter_bytes():
        total += len(chunk)
        if total > total_cap:
            raise ResponseTooLarge(f"stream exceeds {total_cap} bytes")
        buf += chunk
        *lines, buf = buf.split(b"\n")
        if len(buf) > line_cap or any(len(line) > line_cap for line in lines):
            raise ResponseTooLarge(f"SSE line exceeds {line_cap} bytes")
        for line in lines:
            yield line.rstrip(b"\r").decode("utf-8", "replace")
    if buf:
        yield buf.rstrip(b"\r").decode("utf-8", "replace")


def _classify_from_map(err_msg: str, error_map: dict | None) -> str | None:
    """Provider-declared body-substring -> canonical kind."""
    if not error_map:
        return None
    msg = (err_msg or "").lower()
    for needle, kind in error_map.items():
        if needle.lower() in msg:
            return str(kind)
    return None


async def stream_openai_compatible(
    request: dict,
    emit: Emit,
    *,
    client: Any = None,
    env_get=None,
    extra_headers: dict | None = None,
    timeout_s: float = 45.0,
    token_providers: dict | None = None,
    provider_rules: dict[str, dict] | None = None,
    _timing: ProviderTiming | None = None,
) -> dict:
    timing = _timing or ProviderTiming(request, (request.get("timeout_ms") or timeout_s * 1000) / 1000, "streaming")
    result = await _stream_openai_compatible_impl(
        request, emit, env_get=env_get, extra_headers=extra_headers, client=client,
        timeout_s=timeout_s, token_providers=token_providers,
        provider_rules=provider_rules, _timing=timing)
    return timing.attach(result)


async def _stream_openai_compatible_impl(
    request: dict,
    emit: Emit,
    *,
    client: Any = None,
    env_get=None,
    extra_headers: dict | None = None,
    timeout_s: float = 45.0,
    token_providers: dict | None = None,
    provider_rules: dict[str, dict] | None = None,
    _timing: ProviderTiming,
) -> dict:
    """The OpenAI-compatible STREAMING wire backend (sibling of `call`). Returns the
    SAME complete-response dict the non-streaming backend does, so the core's
    fallback/retry is wire-agnostic. Lives here, beside `call`/`_prepare_openai_call`,
    rather than in `streaming.py` — both are openai-compatible wire backends, and
    keeping it here keeps the adapter leaf from importing the shim-layer module
    (streaming.py re-exports this name). Honors `request.first_token_timeout_ms`: a
    pre-delta timeout returns a classified `timeout` error WITHOUT emitting, so the
    core falls through to the next candidate."""
    prep, err = _prepare_openai_call(
        request, env_get or os.environ.get, dict(extra_headers or {}),
        timeout_s, token_providers)
    if err is not None:
        return err
    url, body, headers, timeout = prep
    body["stream"] = True
    rules = (provider_rules or {}).get(request.get("provider_id")) or {}

    t0 = time.monotonic()
    offer = request.get("offer") or {}
    peer_id = offer.get("peer_id")
    cap = offer.get("max_concurrency")
    slot, gate_error = await _acquire_peer_capacity(request, timeout, env_get or os.environ.get)
    if gate_error:
        return _peer_capacity_error(
            str(peer_id or ""), int(cap or 0), gate_error, t0)

    from byo_http import buyer_client, is_byo_buyer
    _byo = is_byo_buyer(request, env_get or os.environ.get)
    _owns_client = client is None or _byo
    if _owns_client:
        import httpx
        client = buyer_client() if _byo else httpx.AsyncClient()

    emitted = False
    text_parts: list[str] = []
    reasoning_parts: dict[str, list[str]] = {}
    reasoning_details: list[dict] = []
    tool_calls_acc: dict[int, dict] = {}
    finish_reason = None
    usage: dict = {}
    raw_model = None
    saw_output = False
    first_timeout_s = first_token_timeout_s(request)

    def _latency() -> int:
        return int((time.monotonic() - t0) * 1000)

    def _saw_output() -> bool:
        return saw_output

    def _timeout_err() -> dict:
        _timing.data["timeout_source"] = "first_output_deadline"
        return first_token_timeout_err(first_timeout_s, _latency())

    try:
        try:
            async with AsyncExitStack() as stack:
                try:
                    _timing.data["phase"] = "connection_pool"
                    resp = await before_first_output(stack.enter_async_context(
                        client.stream("POST", url, json=body, headers=headers,
                                      timeout=timeout, **_timing.http_options(client))), first_timeout_s, t0, _saw_output)
                except (asyncio.TimeoutError, TimeoutError):
                    return _timeout_err()
                if not (200 <= resp.status_code < 300):
                    raw = (await _read_head(resp) if _byo else await resp.aread()
                           ).decode("utf-8", "replace")[:500]
                    kind = _classify_from_map(raw, rules.get("error_map")) \
                        or _classify_status(resp.status_code, raw)
                    return _err(kind, resp.status_code, _latency(), raw)

                lines = (_bounded_lines(resp) if _byo else resp.aiter_lines()).__aiter__()
                while True:
                    try:
                        line = await before_first_output(
                            lines.__anext__(), first_timeout_s, t0, _saw_output)
                    except StopAsyncIteration:
                        break
                    except (asyncio.TimeoutError, TimeoutError):
                        if not saw_output:
                            return _timeout_err()
                        raise
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    _timing.observe_chunk(chunk)
                    if raw_model is None:
                        raw_model = chunk.get("model")
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        for key in ("reasoning", "reasoning_content"):
                            if isinstance(delta.get(key), str):
                                reasoning_parts.setdefault(key, []).append(delta[key])
                        if isinstance(delta.get("reasoning_details"), list):
                            reasoning_details.extend(delta["reasoning_details"])
                        if _has_reasoning_content(delta):
                            saw_output = True
                        content = delta.get("content")
                        if content:
                            saw_output = True
                            text_parts.append(content)
                            await emit(content)
                            emitted = True
                        for tc in delta.get("tool_calls") or []:
                            saw_output = True
                            idx = tc.get("index", 0)
                            acc = tool_calls_acc.setdefault(idx, {
                                "id": None, "type": "function",
                                "function": {"name": "", "arguments": ""}})
                            if tc.get("id"):
                                acc["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                acc["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                acc["function"]["arguments"] += fn["arguments"]
        except ResponseTooLarge as exc:
            return _err("stream_interrupted" if emitted else "bad_response", 200, _latency(), str(exc))
        except Exception as exc:  # noqa: BLE001 — classified below
            partial = "".join(text_parts)
            if emitted:
                return _err("stream_interrupted", 0, _latency(),
                            f"{type(exc).__name__}: {exc} (partial: {partial[:200]!r})")
            return _err("network_error", 0, _latency(), f"{type(exc).__name__}: {exc}")

        tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)] or None
        text = "".join(text_parts)
        provider_message = {**{k: "".join(v) for k, v in reasoning_parts.items()},
                            **({"reasoning_details": reasoning_details} if reasoning_details else {})}
        if not text.strip() and not tool_calls and not _has_reasoning_content(provider_message):
            return _err("bad_response", 200, _latency(), "empty assistant content")
        return {
            "ok": True,
            "latency_ms": _latency(),
            "response": {
                "text": text,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
                "tokens_in": usage.get("prompt_tokens"),
                "tokens_out": usage.get("completion_tokens"),
                "tokens_total": usage.get("total_tokens"),
                "tokens_cached": _cached_tokens(usage),
                "cost_reported": usage.get("cost"),
                "raw_model": raw_model,
                "provider_message": provider_message,
                "provider_usage": usage,
                "upstream": {k: _timing.data[k] for k in ("upstream_id", "upstream_provider", "tokens_reasoning") if k in _timing.data},
                "tokens_reasoning": _timing.data.get("tokens_reasoning"),
            },
        }
    finally:
        # Close the client if we created it, to prevent connection leaks.
        # Caller-owned clients are their responsibility.
        if _owns_client:
            await client.aclose()
        if slot is not None:
            await slot.release()


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
        if not str(text).strip() and not tool_calls and not _has_reasoning_content(msg):
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
                "provider_message": msg,
                "provider_usage": usage,
                "upstream": upstream_metadata(data),
                "tokens_reasoning": upstream_metadata(data).get("tokens_reasoning"),
            },
        }

    try:
        err_body = resp.json()
        err_msg = _provider_error_message(err_body)
    except Exception:
        err_msg = (resp.text or "")[:500]
    kind = _classify_from_map(err_msg, error_map) or _classify_status(status, err_msg)
    return _err(kind, status, latency, err_msg[:500])
