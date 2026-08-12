"""Stateless client for the internal, PVC-owning Codex broker.

The public router replicas must not mount or copy ChatGPT refresh credentials.
They send only the already-normalized provider request over the cluster network;
the broker selects an account, performs the call, and returns the canonical
router result plus sanitized quota signals.
"""
from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable

import httpx

from provider_adapters.common import _err

Emit = Callable[[str], Awaitable[None]]


class RemoteCodexClient:
    def __init__(self, base_url: str, token: str, *,
                 timeout_s: float = 120.0, client: Any = None):
        base_url = str(base_url or "").strip().rstrip("/")
        token = str(token or "").strip()
        if not base_url:
            raise ValueError("CODEX_BROKER_URL must not be empty")
        if not token:
            raise ValueError("CODEX_BROKER_TOKEN must not be empty")
        self.base_url = base_url
        self.timeout_s = max(1.0, float(timeout_s))
        self._headers = {"authorization": f"Bearer {token}"}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100,
                                max_keepalive_connections=40))

    def _timeout(self, request: dict) -> float:
        requested = request.get("timeout_ms")
        if isinstance(requested, (int, float)) and requested > 0:
            return max(1.0, float(requested) / 1000.0 + 5.0)
        return self.timeout_s + 5.0

    @staticmethod
    def _replay_signals(payload: Any, observe) -> None:
        if observe is None or not isinstance(payload, list):
            return
        for raw in payload[:16]:
            if not isinstance(raw, dict):
                continue
            headers = raw.get("headers")
            signal = {
                "status": int(raw.get("status") or 0),
                "headers": {
                    str(k)[:100]: str(v)[:1000]
                    for k, v in (headers.items()
                                 if isinstance(headers, dict) else [])
                },
                "ts": int(raw.get("ts") or time.time()),
            }
            try:
                observe(signal)
            except Exception:
                pass

    @staticmethod
    def _http_error(status: int, detail: str) -> dict:
        if status in (401, 403):
            kind = "auth_error"
        elif status == 429:
            kind = "rate_limit"
        else:
            kind = "server_error"
        return _err(kind, status, 0,
                    f"Codex broker returned {status}: {detail[:300]}")

    async def call(self, request: dict, *, observe=None) -> dict:
        try:
            response = await self._client.post(
                f"{self.base_url}/v1/call", json=request,
                headers=self._headers, timeout=self._timeout(request))
        except httpx.TimeoutException:
            return _err("timeout", 0, 0, "Codex broker timed out")
        except (httpx.NetworkError, httpx.RequestError) as exc:
            return _err("network_error", 0, 0,
                        f"Codex broker unavailable: {exc}")
        if response.status_code != 200:
            return self._http_error(response.status_code, response.text)
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            return _err("bad_response", 200, 0,
                        f"invalid Codex broker response: {exc}")
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
            return _err("bad_response", 200, 0,
                        "Codex broker response has no result")
        self._replay_signals(payload.get("signals"), observe)
        return payload["result"]

    async def stream(self, request: dict, emit: Emit, *, observe=None) -> dict:
        emitted = False
        try:
            async with self._client.stream(
                    "POST", f"{self.base_url}/v1/stream", json=request,
                    headers=self._headers,
                    timeout=self._timeout(request)) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode(
                        "utf-8", "replace")[:500]
                    return self._http_error(response.status_code, detail)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        return _err("stream_interrupted" if emitted else "bad_response",
                                    200, 0, "invalid Codex broker stream event")
                    if not isinstance(event, dict):
                        continue
                    if event.get("event") == "delta":
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            await emit(delta)
                            emitted = True
                    elif event.get("event") == "result":
                        result = event.get("result")
                        if not isinstance(result, dict):
                            return _err("bad_response", 200, 0,
                                        "Codex broker stream has invalid result")
                        self._replay_signals(event.get("signals"), observe)
                        return result
        except httpx.TimeoutException:
            return _err("stream_interrupted" if emitted else "timeout", 0, 0,
                        "Codex broker stream timed out")
        except (httpx.NetworkError, httpx.RequestError) as exc:
            return _err("stream_interrupted" if emitted else "network_error", 0, 0,
                        f"Codex broker stream unavailable: {exc}")
        return _err("stream_interrupted" if emitted else "bad_response", 200, 0,
                    "Codex broker stream ended without a result")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
