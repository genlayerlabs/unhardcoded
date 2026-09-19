"""Jev Choice transport. Fixed HTTPS endpoints; never use chat completions.

Credentials are passed explicitly by the caller, never read from process env.
Async cancellation propagates. Retries share the caller's absolute deadline.
"""
from __future__ import annotations

import asyncio
import json
import math
import time

import httpx

from . import DecisionError, DecisionResult, validate_answer

ENDPOINTS = {"openrouter": "https://openrouter.ai/api/alpha/decisions",
             "typesafe": "https://api.typesafe.ai/v1/systemone"}
MODELS = {"openrouter": "typesafe/jev-1.13", "typesafe": "jev-1.13"}
MAX_REQUEST_BYTES = 32000
MAX_RESPONSE_BYTES = 65536


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


class JevDecisionProvider:
    def __init__(self, api_key, *, transport="openrouter", model=None, max_retries=0,
                 client=None):
        if transport not in ENDPOINTS or type(max_retries) is not int or not 0 <= max_retries <= 1:
            raise ValueError("Invalid decision transport configuration")
        self.api_key = api_key
        self.transport, self.model = transport, model or MODELS[transport]
        if not isinstance(self.model, str) or not self.model or "latest" in self.model or len(self.model) > 100:
            raise ValueError("Pin a decision model version")
        self.max_retries, self.client = max_retries, client

    async def decide(self, request, *, deadline):
        request.validate()
        if not self.api_key:
            raise DecisionError("missing_credentials", cost_usd=0)
        body = json.dumps({"model": self.model, "state": request.state,
            "questions": {"selection": {"type": "choice", "instructions": request.instruction,
                "criteria": {c.id: c.description for c in request.choices}}}},
            allow_nan=False, separators=(",", ":")).encode()
        if len(body) > MAX_REQUEST_BYTES:
            raise DecisionError("request_too_large", cost_usd=0)
        started = time.monotonic()
        if deadline <= started:
            raise DecisionError("deadline", cost_usd=0)
        if self.client is not None:
            return await self._run(self.client, body, request, deadline, started)
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            return await self._run(client, body, request, deadline, started)

    async def _run(self, client, body, request, deadline, started):
        attempts = 0
        had_unmetered_attempt = False
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                for attempt in range(self.max_retries + 1):
                    attempts += 1
                    try:
                        async with client.stream("POST", ENDPOINTS[self.transport], content=body,
                                headers={"Authorization": f"Bearer {self.api_key}",
                                         "Content-Type": "application/json"},
                                timeout=max(.001, deadline - time.monotonic()), follow_redirects=False) as response:
                            if response.status_code != 200:
                                transient = response.status_code == 429 or response.status_code >= 500
                                if transient and attempt < self.max_retries:
                                    had_unmetered_attempt = True
                                    await asyncio.sleep(.05)
                                    continue
                                raise DecisionError(f"http_{response.status_code}", attempts=attempts)
                            raw = bytearray()
                            async for chunk in response.aiter_bytes():
                                raw.extend(chunk)
                                if len(raw) > MAX_RESPONSE_BYTES:
                                    raise DecisionError("response_too_large", attempts=attempts)
                            return self._parse(bytes(raw), request, started, attempts, had_unmetered_attempt)
                    except httpx.TransportError:
                        if attempt == self.max_retries:
                            raise DecisionError("transport", attempts=attempts) from None
                        had_unmetered_attempt = True
                        await asyncio.sleep(.05)
        except TimeoutError:
            raise DecisionError("timeout", attempts=attempts) from None

    def _parse(self, raw, request, started, attempts, had_unmetered_attempt):
        cost = None
        try:
            data = json.loads(raw, object_pairs_hook=_unique_object)
            usage = data.get("usage") or {}
            cost = usage.get("cost")
            if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                cost = None
                raise ValueError("Invalid usage")
            tokens = usage.get("input_tokens")
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise ValueError("Invalid usage")
            model = data.get("model")
            if not isinstance(model, str) or not model or len(model) > 160:
                raise ValueError("Missing model")
            selected, probs, confidence, entropy = validate_answer(
                data.get("answers", {}).get("selection"), [c.id for c in request.choices])
            return DecisionResult(selected, probs, confidence, entropy, self.transport, model,
                (time.monotonic() - started) * 1000, None if had_unmetered_attempt else cost, tokens, attempts)
        except (ValueError, AttributeError, TypeError, DecisionError) as exc:
            raise DecisionError(exc.reason if isinstance(exc, DecisionError) else "invalid_response",
                                cost_usd=None if had_unmetered_attempt else cost, attempts=attempts) from None
