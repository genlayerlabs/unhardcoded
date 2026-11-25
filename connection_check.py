"""Isolated, short-lived probes for tenant provider connections."""
from __future__ import annotations

import asyncio
import json
import math
import time

import httpx
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import Request
from fastapi.responses import JSONResponse

import control_plane_client as cp
import host_store
from internal_api import CONNECTION_CHECK_PATH, _gate, connection_check_scope
from provider_adapters.anthropic import _anthropic_request
from provider_adapters.aws_credentials import client as aws_client
from provider_adapters.common import auth_token
from provider_adapters.google import _google_request
from provider_adapters.openai_compatible import _prepare_openai_call
from provider_connections import credential_names
from sources.openrouter import BASE_URL as OPENROUTER_URL
from tenant_providers import buyer_snapshot

TIMEOUT = 4
INTERVAL = 12
MAX_BODY = 4 * 1024 * 1024

_MESSAGES = {
    "ok": "Connection works.",
    "bad_credential": "The provider did not accept the credential.",
    "expired": "The credential has expired.",
    "unreachable": "The provider could not be reached.",
    "unfunded": "The connection has insufficient funds.",
    "rate_limited": "The provider rate limit was reached.",
    "unknown": "The connection could not be verified.",
}
# Providers and relays may echo whole credentials, truncated keys or signed
# URLs in arbitrary errors. Only return exact, known credential-free messages.
_SAFE_MESSAGES = {
    "invalid api key",
    "unauthorized",
    "forbidden",
    "token expired",
    "expired token",
    "insufficient funds",
    "insufficient quota",
    "rate limit exceeded",
    "too many requests",
    "service unavailable",
}


class ProbeFailure(Exception):
    def __init__(self, reason, message=""):
        self.reason = reason
        clean = message.strip() if isinstance(message, str) else ""
        self.message = clean if clean.lower() in _SAFE_MESSAGES else _MESSAGES[reason]


def _provider_failure(status, error):
    if not isinstance(error, dict):
        error = {}
    message = error.get("message")
    codes = {str(error.get(key) or "").lower() for key in ("code", "type", "status")}
    details = error.get("details")
    if isinstance(details, list):
        codes.update(str(item.get("reason", "")).lower() for item in details if isinstance(item, dict))
    # Expiration is only distinguished when the provider identifies it.
    if codes & {"expiredtoken", "expiredtokenexception", "token_expired", "expired_token",
                "credential_expired", "api_key_expired"} or (
        status in {400, 401, 403} and isinstance(message, str)
        and "expired" in message.lower()
    ):
        reason = "expired"
    elif status == 402 or codes & {"insufficient_quota", "insufficient_deposits", "insufficient_funds"}:
        reason = "unfunded"
    elif status == 429 or codes & {"throttlingexception", "resource_exhausted"}:
        reason = "rate_limited"
    elif status in {401, 403} or codes & {"invalid_api_key", "api_key_invalid", "invalidclienttokenid",
                                         "unrecognizedclientexception", "accessdeniedexception"}:
        reason = "bad_credential"
    elif status >= 500 or status == 408:
        reason = "unreachable"
    else:
        reason = "unknown"
    return ProbeFailure(reason, message)


async def _document(client, method, url, headers, body=None):
    # Keep credentials in headers. Reject URL components that could leak into
    # HTTP client logging or change where a path suffix is sent.
    target = httpx.URL(url)
    if target.username or target.password or target.query or target.fragment:
        raise ProbeFailure("unknown")
    async with client.stream(method, target, headers=headers, json=body) as response:
        buffer = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=65536):
            buffer.extend(chunk)
            if len(buffer) > MAX_BODY:
                if response.is_error:
                    raise _provider_failure(response.status_code, None)
                raise ProbeFailure("unknown")
        try:
            data = json.loads(buffer)
        except (ValueError, UnicodeError):
            data = {}
        return response.status_code, data if isinstance(data, dict) else {}


def _check_document(status, data):
    if not 200 <= status < 300 or "error" in data:
        error = data.get("error", data)
        if isinstance(error, str):
            error = {"message": error}
        raise _provider_failure(status, error)


async def _http_probe(request, env, model):
    kind = request.get("api_kind") or "openai_compatible"
    base = request["base_url"].rstrip("/")
    request.update(served_model_id=model or "", messages=[{"role": "user", "content": "Hi"}], max_tokens=1)
    list_key = "data"
    if kind == "openai_compatible":
        prepared, error = _prepare_openai_call(request, env.get, {}, TIMEOUT)
        if error:
            raise ProbeFailure("bad_credential")
        generation_url, body, headers, _ = prepared
    elif kind in {"anthropic", "google"}:
        google = kind == "google"
        token, error = auth_token(request, env.get, "GEMINI_API_KEY" if google else "ANTHROPIC_API_KEY")
        if error:
            raise ProbeFailure("bad_credential")
        prepare = _google_request if google else _anthropic_request
        generation_url, body, headers = prepare(request, token, {})
        if google:
            list_key = "models"
    else:
        raise ProbeFailure("unknown")

    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False) as client:
        # This public catalog accepts invalid keys. Verify the credential
        # separately only on the canonical origin, never on a configured relay.
        if base == OPENROUTER_URL:
            status, data = await _document(client, "GET", base + "/key", headers)
            _check_document(status, data)
            if not isinstance(data.get("data"), dict):
                raise ProbeFailure("unknown")
        status, data = await _document(client, "GET", base + "/models", headers)
        if status in {404, 405, 501} and model:
            status, data = await _document(client, "POST", generation_url, headers, body)
            _check_document(status, data)
            output_key = {"anthropic": "content", "google": "candidates"}.get(kind, "choices")
            if not isinstance(data.get(output_key), list) or not data[output_key]:
                raise ProbeFailure("unknown")
            return None
        _check_document(status, data)
        models = data.get(list_key)
        if not isinstance(models, list):
            raise ProbeFailure("unknown")
        return len(models)


def _bedrock_probe(env):
    try:
        with aws_client(
            "bedrock", env["BEDROCK_REGION"], env.get,
            config=Config(connect_timeout=1, read_timeout=2, retries={"total_max_attempts": 1}),
        ) as client:
            data = client.list_foundation_models()
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        error = {k.lower(): v for k, v in exc.response.get("Error", {}).items()}
        raise _provider_failure(status, error) from None
    except BotoCoreError:
        raise ProbeFailure("unreachable") from None
    models = data.get("modelSummaries")
    if not isinstance(models, list):
        raise ProbeFailure("unknown")
    return len(models)


async def _buyer_probe(env):
    from sources.antseed import STALE_AFTER_S

    target = httpx.URL(env["ANTSEED_BYO_URL"])
    if target.query or target.username or target.password or target.fragment:
        raise ProbeFailure("unknown")
    try:
        snapshot = await buyer_snapshot(str(target), env["ANTSEED_BYO_TOKEN"])
    except httpx.HTTPStatusError as exc:
        raise _provider_failure(exc.response.status_code, None) from None
    status = snapshot.buyer_status("antseed") or {}
    age = time.time() - float(status.get("fetched_at", 0)) / 1000
    if not 0 <= age <= STALE_AFTER_S:
        raise ProbeFailure("unreachable")
    funds = float(status.get("deposits_available", 0))
    if not math.isfinite(funds):
        raise ProbeFailure("unknown")
    if funds <= 0:
        raise ProbeFailure("unfunded")
    return len({row["service"] for row in snapshot.peer_offers(900_000)
                if isinstance(row.get("service"), str) and row["service"]})


async def _probe(host, tenant_id, provider_id, scope):
    catalog = host.catalog()
    provider = catalog.get("providers", {}).get(provider_id)
    if provider is None:
        raise ProbeFailure("unknown")
    env, connections = await cp.tenant_connections(
        tenant_id, credential_names(catalog), **scope, require_fresh=True,
    )
    env, allowed, _ = host.tenant_provider_config(catalog, tenant_id, env, connections=connections, **scope)
    if provider_id not in allowed:
        raise ProbeFailure("bad_credential")
    if provider.get("api_kind") == "bedrock":
        return await asyncio.to_thread(_bedrock_probe, env)
    if provider_id == "antseed":
        return await _buyer_probe(env)
    pin_request = {"provider_id": provider_id}
    if provider_id == "openrouter_market" and provider.get("discovery_id") == "openrouter_market":
        # Tenant discovery pins these offers to the source origin.
        pin_request["offer"] = {"seller_endpoint": OPENROUTER_URL}
    pinned = host._pin_route(pin_request)
    model = next((row["provider_model_id"]
                  for _, entry in sorted(catalog.get("models", {}).items())
                  for row in entry.get("served_by", [])
                  if row.get("provider") == provider_id and row.get("provider_model_id")), None)
    return await _http_probe(pinned, env, model)


def install(app, host):
    slots = asyncio.Semaphore(2)

    @app.post(CONNECTION_CHECK_PATH)
    async def check(request: Request, tenant_id: str, provider_id: str):
        denied = _gate(request)
        if denied is not None:
            return denied
        scope = connection_check_scope(request, tenant_id, provider_id)
        if scope is None:
            return JSONResponse({"error": "invalid_connection_check"}, status_code=400)
        if slots.locked():
            return JSONResponse({"error": "connection_check_busy"}, status_code=503)
        async with slots:
            try:
                reserved = await asyncio.to_thread(host_store.reserve_connection_check, int(tenant_id), INTERVAL)
            except Exception:
                return JSONResponse({"error": "connection_check_store_unavailable"}, status_code=503)
            if not reserved:
                return JSONResponse({"error": "rate_limited"}, status_code=429,
                                    headers={"Retry-After": str(INTERVAL)})
            started = time.monotonic()
            count = None
            reason = "ok"
            message = _MESSAGES[reason]
            try:
                async with asyncio.timeout(TIMEOUT):
                    count = await _probe(host, int(tenant_id), provider_id, scope)
            except ProbeFailure as exc:
                reason, message = exc.reason, exc.message
            except (httpx.RequestError, TimeoutError, cp.RouteUnavailable):
                reason, message = "unreachable", _MESSAGES["unreachable"]
            except Exception:
                reason, message = "unknown", _MESSAGES["unknown"]
            return {"working": reason == "ok", "reason": reason, "message": message,
                    "model_count": count, "elapsed_ms": round((time.monotonic() - started) * 1000)}
