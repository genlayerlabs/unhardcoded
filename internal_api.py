"""Internal metering API for an external control plane.

Served by the ingress, gated by the same shared secret the control-plane
client sends outbound (`x-internal-secret`); see control_plane_client for the
overall contract. Hidden entirely (404) while the secret is unconfigured, so
the surface does not exist on operator-only deployments.

  GET /internal/usage?caller=<slug>&since_ts=<epoch>[&bucket=day]
  GET /internal/usage/recent?caller=<slug>&limit=<n<=500>

Rows come from the `calls` ledger, which only records LLM calls that reached
the router — ingress-level rejects (401/429) are not counted here.
"""
from __future__ import annotations

import time
import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import control_plane_client
import host_store

router = APIRouter()

_RECENT_LIMIT_MAX = 500


def _gate(request: Request) -> JSONResponse | None:
    if not control_plane_client.CONTROL_PLANE_INTERNAL_SECRET:
        return JSONResponse({"error": "not_found"}, status_code=404)
    if not control_plane_client.internal_secret_ok(request.headers):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return None


@router.get("/internal/usage")
async def internal_usage(request: Request, caller: str = "",
                         since_ts: int | None = None,
                         bucket: str | None = None,
                         project_id: int | None = None, environment_id: int | None = None) -> JSONResponse:
    denied = _gate(request)
    if denied is not None:
        return denied
    scope = {}
    if project_id is not None or environment_id is not None:
        if project_id is None or environment_id is None or min(project_id, environment_id) <= 0:
            return JSONResponse({"error": "invalid_scope"}, status_code=400)
        scope = {"project_id": project_id, "environment_id": environment_id}
    caller = caller.strip()
    if not caller:
        return JSONResponse({"error": "caller_required"}, status_code=400)
    try:
        totals = await asyncio.to_thread(host_store.usage_totals, since_ts=since_ts, caller=caller, strict=True, **scope)
        by_day = (await asyncio.to_thread(host_store.usage_aggregate, since_ts=since_ts, caller=caller, strict=True, **scope))["by_day"] if bucket == "day" else {}
    except Exception:
        return JSONResponse({"error": "ledger_unavailable"}, status_code=503)
    out: dict[str, Any] = {
        "caller": caller,
        **({"scope_version": 2, **scope} if scope else {}),
        "window": {"since_ts": since_ts, "until_ts": int(time.time())},
        "runs": totals["requests"],
        "errors": totals["errors"],
        "tokens_in": totals["tokens_in"],
        "tokens_out": totals["tokens_out"],
        "tokens_cached": totals["tokens_cached"],
        "tokens_total": totals["tokens_total"],
        "cost_usd": totals["cost_usd"],
    }
    if bucket == "day":
        out["buckets"] = [
            {"date": day, "runs": counter["requests"], "cost_usd": counter["cost_usd"]}
            for day, counter in sorted(by_day.items())
        ]
    return JSONResponse(out)


@router.get("/internal/usage/recent")
async def internal_usage_recent(request: Request, caller: str = "",
                                limit: int = 50, project_id: int | None = None,
                                environment_id: int | None = None) -> JSONResponse:
    denied = _gate(request)
    if denied is not None:
        return denied
    scope = {}
    if project_id is not None or environment_id is not None:
        if project_id is None or environment_id is None or min(project_id, environment_id) <= 0:
            return JSONResponse({"error": "invalid_scope"}, status_code=400)
        scope = {"project_id": project_id, "environment_id": environment_id}
    caller = caller.strip()
    if not caller:
        return JSONResponse({"error": "caller_required"}, status_code=400)
    limit = max(1, min(int(limit), _RECENT_LIMIT_MAX))
    calls = []
    try:
        rows = await asyncio.to_thread(host_store.recent_calls, limit=limit, caller=caller, strict=True, **scope)
    except Exception:
        return JSONResponse({"error": "ledger_unavailable"}, status_code=503)
    for row in rows:
        calls.append({
            "ts": row.get("ts"),
            "status": row.get("status"),
            "requested_model": row.get("requested_model"),
            "model_family": row.get("model_family"),
            "provider": row.get("provider_id"),
            "served_model_id": row.get("served_model_id"),
            "latency_ms": row.get("latency_ms"),
            "tokens_in": row.get("tokens_in"),
            "tokens_out": row.get("tokens_out"),
            "tokens_total": row.get("tokens_total"),
            "tokens_cached": row.get("tokens_cached"),
            "cost_usd": row.get("cost_usd"),
            "error_type": row.get("error_type"),
            "routing_summary": row.get("routing_summary"),
            "key_sha256_prefix": (row.get("consumer_sha") or "")[:12] or None,
        })
    return JSONResponse({"caller": caller, "calls": calls, **({"scope_version": 2, **scope} if scope else {})})
