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
                         bucket: str | None = None) -> JSONResponse:
    denied = _gate(request)
    if denied is not None:
        return denied
    caller = caller.strip()
    if not caller:
        return JSONResponse({"error": "caller_required"}, status_code=400)
    totals = host_store.usage_totals(since_ts=since_ts, caller=caller)
    out: dict[str, Any] = {
        "caller": caller,
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
        by_day = host_store.usage_aggregate(since_ts=since_ts, caller=caller)["by_day"]
        out["buckets"] = [
            {"date": day, "runs": counter["requests"], "cost_usd": counter["cost_usd"]}
            for day, counter in sorted(by_day.items())
        ]
    return JSONResponse(out)


@router.get("/internal/usage/recent")
async def internal_usage_recent(request: Request, caller: str = "",
                                limit: int = 50) -> JSONResponse:
    denied = _gate(request)
    if denied is not None:
        return denied
    caller = caller.strip()
    if not caller:
        return JSONResponse({"error": "caller_required"}, status_code=400)
    limit = max(1, min(int(limit), _RECENT_LIMIT_MAX))
    calls = []
    for row in host_store.recent_calls(limit=limit, caller=caller):
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
            "key_sha256_prefix": (row.get("consumer_sha") or "")[:12] or None,
        })
    return JSONResponse({"caller": caller, "calls": calls})
