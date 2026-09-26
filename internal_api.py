"""Internal metering API for an external control plane.

Served by the ingress, gated by the same shared secret the control-plane
client sends outbound (`x-internal-secret`); see control_plane_client for the
overall contract. Hidden entirely (404) while the secret is unconfigured, so
the surface does not exist on operator-only deployments.

  GET /internal/usage?caller=<slug>&since_ts=<epoch>[&until_ts=<epoch>]
      [&bucket=day][&group_by=key|model|route|day|hour]
  GET /internal/usage/recent?caller=<slug>&limit=<n<=500>
  GET /internal/usage/export?caller=<slug>&after_id=<id>&limit=<n<=5000>
  GET /internal/budgets?tenant_id=<id>&period=YYYY-MM&subject=<g:N|k:N>...

The usage endpoints accept project_id+environment_id (both, positive) and then
filter by the recorded routing scope BEFORE aggregating or paging. Windows are
half-open: since_ts <= ts < until_ts. Usage and export responses carry
`watermark_ts` (host_store.ledger_watermark): the ledger is complete below it.

Rows come from the `calls` ledger, which only records LLM calls that reached
the router — ingress-level rejects (401/402/429) are not counted here.
"""
from __future__ import annotations

import re
import time
import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import control_plane_client
import host_store

router = APIRouter()

_RECENT_LIMIT_MAX = 500
_EXPORT_LIMIT_MAX = 5000
_BUDGET_SUBJECTS_MAX = 500
_SUBJECT_RE = re.compile(r"[gk]:[1-9][0-9]{0,17}")
_PERIOD_RE = re.compile(r"[0-9]{4}-(0[1-9]|1[0-2])")
_MAX_TS = 253402300799   # 9999-12-31T23:59:59Z: beyond it the ledger's to_timestamp overflows


def _gate(request: Request) -> JSONResponse | None:
    if not control_plane_client.CONTROL_PLANE_INTERNAL_SECRET:
        return JSONResponse({"error": "not_found"}, status_code=404)
    if not control_plane_client.internal_secret_ok(request.headers):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return None


def _scope(project_id: int | None, environment_id: int | None) -> dict | None:
    """{} unscoped, the scope when both are positive, None when invalid."""
    if project_id is None and environment_id is None:
        return {}
    if project_id is None or environment_id is None or min(project_id, environment_id) <= 0:
        return None
    return {"project_id": project_id, "environment_id": environment_id}


@router.get("/internal/usage")
async def internal_usage(request: Request, caller: str = "",
                         since_ts: int | None = None, until_ts: int | None = None,
                         bucket: str | None = None, group_by: str | None = None,
                         project_id: int | None = None, environment_id: int | None = None) -> JSONResponse:
    denied = _gate(request)
    if denied is not None:
        return denied
    scope = _scope(project_id, environment_id)
    if scope is None:
        return JSONResponse({"error": "invalid_scope"}, status_code=400)
    caller = caller.strip()
    if not caller:
        return JSONResponse({"error": "caller_required"}, status_code=400)
    if group_by is not None and group_by not in host_store._USAGE_GROUPS:
        return JSONResponse({"error": "invalid_group_by"}, status_code=400)
    if any(ts is not None and not 0 <= ts <= _MAX_TS for ts in (since_ts, until_ts)) or (
            since_ts is not None and until_ts is not None and until_ts < since_ts):
        return JSONResponse({"error": "invalid_window"}, status_code=400)
    window = {"since_ts": since_ts, "until_ts": until_ts}
    try:
        totals = await asyncio.to_thread(host_store.usage_totals, caller=caller, strict=True, **window, **scope)
        by_day = (await asyncio.to_thread(host_store.usage_aggregate, caller=caller, strict=True, **window, **scope))["by_day"] if bucket == "day" else {}
        groups = (await asyncio.to_thread(host_store.usage_groups, group_by, caller=caller, **window, **scope)
                  if group_by else None)
    except Exception:
        return JSONResponse({"error": "ledger_unavailable"}, status_code=503)
    out: dict[str, Any] = {
        "caller": caller,
        **({"scope_version": 2, **scope} if scope else {}),
        "window": {"since_ts": since_ts, "until_ts": until_ts if until_ts is not None else int(time.time())},
        # Calls with ts < watermark_ts are all in the ledger: a window whose
        # until_ts <= watermark_ts is closed and will not change.
        "watermark_ts": host_store.ledger_watermark(),
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
    if groups is not None:
        out["group_by"] = group_by
        out["groups"] = groups
    return JSONResponse(out)


@router.get("/internal/usage/recent")
async def internal_usage_recent(request: Request, caller: str = "",
                                limit: int = 50, project_id: int | None = None,
                                environment_id: int | None = None) -> JSONResponse:
    denied = _gate(request)
    if denied is not None:
        return denied
    scope = _scope(project_id, environment_id)
    if scope is None:
        return JSONResponse({"error": "invalid_scope"}, status_code=400)
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


@router.get("/internal/usage/export")
async def internal_usage_export(request: Request, caller: str = "", after_id: int = 0,
                                limit: int = 1000, project_id: int | None = None,
                                environment_id: int | None = None) -> JSONResponse:
    """Raw ledger rows by id cursor — the reconciliation source of truth. Page
    until `rows` is empty; periods ending before `watermark_ts` are complete."""
    denied = _gate(request)
    if denied is not None:
        return denied
    scope = _scope(project_id, environment_id)
    if scope is None:
        return JSONResponse({"error": "invalid_scope"}, status_code=400)
    caller = caller.strip()
    if not caller:
        return JSONResponse({"error": "caller_required"}, status_code=400)
    if after_id < 0:
        return JSONResponse({"error": "invalid_cursor"}, status_code=400)
    limit = max(1, min(int(limit), _EXPORT_LIMIT_MAX))
    try:
        page = await asyncio.to_thread(host_store.usage_export, caller, after_id, limit, **scope)
    except Exception:
        return JSONResponse({"error": "ledger_unavailable"}, status_code=503)
    return JSONResponse({"caller": caller, **({"scope_version": 2, **scope} if scope else {}), **page})


@router.get("/internal/budgets")
async def internal_budgets(request: Request, tenant_id: int | None = None,
                           period: str = "") -> JSONResponse:
    """Spent/reserved per key subject for one tenant and UTC month. Subjects
    are namespaced by tenant, so another tenant's identical subject never
    leaks into this answer."""
    denied = _gate(request)
    if denied is not None:
        return denied
    subjects = request.query_params.getlist("subject")
    if tenant_id is None or tenant_id <= 0:
        return JSONResponse({"error": "tenant_required"}, status_code=400)
    if not _PERIOD_RE.fullmatch(period):
        return JSONResponse({"error": "invalid_period"}, status_code=400)
    if (not subjects or len(subjects) > _BUDGET_SUBJECTS_MAX
            or any(not _SUBJECT_RE.fullmatch(s) for s in subjects)):
        return JSONResponse({"error": "invalid_subject"}, status_code=400)
    try:
        rows = await asyncio.to_thread(host_store.subject_budgets, tenant_id, period, subjects)
    except Exception:
        return JSONResponse({"error": "budget_store_unavailable"}, status_code=503)
    return JSONResponse({"tenant_id": tenant_id, "period": period, "subjects": rows})
