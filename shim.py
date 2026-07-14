"""
shim.py — OpenAI-compatible HTTP façade in front of router.lua.

Any client that speaks /v1/chat/completions can POST OpenAI-shaped requests.
The shim translates them to a router contract, runs `router.execute`, and
translates the router result back to an OpenAI response. Provider selection,
fallback, retries and provider auth all live on the router side; the client
sees a single endpoint.

Model field convention (explicit prefixes, no magic):

    model = ""                          -> default_profile
    model = "profile:cheap_explore"     -> contract.profile
    model = "family:deepseek-v3"        -> contract.requirements.model_family
    model = "pin:<provider>/<family>"   -> contract.requirements.pin
    model = anything else               -> default_profile (logged)

Streaming is supported. With a `streaming_call` dispatcher, `stream: true`
streams token-by-token (with fallback before the first byte); without one — and
for flows, which have no token stream — the finished result is pseudo-streamed
as SSE. Either way the client gets a valid `text/event-stream`.

Concurrency note: lupa serializes Lua execution. FastAPI's threadpool will
queue concurrent /v1/chat/completions calls behind the single LuaRuntime.
Fine for one or a handful of concurrent clients; for hundreds-to-thousands
of concurrent callers, use a luerl-based host instead.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

import host_store

_log = logging.getLogger("unhardcoded.shim")


# Profile name used when nothing else can be inferred. Replaced via
# create_app(default_profile=...).
DEFAULT_PROFILE_FALLBACK = "default"

# max_tokens supplied when the client omits it. max_tokens is *optional* in
# the OpenAI spec, but some upstream providers / policy candidates reject
# requests that leave it out (the symptom is no_candidates/exhausted), so the
# shim fills in a sane ceiling instead of forwarding nothing. Override via
# create_app(default_max_tokens=...); pass None to keep the strict
# omit-when-absent behaviour.
DEFAULT_MAX_TOKENS_FALLBACK = 4096


class ChatRequest(BaseModel):
    """Permissive OpenAI /v1/chat/completions body.

    Unknown fields are kept (`extra="allow"`) so future OpenAI fields don't
    require shim edits; the shim only forwards the fields the router knows.
    """
    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: list[dict] = []
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: Any = None
    response_format: dict | None = None
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    # Optional upstream liveness guard. For OpenAI-compatible streaming-capable
    # routes, fail/fallback if no content/tool delta arrives before this budget.
    first_token_timeout_ms: int | None = None
    # Σ_pol per-call policy: a TERM (plain JSON array, e.g.
    # ["policy", ["ev_zero"], ["meets_req"], ...]). Data, never code: the
    # core admits it (sorts/arity/depth/node bounds) and ∧-composes the
    # host's config.policy_envelope so callers narrow, never widen.
    # Admission failure -> 400 invalid_policy.
    policy_ir: list | None = None
    # Σ_flow per-call composition: a flow TERM (plain JSON, { "flow", nodes })
    # where each node carries its own policy_ir + system prompt. Data, never
    # code: the core admits the whole DAG (graph validity + every node's
    # policy). When present it takes precedence over policy_ir/model. Admission
    # failure -> 400 invalid_flow.
    flow_ir: list | None = None
    # Conversation/session id (optional). When present the host learns which peer
    # served this session (route_cache) and, next turn, marks that peer's offer
    # cache_hot so a cache-aware policy keeps the prompt-cache-hot peer sticky.
    # Pure host state — never enters the algebra's signature; clients without a
    # session simply get no affinity. Additive to OpenAI-compat.
    session: str | None = None
    # The authed consumer key behind this request, set by the ingress proxy via
    # the x-llm-router-caller header (consumers cannot set it — the proxy strips
    # and re-injects it). Never sent to the core; used only to bind sid->owner in
    # the session meter so the consumer-facing session view stays per-consumer.
    caller: str | None = None


class ResponsesRequest(BaseModel):
    """Permissive OpenAI /v1/responses body. Unknown fields are kept
    (extra="allow") so Responses params the shim does not read (reasoning,
    include, store, parallel_tool_calls, prompt_cache_key, text,
    previous_response_id, …) never break the request."""
    model_config = ConfigDict(extra="allow")

    model: str = ""
    input: Any = None             # str | list[item]
    instructions: str | None = None
    tools: list[dict] | None = None
    tool_choice: Any = None
    stream: bool = False
    max_output_tokens: int | None = None
    temperature: float | None = None
    first_token_timeout_ms: int | None = None
    policy_ir: list | None = None
    session: str | None = None
    caller: str | None = None


class PolicyRankRequest(BaseModel):
    """Body of POST /x/rank — dry-run a per-call policy term."""
    policy_ir: list
    context: int = 32000
    requirements: dict | None = None


class CompactRequest(BaseModel):
    """Body of POST /v1/compact — append-only context sealing.

    A STATELESS transform: the caller sends the whole message array, the host
    seals the aged middle into one summary (routed cheaply by `policy_ir`) and
    returns the spliced array. The host holds NO conversation state — the agent
    stays sovereign over its context (the cache_hot peer also stays hot because
    the frozen prefix is never rewritten)."""
    messages: list[dict]
    keep_recent: int = 6          # verbatim tail kept after the seal
    policy_ir: list | None = None  # cheap routing for the summarizer
    max_tokens: int | None = 512


class FlowNormalizeRequest(BaseModel):
    """Body of POST /x/flow/normalize — admit + identify a Σ_flow term."""
    flow_ir: list


class PolicyBuildRequest(BaseModel):
    """Body of POST /x/policy/build — the declarative surface the dashboard
    builder collects; lowered by the core's elaborate, never host-side."""
    weights: dict | None = None
    filter: list | dict | str | None = None
    selector: str | None = None
    selector_opts: dict | None = None
    mutate: list | dict | str | None = None
    retry_table: dict | None = None


def _rank_rows(ranked: list) -> list[dict]:
    rows = []
    for r in ranked:
        c = r.get("candidate") or {}
        rows.append({
            "provider": c.get("provider_id"),
            "model_family": c.get("model_family"),
            "served_model_id": c.get("served_model_id"),
            "tier": c.get("tier"),
            "discovery": c.get("discovery"),
            "price_in": c.get("price_in"),
            "price_out": c.get("price_out"),
            "quality": c.get("quality_hint"),
            "score": r.get("score"),
        })
    return rows


# ---- context compaction (POST /v1/compact) --------------------------------

_SEAL_SYSTEM = (
    "Summarize the conversation below, preserving decisions, open threads, file "
    "paths, and key identifiers. Be dense and concrete.")
_SEAL_PREFIX = "[Earlier conversation, sealed summary]\n"

# Cheapest healthy route — a seal is auxiliary work, never the main model.
_DEFAULT_COMPACT_POLICY = [
    "policy",
    ["and", ["meets_req"], ["not", ["is", "disabled"]]],
    ["neg", ["normalize", ["field", "price_in"]]],
    ["argmax"], ["id"], ["always", {"action": "next_candidate"}],
]


def _render_messages(msgs: list[dict]) -> str:
    out = []
    for m in msgs:
        content = m.get("content")
        if isinstance(content, list):  # multimodal parts -> text only
            content = " ".join(p.get("text", "") for p in content
                               if isinstance(p, dict))
        out.append(f"{m.get('role', '?')}: {content}")
    return "\n".join(out)


def _compact_suggested(resp: dict) -> bool:
    """Context-pressure hint: True once a call's INPUT crosses the operator
    threshold (settings `compaction.at_tokens`). Surfaced on x_router so an agent
    knows to POST /v1/compact — it owns the threshold decision, not the agent.
    Measured on the real prompt_tokens the call reported, so it costs nothing."""
    import settings as _settings
    n = resp.get("tokens_in")
    if not n:
        return False
    try:
        return int(n) >= int(_settings.get("compaction.at_tokens"))
    except (TypeError, ValueError):
        return False


def create_app(host, default_profile: str = DEFAULT_PROFILE_FALLBACK,
               streaming_call=None,
               default_max_tokens: int | None = DEFAULT_MAX_TOKENS_FALLBACK,
               codex_store=None,
               ) -> FastAPI:
    """Build a FastAPI app wired to a pre-initialized LLMRouterHost.

    The host must already have `init()` called and `host.call_provider`
    pointing at something that actually talks to providers (or a mock for
    tests).

    `streaming_call(request, emit)` is the streaming api_kind dispatcher
    (streaming.make_streaming_dispatcher). Without it, `stream: true`
    requests still work via the pseudo-stream path (complete result encoded
    as SSE) — which is also what mocked backends produce.
    """
    import asyncio

    import streaming as _streaming

    app = FastAPI(title="llm-router shim", docs_url=None, redoc_url=None)

    # subscription backends (codex) are billed $0 per request — their ranking
    # price is a scarcity shadow price, not a cost
    subscription_providers = frozenset(
        pid for pid, p in ((host.catalog() or {}).get("providers") or {}).items()
        if isinstance(p, dict) and p.get("api_kind") == "openai_codex")

    @app.post("/x/codex/reload")
    def reload_codex_accounts():
        """Re-scan the Codex accounts dir on the PVC so accounts added/removed
        from the dashboard go live without a router restart. Internal — /x/* is
        hidden from consumers."""
        if codex_store is None:
            return JSONResponse(status_code=404, content={"error": {
                "message": "codex multi-account store not configured",
                "type": "not_found", "code": "codex_store_absent"}})
        names = codex_store.reload()
        return {"ok": True, "accounts": names}

    @app.post("/x/config/reload")
    def reload_config():
        """Re-read operator config overrides (dashboard Config tab) so tunable
        knobs (antseed top-N, codex scarcity ramp, runway thresholds, price
        multipliers) apply without a router restart. Sources read settings.get
        live; marketplace discovery is invalidated so source-level filters that
        affect offers refresh immediately instead of waiting for the discovery
        TTL. Price multipliers are applied live at candidate assembly time.
        Internal — /x/* is hidden from consumers."""
        import settings as _settings
        overrides = _settings.reload()
        for provider in (host.catalog().get("providers") or {}).values():
            if isinstance(provider, dict) and provider.get("discovery") == "marketplace":
                did = provider.get("discovery_id")
                if did:
                    host.invalidate_discovery(did)
        return {"ok": True, "overrides": overrides}

    @app.get("/x/session/{sid}")
    def session_meter(sid: str, request: Request):
        """Accumulated usage for a session: calls, tokens_in/out, tokens_cached,
        cost_usd — the running total the per-call x_router.session_acc reflects —
        plus `warm`: the routes (family/provider/served_by) currently holding the
        session's prompt-cache prefix. Internal (/x/* hidden from consumers).

        Cross-consumer isolation: a session's economics + warm peers belong to the
        consumer that first wrote the sid (bound in observe(owner=...)). When the
        ingress proxy forwards a consumer's authed key as x-llm-router-caller (the
        consumer-facing /v1/session/{sid} view), only that owner may read it —
        anyone else gets 404 (NOT 403: confirming the sid exists would itself leak
        that consumer A holds it). Operator callers (dashboard /x/*, no caller
        header) are unscoped, as before. The in-process meter resets on restart,
        so an unknown owner also means there is simply nothing to show — 404 is
        consistent either way."""
        caller = request.headers.get("x-llm-router-caller")
        if caller:
            owner = host_store.session_owner(sid)
            if owner is None or owner != caller:
                return JSONResponse(status_code=404, content={"error": {
                    "message": "session not found", "type": "not_found",
                    "code": "session_not_found"}})
        acc = host_store.session_totals(sid)
        return {**acc, "warm": host_store.session_warm(sid)}

    @app.get("/x/sessions")
    def session_meters():
        """All session meters (operator view of per-session spend/cache)."""
        return {"sessions": host_store.all_session_totals()}

    @app.get("/x/calls")
    def recent_calls(limit: int = 100):
        """Recent rows from the host-store call ledger (operator view /
        verification of the emerging source of truth). Read-only."""
        import host_store
        return {"calls": host_store.recent_calls(min(max(int(limit), 1), 1000)),
                "total": host_store.count()}

    # ---- AntSeed buyer hot-wallet control (dashboard self-service) -----------
    # Proxy deposit/withdraw/refresh to the sidecar control server, then refresh
    # SOURCE_STATE so /x/market reflects the new escrow at once. Internal — /x/*
    # is hidden from consumers.
    import os as _os
    import re as _re

    _AMOUNT_RE = _re.compile(r"^\d+(\.\d{1,6})?$")

    def _wallet_control():
        url = (_os.getenv("ANTSEED_CONTROL_URL") or "").rstrip("/")
        token = _os.getenv("ANTSEED_CONTROL_TOKEN") or ""
        return (url, token) if (url and token) else (None, None)

    def _antseed_wallet_view():
        import sources as _sources
        for pid, bal in ((_sources.SOURCE_STATE.get("antseed") or {})
                         .get("balances") or {}).items():
            det = bal.get("detail") or {}
            return {"provider": pid, "address": det.get("wallet"),
                    "deposits_available": bal.get("value"),
                    "deposits_reserved": det.get("reserved"),
                    "wallet_usdc": det.get("wallet_usdc"),
                    "wallet_eth": det.get("wallet_eth"),
                    "connection": det.get("connection"),
                    "fetched_at": bal.get("fetched_at")}
        return None

    async def _refresh_antseed_wallet():
        import sources as _sources
        from sources.antseed import AntSeedSource
        try:
            await _sources.refresh_once(host, host.catalog(), AntSeedSource(host.catalog()))
        except Exception:  # noqa: BLE001 — refresh is best-effort
            pass
        return _antseed_wallet_view()

    async def _wallet_mutate(op: str, body: dict):
        url, token = _wallet_control()
        if not url:
            return JSONResponse(status_code=503, content={"error": {
                "message": "antseed wallet control not configured",
                "type": "wallet_error", "code": "wallet_control_unavailable"}})
        amount = str((body or {}).get("amount", "")).strip()
        if not _AMOUNT_RE.match(amount) or float(amount) <= 0:
            return JSONResponse(status_code=400, content={"error": {
                "message": "amount must be a positive USDC value (<=6 decimals)",
                "type": "invalid_request", "code": "wallet_amount"}})
        import httpx
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{url}/{op}", json={"amount": amount},
                                 headers={"x-antseed-control-token": token}, timeout=130.0)
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"error": {
                "message": f"wallet control unreachable: {e}",
                "type": "wallet_error", "code": "wallet_control_unreachable"}})
        if r.status_code != 200:
            try:
                detail = (r.json() or {}).get("error")
            except Exception:  # noqa: BLE001
                detail = (r.text or "")[:300]
            return JSONResponse(status_code=502, content={"error": {
                "message": str(detail), "type": "wallet_error", "code": "wallet_op_failed"}})
        return {"ok": True, "action": op, "amount": amount,
                "wallet": await _refresh_antseed_wallet()}

    @app.get("/x/wallet")
    async def wallet_view():
        # Read-only balance snapshot for the dashboard wallet panel (no on-chain
        # tx). Falls back to a live refresh if the source cache is empty.
        w = _antseed_wallet_view()
        if w is None:
            w = await _refresh_antseed_wallet()
        return {"ok": True, "wallet": w}

    @app.post("/x/wallet/deposit")
    async def wallet_deposit(body: dict):
        return await _wallet_mutate("deposit", body)

    @app.post("/x/wallet/withdraw")
    async def wallet_withdraw(body: dict):
        return await _wallet_mutate("withdraw", body)

    @app.post("/x/wallet/refresh")
    async def wallet_refresh():
        url, token = _wallet_control()
        if url:
            import httpx
            try:
                async with httpx.AsyncClient() as c:
                    await c.post(f"{url}/status",
                                 headers={"x-antseed-control-token": token}, timeout=35.0)
            except httpx.HTTPError:
                pass
        return {"ok": True, "wallet": await _refresh_antseed_wallet()}

    async def _wallet_reclaim(op: str, timeout: float, with_wallet: bool):
        # Channel reclaim: recover USDC reserved in idle payment channels.
        #   scan          read-only enumeration of on-chain reclaimable funds
        #   request-close start the ~15-min on-chain challenge (one tx/channel)
        #   withdraw      pull funds from channels whose window has elapsed
        url, token = _wallet_control()
        if not url:
            return JSONResponse(status_code=503, content={"error": {
                "message": "antseed wallet control not configured",
                "type": "wallet_error", "code": "wallet_control_unavailable"}})
        import httpx
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{url}/reclaim/{op}",
                                 headers={"x-antseed-control-token": token}, timeout=timeout)
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"error": {
                "message": f"wallet control unreachable: {e}",
                "type": "wallet_error", "code": "wallet_control_unreachable"}})
        try:
            payload = r.json() or {}
        except Exception:  # noqa: BLE001
            payload = {}
        if r.status_code != 200:
            detail = payload.get("error") or (r.text or "")[:300]
            return JSONResponse(status_code=502, content={"error": {
                "message": str(detail), "type": "wallet_error", "code": "reclaim_failed"}})
        if with_wallet:
            payload["wallet"] = await _refresh_antseed_wallet()
        return payload

    @app.post("/x/wallet/reclaim/scan")
    async def wallet_reclaim_scan():
        return await _wallet_reclaim("scan", 95.0, with_wallet=False)

    @app.post("/x/wallet/reclaim/set-operator")
    async def wallet_reclaim_set_operator():
        # One-time: assign the buyer wallet as its own deposits operator so
        # requestClose/withdraw stop reverting NotAuthorized(). Moves no funds.
        return await _wallet_reclaim("set-operator", 250.0, with_wallet=False)

    @app.post("/x/wallet/reclaim/request-close")
    async def wallet_reclaim_request_close():
        return await _wallet_reclaim("request-close", 250.0, with_wallet=False)

    @app.post("/x/wallet/reclaim/withdraw")
    async def wallet_reclaim_withdraw():
        return await _wallet_reclaim("withdraw", 250.0, with_wallet=True)

    @app.get("/healthz")
    def healthz():
        info = host.info()
        return {"ok": True, "initialized": info.get("initialized", False)}

    @app.get("/v1/models")
    def list_models():
        import sources as _sources
        info = host.info()
        ids = [f"profile:{p}" for p in (info.get("profile_names") or [])]
        seen = set(info.get("models_loaded") or [])
        ids += [f"family:{f}" for f in seen]
        # discovered marketplace families (e.g. live OpenRouter models) are
        # routable too, so list them alongside the curated families.
        for sstate in _sources.SOURCE_STATE.values():
            for r in (sstate.get("book") or {}).get("rows") or []:
                fam = r.get("model_family")
                if fam and fam not in seen:
                    seen.add(fam)
                    ids.append(f"family:{fam}")
        return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}

    @app.get("/x/runtime")
    def runtime_state():
        """Live router runtime for the operator dashboard: circuit breakers,
        disabled providers, EMA metrics (incl. live prices), source freshness
        and balances. Internal — the ingress proxy hides /x/* from consumer
        callers and fetches this server-side."""
        import sources as _sources
        state = host.dump_state() or {}
        balances: dict = {}
        sources_view: dict = {}
        for name, s in _sources.SOURCE_STATE.items():
            balances.update(s.get("balances") or {})
            sources_view[name] = {k: v for k, v in s.items()
                                  if k not in ("balances", "book")}
        return {
            "ts": int(time.time()),
            "circuit_breakers": state.get("circuit_breakers") or {},
            "disabled_providers": state.get("disabled_providers") or {},
            "ema_metrics": state.get("ema_metrics") or {},
            "balances": balances,
            "sources": sources_view,
        }

    @app.get("/x/market")
    def market_view():
        """Full price book per curated family: every seller each source knows
        about (marketplace sellers trimmed to the best few per family by the
        source), with live performance from EMA metrics where the router has
        actually called that provider|family. Internal — the dashboard
        fetches this server-side; /x/* is hidden from consumers."""
        import sources as _sources
        import host_store
        catalog = host.catalog() or {}
        models = catalog.get("models") or {}
        state = host.dump_state() or {}
        ema = state.get("ema_metrics") or {}   # still carries seeded price + credits
        disabled = state.get("disabled_providers") or {}
        marketplace_pids = {
            pid for pid, p in (catalog.get("providers") or {}).items()
            if isinstance(p, dict) and p.get("discovery") == "marketplace"}

        # Live perf is host-owned now (#15): the engine folds no EMA, so build it
        # from the host's per-route measurements — DERIVED on the fly from
        # route_observations (#4a), aggregated across the peers/route ids that
        # serve a given provider|family. None until the router has called it.
        _stats = host_store.route_stats()  # {route_key: {success_rate, latency_ms, count}}

        def _perf(provider, family):
            prefix = f"{provider}|{family}|"
            rows = [v for k, v in _stats.items() if k.startswith(prefix)]
            total = sum(r["count"] for r in rows)
            if not total:
                return None
            sr_rows = [r for r in rows if r.get("success_rate") is not None]
            lt_rows = [r for r in rows if r.get("latency_ms") is not None]
            sr_calls = sum(r["count"] for r in sr_rows)
            lt_calls = sum(r["count"] for r in lt_rows)
            sr = sum(r["success_rate"] * r["count"] for r in sr_rows)
            lt = sum(r["latency_ms"] * r["count"] for r in lt_rows)
            return {"success_rate": (sr / sr_calls) if sr_calls else None,
                    "latency_ms": round(lt / lt_calls) if lt_calls else None,
                    "calls": total}

        def _antseed_row(r, family, book):
            via = (r.get("tradable_via") or [None])[0]
            return {
                "source": "antseed",
                "seller": f"peer {str(r.get('seller') or '')[:8]}",
                "wire_model_id": r.get("wire_model_id"),
                "price_in": r.get("price_in"),
                "price_out": r.get("price_out"),
                "price_refreshed_at": book.get("fetched_at"),
                "last_seen": r.get("last_seen"),
                "pinned": bool(r.get("pinned_by")),
                "tradable": bool(via),
                "via": via,
                "perf": _perf(via, family) if via else None,
            }

        # Registered model-level traits (OpenRouter benchmarks/modalities/caps),
        # per curated family — same source the policy/builder gate on.
        meta = host.model_meta() or {}
        book = (_sources.SOURCE_STATE.get("antseed") or {}).get("book") or {}
        book_rows: dict[str, list] = {}
        for r in book.get("rows") or []:
            book_rows.setdefault(r.get("model_family"), []).append(r)
        # without a book (source down / first boot), fall back to the
        # marketplace EMA rows so pinned offers stay visible
        have_book = bool(book.get("rows"))

        families = []
        for family, model in models.items():
            rows = []
            for key, m in ema.items():
                provider, _, fam = key.partition("|")
                if fam != family:
                    continue
                if have_book and provider in marketplace_pids:
                    continue
                if m.get("price_in") is None and m.get("price_out") is None:
                    continue
                rows.append({
                    "source": provider,
                    "seller": provider,
                    "wire_model_id": None,
                    "price_in": m.get("price_in"),
                    "price_out": m.get("price_out"),
                    "price_refreshed_at": m.get("price_refreshed_at"),
                    "pinned": None,
                    "tradable": provider not in disabled,
                    "via": provider,
                    "perf": _perf(provider, family),
                })
            for r in book_rows.get(family) or []:
                rows.append(_antseed_row(r, family, book))
            rows.sort(key=lambda r: (r.get("price_in") is None,
                                     r.get("price_in") or 0,
                                     r.get("price_out") or 0))
            direct = len([r for r in rows if r["source"] != "antseed"])
            market_total = ((book.get("families") or {}).get(family) or {}).get(
                "sellers_total", len([r for r in rows if r["source"] == "antseed"]))
            families.append({
                "family": family,
                "quality": model.get("static_quality_hint"),
                "sellers_total": direct + market_total,
                "meta": meta.get(family) or {},
                "rows": rows,
            })
        # uncurated marketplace services: routable but absent from the curated
        # catalog (no benchmark). Surface them too so the dashboard shows the
        # WHOLE market, flagged uncurated and sorted after the curated families.
        for family, brows in book_rows.items():
            if family in models:
                continue
            rows = sorted((_antseed_row(r, family, book) for r in brows),
                          key=lambda r: (r.get("price_in") is None,
                                         r.get("price_in") or 0,
                                         r.get("price_out") or 0))
            market_total = ((book.get("families") or {}).get(family) or {}).get(
                "sellers_total", len(rows))
            families.append({
                "family": family,
                "quality": None,
                "uncurated": True,
                "sellers_total": market_total,
                "meta": {},
                "rows": rows,
            })
        # Other marketplace sources that expose a book (e.g. live OpenRouter
        # discovery). Their rows are already source-tagged; surface the uncurated
        # families the same way antseed's are, so the Catalog shows the whole
        # OpenRouter catalog without hand curation.
        for sname, sstate in _sources.SOURCE_STATE.items():
            if sname == "antseed":
                continue
            sbook = sstate.get("book") or {}
            srows: dict[str, list] = {}
            for r in sbook.get("rows") or []:
                srows.setdefault(r.get("model_family"), []).append(r)
            for family, brows in srows.items():
                if family in models:
                    continue  # curated families already shown above
                rows = sorted(({
                    "source": r.get("source") or sname,
                    "seller": r.get("seller") or sname,
                    "wire_model_id": r.get("wire_model_id"),
                    "price_in": r.get("price_in"),
                    "price_out": r.get("price_out"),
                    "price_refreshed_at": sbook.get("fetched_at"),
                    "pinned": None,
                    "tradable": bool(r.get("tradable", True)),
                    "via": r.get("via"),
                    "perf": _perf(r.get("via"), family) if r.get("via") else None,
                } for r in brows), key=lambda r: (r.get("price_in") is None,
                                                  r.get("price_in") or 0,
                                                  r.get("price_out") or 0))
                fam_info = (sbook.get("families") or {}).get(family) or {}
                market_total = fam_info.get("sellers_total", len(rows))
                families.append({
                    # discovered, but first-class: full live benchmarks/modalities
                    # in `meta`, same shape the curated families expose.
                    "family": family, "quality": None, "discovered": True,
                    "sellers_total": market_total, "meta": fam_info.get("meta") or {},
                    "rows": rows,
                })
        def _family_sort_key(f):
            # curated first, then discovered/uncurated; within a group sort by
            # quality hint or, for discovered families, their live benchmark.
            grp = 1 if (f.get("uncurated") or f.get("discovered")) else 0
            q = f["quality"] if f["quality"] is not None \
                else (f.get("meta") or {}).get("bench_intelligence")
            return (grp, -(q if q is not None else -1), f["family"])
        families.sort(key=_family_sort_key)
        # AntSeed buyer hot-wallet: address (where to top up), deposits and
        # connection, read live from the source's balances() — so the address
        # always reflects the CURRENT identity (if the data volume regenerates
        # it, the new address shows here automatically). One buyer -> one wallet.
        wallet = None
        for pid, bal in ((_sources.SOURCE_STATE.get("antseed") or {})
                         .get("balances") or {}).items():
            det = bal.get("detail") or {}
            wallet = {
                "provider": pid,
                "address": det.get("wallet"),
                "deposits_available": bal.get("value"),
                "deposits_reserved": det.get("reserved"),
                "wallet_usdc": det.get("wallet_usdc"),
                "wallet_eth": det.get("wallet_eth"),
                "connection": det.get("connection"),
                "fetched_at": bal.get("fetched_at"),
            }
            break
        return {"families": families, "wallet": wallet, "ts": int(time.time())}

    @app.post("/x/providers")
    def add_provider(body: dict):
        """Hot-add an operator-defined provider (openai_compatible + env key
        only). Validates against the live catalog, injects the key into the
        process env, merges the provider into the Lua config and re-inits the
        core with breakers/EMA state preserved. Persistence is the ingress's
        job (the provider_overlays store + .env.secrets); this endpoint only
        makes it live. Internal — /x/* is hidden from consumers."""
        import os

        from provider_overlay import apply_to_host, validate_entry
        pid = str(body.get("id") or "").strip()
        entry = {
            "base_url": body.get("base_url"),
            "api_kind": body.get("api_kind") or "openai_compatible",
            "tier": body.get("tier") or "partner",
            "auth_env": body.get("auth_env"),
            "served_models": body.get("served_models") or [],
        }
        errors = validate_entry(pid, entry, host.catalog())
        if any("already exists" in e for e in errors):
            return JSONResponse(status_code=409, content={"error": {
                "message": f"provider {pid!r} already exists",
                "type": "conflict", "code": "provider_exists"}})
        if errors:
            return JSONResponse(status_code=400, content={"error": {
                "message": "; ".join(errors), "type": "invalid_request",
                "code": "provider_invalid"}})
        key = body.get("key")
        if key:
            os.environ[str(entry["auth_env"])] = str(key)
            host.set_env(str(entry["auth_env"]), str(key))
        snapshot = host.dump_state()
        applied = apply_to_host(host, {"providers": {pid: entry}})
        host.init()
        host.restore_state(snapshot)
        return {"ok": True, "provider": pid, "applied": applied,
                "key_installed": bool(key)}

    @app.post("/x/provider-key")
    def set_provider_key(body: dict):
        """Update the API key of an EXISTING provider — no catalog change, so
        no 'already exists' rejection and no re-init. Injects the key into the
        process env and host._env; the core reads host._env live, so the next
        call uses the new key. Persistence (.env.secrets) is the ingress's job.
        Internal — /x/* is hidden from consumers."""
        import os

        pid = str(body.get("provider") or body.get("id") or "").strip()
        key = body.get("key")
        if not key:
            return JSONResponse(status_code=400, content={"error": {
                "message": "key is required", "type": "invalid_request",
                "code": "provider_key"}})
        provider = (host.catalog().get("providers") or {}).get(pid)
        if not isinstance(provider, dict):
            return JSONResponse(status_code=404, content={"error": {
                "message": f"provider {pid!r} not found",
                "type": "not_found", "code": "provider_not_found"}})
        auth_env = str(provider.get("auth_env") or "").strip()
        if not auth_env:
            return JSONResponse(status_code=400, content={"error": {
                "message": f"provider {pid!r} has no auth_env (e.g. oauth/codex); "
                           "key update not applicable",
                "type": "invalid_request", "code": "provider_no_auth_env"}})
        os.environ[auth_env] = str(key)
        host.set_env(auth_env, str(key))
        return {"ok": True, "provider": pid, "auth_env": auth_env,
                "key_installed": True}

    @app.get("/x/rank")
    def rank_profile(profile: str, context: int = 32000):
        """Live ranking for one profile — live prices, breakers, marketplace
        offers included. Internal: the dashboard renders THIS instead of
        rebuilding a seed-priced copy of the router."""
        try:
            ranked, rejected = host.rank({"profile": profile,
                                          "requirements": {"context": context}})
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": {
                "message": str(exc), "type": "router_error", "code": "rank"}})
        return {"profile": profile, "rank_source": "router",
                "ranked": _rank_rows(ranked), "rejected": rejected,
                "ts": int(time.time())}

    @app.post("/x/policy/build")
    def policy_build(body: PolicyBuildRequest):
        """Lower a declarative spec (weights + filter gates + selector) to a
        Σ_pol term via the core's elaborate — the policy builder's compose
        step. Returns {policy_ir, fingerprint, version}. The term is NOT
        admitted here: admission happens with the live schema and the host
        envelope where the term is used (POST /x/rank, execution)."""
        try:
            return host.build_policy(body.model_dump(exclude_none=True))
        except Exception as exc:
            msg = str(exc)
            idx = msg.find("elaborate: ")
            if idx == -1 and "lua" not in type(exc).__module__:
                raise
            return JSONResponse(status_code=400, content={"error": {
                "message": msg[idx + 11:] if idx != -1 else msg,
                "type": "invalid_request_error", "code": "invalid_policy_spec"}})

    @app.get("/x/fields")
    def list_fields():
        """The observable fields (core vocabulary + config.fields extensions)
        with sort and builder group — drives the data-driven builder."""
        return {"fields": host.field_schema()}

    @app.post("/x/policy/normalize")
    def policy_normalize(body: PolicyRankRequest):
        """Normalize a raw Σ_pol term and return {policy_ir, fingerprint,
        version} — the builder's download/identify step when the frontend
        composes the IR directly. Admission still happens at use (POST /x/rank,
        execution)."""
        try:
            return host.normalize_policy(body.policy_ir)
        except Exception as exc:
            admission = _policy_admission_error(exc)
            if admission is not None:
                return _invalid_policy_response(admission)
            return JSONResponse(status_code=400, content={"error": {
                "message": str(exc), "type": "invalid_request_error",
                "code": "invalid_policy"}})

    @app.post("/x/flow/normalize")
    def flow_normalize(body: FlowNormalizeRequest):
        """Admit + normalize a Σ_flow term and return {flow_ir, fingerprint,
        version} — the Flow Builder's review/identify step. Admission (graph
        validity + every node's policy) is the core's job, like policy_ir."""
        try:
            return host.flow_admit(body.flow_ir)
        except Exception as exc:
            admission = _flow_admission_error(exc)
            if admission is not None:
                return _invalid_flow_response(admission)
            return JSONResponse(status_code=400, content={"error": {
                "message": str(exc), "type": "invalid_request_error",
                "code": "invalid_flow"}})

    @app.post("/x/rank")
    def rank_policy_ir(body: PolicyRankRequest):
        """Dry-run ranking for a per-call Σ_pol policy term — the policy
        builder's preview. Same admission path as execution: the core checks
        the term and ∧-applies the host envelope; nothing is called."""
        contract: dict = {"policy_ir": body.policy_ir,
                          "requirements": body.requirements
                                          or {"context": body.context}}
        try:
            ranked, rejected = host.rank(contract)
        except Exception as exc:
            admission = _policy_admission_error(exc)
            if admission is None:
                raise
            return _invalid_policy_response(admission)
        return {"rank_source": "router", "ranked": _rank_rows(ranked),
                "rejected": rejected, "ts": int(time.time())}

    @app.post("/v1/compact")
    async def compact(req: CompactRequest):
        """Append-only context sealing (stateless). Seal the aged middle of the
        message array into one cheaply-routed summary and splice it back so the
        frozen prefix and the recent tail are byte-identical — keeping everything
        upstream cache-hot. Returns {messages, compacted}. The agent owns its
        context; the host stores nothing.

        Cost accounting (additive wire contract): the seal is a real billable
        LLM leg, so every response that FOLLOWS host.execute_async — the sealed
        success AND the compacted:false seal-failure — also carries "usage"
        (OpenAI token shape, the summarizer call's tokens) and "x_router" (the
        same block _build_x_router puts on chat responses), so a metering proxy
        in front can record the seal's spend instead of a $0 row. Early returns
        that precede execution made no call and carry neither key."""
        msgs = req.messages or []
        keep = max(1, req.keep_recent)
        # frozen prefix = a leading system message (the skill/tools/rules), if any
        frozen = msgs[:1] if (msgs and msgs[0].get("role") == "system") else []
        head = len(frozen)
        if len(msgs) - head <= keep:        # nothing worth sealing
            return {"messages": msgs, "compacted": False}
        recent = msgs[len(msgs) - keep:]
        aged = msgs[head:len(msgs) - keep]
        # Anti-orphan (the API-400 trap): a leading `tool` message in `recent`
        # answers an assistant tool_call that lives in `aged` and is about to be
        # dropped. Drop those orphaned tool turns at the seam.
        while recent and recent[0].get("role") == "tool":
            recent = recent[1:]
        if not aged:
            return {"messages": msgs, "compacted": False}
        contract = {
            "messages": [{"role": "system", "content": _SEAL_SYSTEM},
                         {"role": "user", "content": _render_messages(aged)}],
            "policy_ir": req.policy_ir or _DEFAULT_COMPACT_POLICY,
            "max_tokens": req.max_tokens or 512,
        }
        try:
            res = await host.execute_async(contract)
        except Exception as exc:
            admission = _policy_admission_error(exc)
            if admission is not None:
                return _invalid_policy_response(admission)
            raise

        def _costed(body: dict) -> dict:
            usage = _openai_usage(res.get("response") or {})
            if usage:
                body["usage"] = usage
            body["x_router"] = _build_x_router(res, subscription_providers)
            return body

        summary = ((res.get("response") or {}).get("text") or "").strip()
        if not summary:                     # seal failed -> never lose content
            return _costed({"messages": msgs, "compacted": False})
        sealed = {"role": "system", "content": _SEAL_PREFIX + summary}
        return _costed({"messages": frozen + [sealed] + recent, "compacted": True})

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest, request: Request):
        _session_from_header(req, request)
        return await _handle_chat(req)

    @app.post("/{profile_name}/v1/chat/completions")
    async def chat_completions_profiled(profile_name: str, req: ChatRequest, request: Request):
        """Path-addressed policy endpoint.

        POST /edge/v1/chat/completions always runs the `edge` profile, ignoring
        the client's model string. This lets each intelligence tier be its own
        base_url (…/edge, …/medium, …/dummy) so a client picks a policy by URL
        instead of a `profile:` model prefix.
        """
        _session_from_header(req, request)
        return await _handle_chat(req, profile_name=profile_name)

    def _session_from_header(req: ChatRequest, request: Request) -> None:
        """Fallback the conversation id from the X-Unhardcoded-Session header when
        the body carries none. Clients (e.g. the opencode plugin) that can set
        request headers but not extra body fields use this to get cache affinity +
        per-session metering. An explicit body `session` always wins.

        Also capture the authed consumer key the ingress proxy injects as
        x-llm-router-caller, so the per-session meter can bind sid->owner (a body
        field can't be trusted for this; only the proxy sets this header)."""
        if not req.session:
            hdr = request.headers.get("x-unhardcoded-session")
            if hdr:
                req.session = hdr
        caller = request.headers.get("x-llm-router-caller")
        if caller:
            req.caller = caller

    # ---- OpenAI Responses API surface (POST /v1/responses) ----------------
    # The inbound mirror of codex_backend.py: translate a Responses request to
    # the SAME chat contract the chat path builds, run the SAME engine, and
    # translate the result back to a Responses object / SSE. Lets Responses-only
    # clients (the Codex CLI) drive the router with all routing/policy/metering.

    @app.post("/v1/responses")
    async def responses(req: ResponsesRequest, request: Request):
        _session_from_header(req, request)
        return await _handle_responses(req)

    @app.post("/{profile_name}/v1/responses")
    async def responses_profiled(profile_name: str, req: ResponsesRequest,
                                 request: Request):
        _session_from_header(req, request)
        return await _handle_responses(req, profile_name=profile_name)

    def _responses_object_with_router(result: dict, req: ResponsesRequest,
                                      response_id: str | None = None) -> dict:
        import responses_api as _rapi
        obj = _rapi.result_to_responses_object(result, req.model or "",
                                               response_id=response_id)
        obj["x_router"] = _build_x_router(result, subscription_providers,
                                          req.session, req.caller)
        return obj

    async def _handle_responses(req: ResponsesRequest, profile_name: str | None = None):
        if profile_name is None and (msg := _policy_model_without_ir(req)):
            return _bad_policy_selector_response(msg)
        import responses_api as _rapi
        dropped = _rapi.dropped_tool_types(req.tools)
        if dropped:
            _log.warning("responses: dropped non-function tool types %s "
                         "(chat providers accept only function tools)", dropped)
        chatreq = ChatRequest(
            model=req.model or "",
            messages=_rapi.input_to_messages(req.input, req.instructions),
            tools=_rapi.tools_to_chat(req.tools),
            tool_choice=_rapi.tool_choice_to_chat(req.tool_choice),
            temperature=req.temperature,
            max_tokens=req.max_output_tokens,
            first_token_timeout_ms=req.first_token_timeout_ms,
            policy_ir=req.policy_ir,
            session=req.session,
        )
        contract = _request_to_contract(chatreq, default_profile, default_max_tokens)
        if profile_name is not None:
            contract["profile"] = profile_name

        if not req.stream:
            try:
                result = await host.execute_async(contract)
            except Exception as exc:
                admission = _policy_admission_error(exc)
                if admission is None:
                    raise
                return _invalid_policy_response(admission)
            if not result.get("ok"):
                return _openai_error_from_router(result)
            return _responses_object_with_router(result, req)
        return await _handle_responses_stream(contract, req)

    async def _handle_responses_stream(contract: dict, req: ResponsesRequest):
        import responses_api as _rapi
        task = asyncio.create_task(host.execute_async(contract))
        done, _ = await asyncio.wait({task}, timeout=_EARLY_FAIL_S,
                                     return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            try:
                result = task.result()
            except Exception as exc:
                admission = _policy_admission_error(exc)
                if admission is None:
                    raise
                return _invalid_policy_response(admission)
            if not result.get("ok"):
                return _openai_error_from_router(result)
            obj = _responses_object_with_router(result, req)

            async def gen_ready():
                yield _rapi.responses_created_event(obj)
                for ev in _rapi.responses_sse_events(obj):
                    yield ev
            return StreamingResponse(gen_ready(), media_type="text/event-stream")

        async def gen_running():
            rid = _rapi._new_id("resp")
            yield _rapi.responses_created_event({"id": rid, "model": req.model or ""})
            while not task.done():
                await asyncio.wait({task}, timeout=_HEARTBEAT_S)
                if not task.done():
                    yield _streaming.HEARTBEAT
            try:
                result = task.result()
            except Exception as exc:
                admission = _policy_admission_error(exc)
                # seq=1: continues the stream after response.created (seq=0) so
                # the Responses sequence_number stays strictly increasing.
                yield _rapi.responses_failed_event(
                    rid, admission or f"responses error: {exc}",
                    "invalid_policy" if admission else "internal_error", seq=1)
                return
            if not result.get("ok"):
                err = str(result.get("error") or "router error")
                yield _rapi.responses_failed_event(
                    rid, err, str(result.get("error") or "error"), seq=1)
                return
            obj = _responses_object_with_router(result, req, response_id=rid)
            for ev in _rapi.responses_sse_events(obj):
                yield ev
        return StreamingResponse(gen_running(), media_type="text/event-stream")

    async def _handle_chat(req: ChatRequest, profile_name: str | None = None):
        if profile_name is None and (msg := _policy_model_without_ir(req)):
            return _bad_policy_selector_response(msg)
        contract = _request_to_contract(req, default_profile, default_max_tokens)
        if profile_name is not None:
            contract["profile"] = profile_name  # the path is authoritative

        # Σ_flow: a per-call composition of routed calls. Takes precedence over
        # policy/model — the flow's nodes each carry their own policy. A flow runs
        # to completion (the answer is the sink node's output), so there is no
        # token-by-token source; a streaming client still gets SSE by pseudo-
        # streaming the finished result, exactly like a non-streamable backend.
        if req.flow_ir is not None:
            if not req.stream:
                try:
                    result = await host.execute_flow_async(req.flow_ir, contract)
                except Exception as exc:
                    admission = _flow_admission_error(exc)
                    if admission is None:
                        raise
                    return _invalid_flow_response(admission)
                if not result.get("ok"):
                    return _openai_error_from_router(result)
                return _router_response_to_openai(result, req.model,
                                                  subscription_providers,
                                                  session=req.session,
                                                  owner=req.caller)
            # Streaming flow: run as a task and wait briefly for a fast failure
            # (admission 400, or a node failing instantly) so it stays a clean
            # JSON error. Anything slower commits to SSE and HEARTBEATs while the
            # flow runs (flows have no token stream, so without this they sat
            # silent ~60s and the idle timeout cut them into an empty 200).
            task = asyncio.create_task(host.execute_flow_async(req.flow_ir, contract))
            done, _ = await asyncio.wait({task}, timeout=_EARLY_FAIL_S,
                                         return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                try:
                    result = task.result()
                except Exception as exc:
                    admission = _flow_admission_error(exc)
                    if admission is None:
                        raise
                    return _invalid_flow_response(admission)
                if not result.get("ok"):
                    return _openai_error_from_router(result)
                return StreamingResponse(_pseudo_stream(result, req),
                                         media_type="text/event-stream")
            return StreamingResponse(_flow_stream(task, req),
                                     media_type="text/event-stream")

        if not req.stream:
            # Async driver: the Lua VM is touched only between awaits, so one
            # shared LuaRuntime overlaps many concurrent requests on one loop.
            try:
                result = await host.execute_async(contract)
            except Exception as exc:
                admission = _policy_admission_error(exc)
                if admission is None:
                    raise
                return _invalid_policy_response(admission)
            if result.get("ok"):
                return _router_response_to_openai(result, req.model,
                                                  subscription_providers,
                                                  session=req.session,
                                                  owner=req.caller)
            return _openai_error_from_router(result)
        return await _handle_stream(contract, req)

    async def _handle_stream(contract: dict, req: ChatRequest):
        """True streaming with fallback before the commit point (the first
        content delta). Pre-commit exhaustion returns the normal JSON error;
        a complete-but-uncommitted success (mocked backends, api_kinds
        without stream variants) is pseudo-streamed as SSE."""
        queue: asyncio.Queue = asyncio.Queue()
        commit: asyncio.Future = asyncio.get_running_loop().create_future()

        override = None
        if streaming_call is not None:
            async def override(request):
                async def emit(delta: str) -> None:
                    if not commit.done():
                        commit.set_result(True)
                    await queue.put(delta)
                return await streaming_call(request, emit)

        task = asyncio.create_task(host.execute_async(contract, call_override=override))

        # A single-model policy is fast: keep the exact prior behavior — wait for
        # the commit point or completion; a pre-delta failure is a clean JSON
        # error. Only a Σ_flow runs long enough (N routed nodes) to need the
        # heartbeat, so only it commits to SSE early.
        is_flow = bool(contract.get("flow_ir"))
        wait_kwargs = ({"timeout": _EARLY_FAIL_S} if is_flow else {})
        done, _ = await asyncio.wait({commit, task}, return_when=asyncio.FIRST_COMPLETED,
                                     **wait_kwargs)

        # Committed (deltas flowing) -> stream. Still running past the flow's brief
        # grace window -> stream too (the SSE generator flushes a first byte at
        # once and heartbeats while it runs, so the 60s ALB idle timeout can't cut
        # a long ensemble call mid-execution — the cause of dropped calls + empty
        # Activity rows). Only a completed-without-commit result is answered
        # synchronously (JSON error, or pseudo-stream of a finished body).
        if commit.done() or task not in done:
            return StreamingResponse(_sse_gen(queue, task, req),
                                     media_type="text/event-stream")
        try:
            result = task.result()
        except Exception as exc:
            admission = _policy_admission_error(exc)
            if admission is None:
                raise
            return _invalid_policy_response(admission)
        if not result.get("ok"):
            return _openai_error_from_router(result)
        return StreamingResponse(_pseudo_stream(result, req),
                                 media_type="text/event-stream")

    def _final_chunk_parts(result: dict, session: str | None = None,
                           owner: str | None = None):
        resp = result.get("response") or {}
        usage = _openai_usage(resp)
        # x_router + the per-session fold are shared with the unary path; the
        # streaming final chunk folds the session here so stream:true clients
        # (e.g. opencode) still accumulate into the session total.
        x_router = _build_x_router(result, subscription_providers, session, owner)
        return resp, usage or None, x_router

    async def _pseudo_stream(result: dict, req: ChatRequest):
        resp, usage, x_router = _final_chunk_parts(result, req.session, req.caller)
        stream_id = _streaming.new_stream_id()
        model = resp.get("raw_model") or req.model or ""
        yield _streaming.encode_role_chunk(stream_id, model)
        if resp.get("text"):
            yield _streaming.encode_text_chunk(stream_id, model, resp["text"])
        yield _streaming.encode_final_chunk(stream_id, model, resp.get("finish_reason"),
                                            resp.get("tool_calls"), usage, x_router)
        yield _streaming.DONE_EVENT

    async def _flow_stream(task: "asyncio.Task", req: ChatRequest):
        # A Σ_flow has no token stream of its own — it runs to completion. Flows
        # bypass _sse_gen, so the keepalive there never reached them: they sat
        # silent in execute_flow_async for ~60s and the 60s idle timeout cut them,
        # surfacing as an empty 200 with no trace. Emit a first byte at once and
        # HEARTBEAT while it runs; then stream the result, or on failure emit the
        # decision_trace (which node failed) + the error so it is VISIBLE in
        # Activity and to the client instead of nothing.
        stream_id = _streaming.new_stream_id()
        model = req.model or ""
        yield _streaming.encode_role_chunk(stream_id, model)
        while not task.done():
            await asyncio.wait({task}, timeout=_HEARTBEAT_S)
            if not task.done():
                yield _streaming.HEARTBEAT
        try:
            result = task.result()
        except Exception as exc:
            admission = _flow_admission_error(exc)
            yield _streaming.encode_error_event(
                admission or f"flow error: {exc}",
                "invalid_flow" if admission else "flow_error")
            yield _streaming.DONE_EVENT
            return
        resp, usage, x_router = _final_chunk_parts(result, req.session, req.caller)
        fmodel = resp.get("raw_model") or model
        if result.get("ok"):
            if resp.get("text"):
                yield _streaming.encode_text_chunk(stream_id, fmodel, resp["text"])
            yield _streaming.encode_final_chunk(stream_id, fmodel, resp.get("finish_reason"),
                                                resp.get("tool_calls"), usage, x_router)
        else:
            err = str(result.get("error") or "flow failed")
            yield _streaming.encode_final_chunk(stream_id, fmodel, "error", None, usage, x_router)
            yield _streaming.encode_error_event(err, err)
        yield _streaming.DONE_EVENT

    async def _sse_gen(queue: "asyncio.Queue", task: "asyncio.Task", req: ChatRequest):
        stream_id = _streaming.new_stream_id()
        model = req.model or ""
        yield _streaming.encode_role_chunk(stream_id, model)  # first byte, now
        streamed_any = False
        while True:
            getter = asyncio.create_task(queue.get())
            kind, payload = await _await_delta_or_beat(getter, task, _HEARTBEAT_S)
            if kind == "delta":
                streamed_any = True
                yield _streaming.encode_text_chunk(stream_id, model, payload)
                continue
            if kind == "done":
                while not queue.empty():
                    streamed_any = True
                    yield _streaming.encode_text_chunk(stream_id, model, queue.get_nowait())
                break
            yield _streaming.HEARTBEAT  # 'beat' — keep the line warm
        result = task.result()
        if result.get("ok"):
            resp, usage, x_router = _final_chunk_parts(result, req.session, req.caller)
            fmodel = resp.get("raw_model") or model
            # A flow that never committed a delta (its result was assembled, not
            # streamed) still owes the client its text before the final chunk.
            if not streamed_any and resp.get("text"):
                yield _streaming.encode_text_chunk(stream_id, fmodel, resp["text"])
            yield _streaming.encode_final_chunk(stream_id, fmodel,
                                                resp.get("finish_reason"),
                                                resp.get("tool_calls"), usage, x_router)
        else:
            # A failed flow/call still carries its decision_trace (which node
            # failed, the error kind). Emit it as the final chunk so the failure
            # is VISIBLE in Activity (provider + trace recorded) instead of an
            # empty 200 with no trace, THEN surface the error to the client so
            # opencode sees a reason rather than silently nothing.
            _resp, usage, x_router = _final_chunk_parts(result, req.session, req.caller)
            err = str(result.get("error") or "stream failed")
            yield _streaming.encode_final_chunk(stream_id, model, "error", None,
                                                usage, x_router)
            yield _streaming.encode_error_event(err, err)
        yield _streaming.DONE_EVENT

    return app


def _request_to_contract(
    req: ChatRequest,
    default_profile: str,
    default_max_tokens: int | None = DEFAULT_MAX_TOKENS_FALLBACK,
) -> dict:
    model = (req.model or "").strip()
    contract: dict = {"messages": req.messages or []}

    if not model:
        contract["profile"] = default_profile
    elif model.startswith("profile:"):
        contract["profile"] = model[len("profile:"):] or default_profile
    elif model.startswith("family:"):
        family = model[len("family:"):]
        contract["profile"] = default_profile
        if family:
            contract["requirements"] = {"model_family": family}
    elif model.startswith("pin:"):
        rest = model[len("pin:"):]
        contract["profile"] = default_profile
        if "/" in rest:
            provider, family = rest.split("/", 1)
            if provider and family:
                contract["requirements"] = {"pin": {"provider": provider, "model": family}}
    else:
        contract["profile"] = default_profile

    if req.tools is not None:
        contract["tools"] = req.tools
    if req.tool_choice is not None:
        contract["tool_choice"] = req.tool_choice
    if req.response_format is not None:
        contract["response_format"] = req.response_format
    if req.temperature is not None:
        contract["temperature"] = req.temperature
    if req.seed is not None:
        contract["seed"] = req.seed
    max_tokens = req.max_tokens if req.max_tokens is not None else default_max_tokens
    if max_tokens is not None:
        contract["max_tokens"] = max_tokens
    if req.first_token_timeout_ms is not None:
        contract["first_token_timeout_ms"] = req.first_token_timeout_ms
    if req.policy_ir is not None:
        # Forwarded verbatim: the CORE is the admission boundary (check ->
        # normalize -> eval, bounded), and it ∧-applies the host envelope.
        # The shim never interprets or pre-validates the term (one admission,
        # one place); it only translates the core's refusal to a 400.
        contract["policy_ir"] = req.policy_ir

    if req.session:
        # The session rides into the contract so the fold can attribute the call
        # to it; and the host resolves the session's hot route NOW (snapshot,
        # before eval) into cache_hot_route, which the cache_hot field getter
        # reads off ctx.request. A brand-new session has no hot route -> the key
        # is simply absent -> cache_hot is false for everyone (no phantom pin).
        contract["session"] = req.session
        hot = host_store.hot_route(req.session)
        if hot is not None:
            contract["cache_hot_route"] = hot

    return contract


def _invalid_policy_response(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": {
        "message": f"policy_ir rejected at admission: {message}",
        "type": "invalid_request_error",
        "code": "invalid_policy",
    }})


def _policy_model_without_ir(req) -> "str | None":
    """A request whose `model` carries the `policy:` selector but supplies NO
    per-call policy is a client error: a policy must travel as a `policy_ir`
    (or `flow_ir`) TERM in the request body, never as the `model` string. Such a
    request used to fall through to the `default` profile silently — routing with
    a policy the caller believed it sent but never did. Return the 400 message,
    or None when the request is fine. (Skip on the profiled endpoints — there the
    URL profile is authoritative and `model` is genuinely irrelevant.)"""
    model = (getattr(req, "model", "") or "").strip()
    if (model.startswith("policy:")
            and getattr(req, "policy_ir", None) is None
            and getattr(req, "flow_ir", None) is None):
        return (
            f"model {model!r} is not a policy selector. A per-call policy must be "
            f"sent as a `policy_ir` term in the request body (or a `flow_ir`); the "
            f"`model` field does not select a policy. Recognized `model` prefixes "
            f"are `profile:`, `family:` and `pin:`.")
    return None


def _bad_policy_selector_response(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": {
        "message": message,
        "type": "invalid_request_error",
        "code": "policy_in_model_field",
    }})


def _policy_admission_error(exc: Exception) -> str | None:
    """The core's per-call policy errors are raised as Lua errors prefixed
    'ir: ' (admission: sorts/arity/bounds, 'expected a Policy term') or
    'ir.constrain: ' (the envelope ∧ rejecting a non-policy term). Returns
    the human part of the message for those, None for anything else (which
    is then a genuine 500, not the caller's fault)."""
    msg = str(exc)
    for prefix in ("ir: ", "ir.constrain: "):
        idx = msg.find(prefix)
        if idx != -1:
            return msg[idx + len(prefix):]
    return None


def _invalid_flow_response(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": {
        "message": f"flow_ir rejected at admission: {message}",
        "type": "invalid_request_error",
        "code": "invalid_flow",
    }})


def _flow_admission_error(exc: Exception) -> str | None:
    """A Σ_flow term rejected at admission is raised as FlowAdmissionError with
    a 'flow: ' prefix (host-side) — or, if a node's embedded policy is the thing
    that fails, as a core 'ir: ' Lua error. Either is the caller's fault (400);
    anything else is a genuine 500."""
    msg = str(exc)
    for prefix in ("flow: ", "ir: "):
        idx = msg.find(prefix)
        if idx != -1:
            return msg[idx + len(prefix):]
    return None


_TRACE_RANKED_TOP_N = 10

# How long _handle_stream waits for a synchronous failure before committing to an
# SSE stream (so admission 400s stay clean JSON), and how often _sse_gen emits a
# keepalive while a slow flow runs. The heartbeat must stay well under the 60s ALB
# idle timeout that was cutting long ensemble calls mid-execution.
_EARLY_FAIL_S = 1.5
_HEARTBEAT_S = 15.0


async def _await_delta_or_beat(getter: "asyncio.Task", task: "asyncio.Task",
                               timeout: float) -> tuple:
    """One step of the streaming loop: wait for the next delta, the task to
    finish, or a heartbeat tick (`timeout`). Returns ('delta', text), ('done',
    None), or ('beat', None). A 'beat' means the task is still running and no
    delta arrived in `timeout`s — the caller emits a keepalive so an idle hop
    (the 60s ALB) can't cut a slow flow mid-run."""
    await asyncio.wait({getter, task}, timeout=timeout,
                       return_when=asyncio.FIRST_COMPLETED)
    if getter.done():
        return ("delta", getter.result())
    getter.cancel()
    if task.done():
        return ("done", None)
    return ("beat", None)


def _cap_ranked(trace):
    """Return `trace` with its `ranked` capped to top-N (+ ranked_total), or
    unchanged. `ranked` is the WHOLE ranked catalog (hundreds of candidates) and
    alone bloats a response to ~95 KB (measured); the chosen route plus the next
    few fallbacks is all Activity needs. decision_path (the real attempts) is
    untouched."""
    if not isinstance(trace, dict):
        return trace
    ranked = trace.get("ranked")
    if isinstance(ranked, list) and len(ranked) > _TRACE_RANKED_TOP_N:
        return {**trace, "ranked": ranked[:_TRACE_RANKED_TOP_N],
                "ranked_total": len(ranked)}
    return trace


def _trim_trace(trace):
    """Bound the decision_trace emitted to the client. Caps the top-level `ranked`
    AND, for a Σ_flow, the `ranked` nested inside each per-node decision_trace —
    an N-node flow otherwise emits N× the full catalog (the reason the flow trace
    overflowed the proxy tail and Activity showed nothing for the ensemble). The
    per-node provider / peer / decision_path / tokens — what makes the flow
    visible — are kept intact."""
    if not isinstance(trace, dict):
        return trace or None
    out = _cap_ranked(trace)
    nodes = out.get("flow_nodes")
    if isinstance(nodes, list) and nodes:
        trimmed = [{**n, "decision_trace": _cap_ranked(n["decision_trace"])}
                   if isinstance(n, dict) and isinstance(n.get("decision_trace"), dict)
                   else n
                   for n in nodes]
        out = {**out, "flow_nodes": trimmed}
    return out


# Fraction of the input price billed for prompt-cache-READ tokens, used ONLY in
# the computed fallback (providers that report no cost). Most caching providers
# discount cache reads ~10x; this is the conservative typical factor.
_CACHE_READ_FACTOR = 0.1


def _cost_basis(result: dict, subscription_providers=frozenset()) -> "str | None":
    """How cost_usd was determined for THIS call — a raw fact for cost analysis:
    'subscription' (codex, $0), 'reported' (the provider's OWN usage.cost,
    authoritative — an INDEPENDENT signal vs the list price), 'computed' (derived
    from the ranked list price — tautological vs that price), or None when there is
    no price to compute from. Single source of the cost tiering."""
    chosen = result.get("chosen") or {}
    resp = result.get("response") or {}
    if chosen.get("provider_id") in subscription_providers:
        return "subscription"
    reported = resp.get("cost_reported")
    if isinstance(reported, (int, float)) and not isinstance(reported, bool) and reported >= 0:
        return "reported"
    if chosen.get("price_in") is not None or chosen.get("price_out") is not None:
        return "computed"
    return None


def _executed_cost_usd(result: dict, subscription_providers=frozenset()) -> float | None:
    """Dollars actually spent on THIS request, accurate across providers, tiered by
    _cost_basis: (1) subscription backends (codex) cost $0; (2) the provider's OWN
    reported cost (e.g. OpenRouter `usage.cost`) — authoritative, net of prompt-cache
    discounts; (3) computed from the ranked price, billing cache-read tokens at a
    fraction. None when uncomputable (read-time estimator is the fallback)."""
    basis = _cost_basis(result, subscription_providers)
    resp = result.get("response") or {}
    if basis == "subscription":
        return 0.0
    if basis == "reported":
        return round(float(resp["cost_reported"]), 6)
    if basis != "computed":
        return None
    # (3) compute from the raw price, discounting cache-read input tokens
    chosen = result.get("chosen") or {}
    pin = chosen.get("raw_price_in", chosen.get("price_in"))
    pout = chosen.get("raw_price_out", chosen.get("price_out"))
    # Back-compat with engine versions that only returned the ranking price:
    # divide out the fictitious ranking multiplier so billing stays on raw
    # list/quote price.
    if chosen.get("raw_price_in") is None and chosen.get("raw_price_out") is None:
        import settings
        mult = chosen.get("price_multiplier")
        if not isinstance(mult, (int, float)) or isinstance(mult, bool):
            mkey = f"{chosen.get('provider_id')}.price_multiplier"
            mult = settings.get(mkey) if mkey in settings.SCHEMA else 1.0
        if mult and mult > 0:
            pin = pin / mult if pin is not None else pin
            pout = pout / mult if pout is not None else pout
    tin = resp.get("tokens_in") or 0
    cached = resp.get("tokens_cached") or 0
    uncached = max(0, tin - cached)
    cost = (uncached / 1e6 * (pin or 0)
            + cached / 1e6 * (pin or 0) * _CACHE_READ_FACTOR
            + (resp.get("tokens_out") or 0) / 1e6 * (pout or 0))
    # A negative price (unpriced sentinel / shadow scarcity price) must never bill
    # negative — clamp at the source (we once saw a large negative-spend row that
    # was exactly tokens × a negative chosen price).
    return max(0.0, round(cost, 6))


def _openai_usage(response: dict) -> dict:
    """OpenAI-shape `usage` block from a router response — the single builder
    shared by the unary chat body, the streaming final chunk and /v1/compact.
    Empty dict when the provider reported no token counts (caller omits the
    key, per the additive wire contract)."""
    usage: dict = {}
    for src_key, dst_key in (("tokens_in", "prompt_tokens"),
                             ("tokens_out", "completion_tokens"),
                             ("tokens_total", "total_tokens")):
        if response.get(src_key) is not None:
            usage[dst_key] = response[src_key]
    # Standard OpenAI cache field so clients (opencode, etc.) parse cache reads
    # natively — not just our x_router.
    if response.get("tokens_cached"):
        usage["prompt_tokens_details"] = {"cached_tokens": response["tokens_cached"]}
    return usage


def _build_x_router(result: dict, subscription_providers=frozenset(),
                    session: str | None = None, owner: str | None = None) -> dict:
    """The `x_router` metadata block shared by every response surface (chat
    unary, chat streaming final chunk, /v1/responses) plus the per-session meter
    fold. Single source so the fields and the fold stay identical across
    surfaces. When `session` is set the call is folded into its running total and
    `session_acc` is attached. Idempotent per request — call once per response."""
    resp = result.get("response") or {}
    chosen = result.get("chosen") or {}
    x_router = {
        "provider": chosen.get("provider_id"),
        "model_family": chosen.get("model_family"),
        "served_model_id": chosen.get("served_model_id"),
        # the route that actually served the call (marketplace peer or provider),
        # stamped by the engine on `chosen` — the per-route identity for stats.
        "served_by": chosen.get("served_by"),
        "price_in": chosen.get("price_in"),
        "price_out": chosen.get("price_out"),
        "cost_usd": _executed_cost_usd(result, subscription_providers),
        "cost_basis": _cost_basis(result, subscription_providers),
        "tokens_cached": resp.get("tokens_cached"),
        "policy_fingerprint": (result.get("trace") or {}).get("policy_fingerprint"),
        "decision_trace": _trim_trace(result.get("trace")),
        "compact": _compact_suggested(resp),
    }
    if session:
        # #4b: derive the running total from the committed `calls` and add THIS
        # in-flight call (not yet in the ledger). Owner is derived from the
        # session's earliest call, so no explicit binding is needed here.
        prior = host_store.session_totals(session)
        x_router["session_acc"] = {
            "calls": prior["calls"] + 1,
            "tokens_in": prior["tokens_in"] + (resp.get("tokens_in") or 0),
            "tokens_out": prior["tokens_out"] + (resp.get("tokens_out") or 0),
            "tokens_cached": prior["tokens_cached"] + (resp.get("tokens_cached") or 0),
            "cost_usd": round(prior["cost_usd"] + (x_router["cost_usd"] or 0.0), 6),
        }
    return x_router


def _router_response_to_openai(result: dict, requested_model: str,
                               subscription_providers=frozenset(),
                               session: str | None = None,
                               owner: str | None = None) -> dict:
    response = result.get("response") or {}
    chosen = result.get("chosen") or {}

    message: dict = {"role": "assistant", "content": response.get("text") or ""}
    if response.get("tool_calls"):
        message["tool_calls"] = response["tool_calls"]

    out: dict = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": (
            response.get("raw_model")
            or chosen.get("served_model_id")
            or requested_model
            or ""
        ),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": response.get("finish_reason") or "stop",
        }],
    }

    usage = _openai_usage(response)
    if usage:
        out["usage"] = usage

    # Non-standard router metadata: ignored by OpenAI clients, useful for
    # debugging. Shared with the streaming + /v1/responses surfaces; when a
    # session is set it also folds this call into the session's running total.
    out["x_router"] = _build_x_router(result, subscription_providers, session, owner)
    return out


def _attempt_summary(trace: dict) -> str | None:
    """One line per failed attempt: provider/family=kind(status) "message".
    This is what makes a 502 self-explanatory to a client that only sees
    the error body."""
    parts: list[str] = []
    for e in (trace or {}).get("decision_path") or []:
        if e.get("event") != "attempted" or not e.get("error_kind"):
            continue
        label = str(e.get("provider_id") or "?")
        if e.get("model_family"):
            label += f"/{e['model_family']}"
        bit = f"{label}={e['error_kind']}"
        if e.get("http_status"):
            bit += f"({e['http_status']})"
        msg = str(e.get("error_message") or "").strip()
        if msg:
            bit += f' "{msg[:80]}"'
        parts.append(bit)
    return "; ".join(parts) or None


def _openai_error_from_router(result: dict) -> JSONResponse:
    error_kind = str(result.get("error") or "unknown")

    # The router returns either a bare kind (abort path: bad_request /
    # context_overflow; or no_candidates) or "exhausted: <kind>" when it tried
    # candidates. Normalize to the bare kind before mapping so both forms map
    # the same way (e.g. an aborting bad_request → 400, not 502).
    kind = error_kind[len("exhausted: "):] if error_kind.startswith("exhausted: ") else error_kind

    if "not initialized" in error_kind:
        status = 500
    elif kind == "no_candidates":
        status = 503
    elif kind == "auth_error":
        status = 401
    elif kind == "rate_limit":
        status = 429
    elif kind in ("bad_request", "context_overflow"):
        status = 400
    elif kind == "timeout":
        status = 504  # upstream (or a flow node's seller) timed out — Gateway Timeout
    else:
        status = 502

    # router's error strings already start with "exhausted: " when candidates
    # were tried — don't double-prefix.
    message = error_kind if error_kind.startswith("exhausted:") else f"router: {error_kind}"

    # Surface the per-attempt detail: append a provider-by-provider summary to
    # the message and ship the full trace in x_router (same field the ingress
    # proxy already extracts and the dashboard stores as decision_trace).
    trace = result.get("trace") or {}
    summary = _attempt_summary(trace)
    if summary:
        message = f"{message} — {summary}"
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "router_error",
                "code": error_kind,
            },
            "x_router": {
                "provider": None,
                "model_family": None,
                "served_model_id": None,
                "decision_trace": trace or None,
            },
        },
    )


def _openai_error(message: str, type_: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": None}},
    )
