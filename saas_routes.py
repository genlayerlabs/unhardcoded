"""Intent-to-term adapter. All admission, ranking and execution belong to Σ_pol."""
from __future__ import annotations

import asyncio
import contextvars
import math
import re

from fastapi import Request
from fastapi.responses import JSONResponse

import control_plane_client as cp

from provider_connections import connections, credential_names, label

_active = contextvars.ContextVar("saas_host", default=None)
_revision = contextvars.ContextVar("saas_revision", default=None)


class ScopedHost:
    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(_active.get() or self.base, name)

    async def execute_async(self, contract, **kwargs):
        host = _active.get() or self.base
        if host._tenant_id is not None and contract.get("session"):
            scope_id = host._env.get('SAAS_TENANT_SCOPE', str(host._tenant_id))
            contract = {**contract, "session": f"tenant:{scope_id}:{contract['session']}"}
        result = await host.execute_async(contract, **kwargs)
        if revision := _revision.get():
            result.setdefault("trace", {}).update(revision)
        return result


def choices(host):
    """Read candidate identities from the engine, never a second catalog."""
    term = ["policy", ["and", ["meets_req"], ["not", ["is", "disabled"]]],
            ["zero"], ["ordered"], ["id"], ["always", {"action": "abort"}]]
    ranked, _ = host.rank({"policy_ir": term})
    out = {}
    for row in ranked:
        c = row["candidate"]
        pid, family = c["provider_id"], c["model_family"]
        if pid not in (host._tenant_allowed or set()):
            continue
        key = f"{pid}|{family}"
        out[key] = {"id": key, "provider": pid, "family": family,
                    "label": f"{family} · {label(pid)}",
                    "tools": bool((c.get("capabilities") or {}).get("supports_tools")),
                    "price_in": finite(c.get("raw_price_in", c.get("price_in"))),
                    "price_out": finite(c.get("raw_price_out", c.get("price_out")))}
    return sorted(out.values(), key=lambda c: (c["provider"], c["family"]))


def finite(value):
    return value if isinstance(value, (float, int)) and math.isfinite(value) else None


def compile_intent(intent):
    if not isinstance(intent, dict):
        raise ValueError("Choose a route configuration.")
    goal = intent.get("goal", "reliability")
    preferences = intent.get('allowed_preferences', [])
    if (not isinstance(preferences, list) or len(preferences) > 3
            or any(p not in ('cost', 'speed', 'reliability') for p in preferences)):
        raise ValueError('Choose supported task preferences.')
    workload = intent.get("workload", "chat")
    if goal not in {"reliability", "cost", "speed"} or workload not in {"chat", "agent", "extraction"}:
        raise ValueError("Choose a supported task and goal.")
    targets = intent.get("targets")
    if not isinstance(targets, list) or not 1 <= len(targets) <= 4:
        raise ValueError("Choose a primary model and up to three alternatives.")
    pairs = []
    for value in targets:
        if not isinstance(value, str) or len(value) > 240 or value.count("|") != 1:
            raise ValueError("Choose models from your connected providers.")
        provider, family = value.split("|", 1)
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', provider) or not family:
            raise ValueError("Choose a valid provider and model from the dataplane catalog.")
        pairs.append({"provider": provider, "model": family})
    if len(set(targets)) != len(targets):
        raise ValueError("Choose each model only once.")
    timeout = intent.get("timeout_seconds", 8)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 20:
        raise ValueError("Response timeout must be between 1 and 20 seconds.")
    predicates = [["and", ["provider_eq", p["provider"]], ["family_eq", p["model"]]] for p in pairs]
    gates = ["and", ["meets_req"], ["not", ["is", "disabled"]], ["or", *predicates]]
    if workload == "agent":
        gates.append(["has_cap", "supports_tools"])
    if workload == "extraction":
        gates.append(["has_cap", "supports_json_mode"])
    for field in ("price_in", "price_out"):
        value = intent.get(f"max_{field}")
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1000:
                raise ValueError("Price limits must be positive numbers up to $1,000 per million tokens.")
            gates.append(["cmp", field, "le", value])
    scorer, selector = ["zero"], ["chain", pairs]
    if goal == "cost":
        # Token-price estimate for a stated 80% input / 20% output mix.
        scorer = ["neg", ["normalize", ["add",
                   ["scale", .8, ["field", "price_in"]],
                   ["scale", .2, ["field", "price_out"]]]]]
        gates.extend([["cmp", "price_in", "le", 1e9], ["cmp", "price_out", "le", 1e9]])
        selector = ["argmax"]
    elif goal == "speed":
        scorer = ["neg", ["normalize", ["field", "latency_ms"]]]
        selector = ["argmax"]
    # Never retry refusals, invalid requests or a stream already delivered.
    failure = ["always", {"action": "abort"}]
    for reason in ("timeout", "server_error", "network_error", "rate_limit",
                   "auth_error", "payment_required", "model_unavailable", "context_overflow"):
        failure = ["override", failure, reason, {"action": "next_candidate"}]
    return ["policy", gates, scorer, selector,
            ["set_param", "timeout_ms", timeout * 1000], failure]


def preview(host, intent):
    term = compile_intent(intent)
    normalized = host.normalize_policy(term, admit=True)
    ranked, rejected = host.rank({"policy_ir": normalized["policy_ir"]})
    rows = []
    for r in ranked:
        c = r["candidate"]
        rows.append({"id": f"{c['provider_id']}|{c['model_family']}",
                     "provider": c["provider_id"], "family": c["model_family"],
                     "label": f"{c['model_family']} · {label(c['provider_id'])}",
                     "price_in": finite(c.get("raw_price_in", c.get("price_in"))),
                     "price_out": finite(c.get("raw_price_out", c.get("price_out")))})
    selected = set(intent["targets"])
    reasons = {f"{r.get('provider')}|{r.get('model', r.get('model_family'))}": r.get("reason", "requirements") for r in rejected}
    exclusions = []
    for target in selected - {r["id"] for r in rows}:
        reason = reasons.get(target, "not_available")
        if "price" in reason:
            explanation = "Above your price limit, or its price is unknown."
        elif "capability" in reason or "tools" in reason:
            explanation = "Does not support the capabilities required by this task."
        elif "disabled" in reason or reason in {"not_available", "not"}:
            explanation = "Not available through your connected provider accounts."
        else:
            explanation = "Does not meet the mandatory requirements."
        exclusions.append({"id": target, "label": target.replace("|", " · "), "reason": explanation})
    warnings = []
    if len(rows) == 1:
        warnings.append("Only one model qualifies. There is no fallback if it fails.")
    if intent.get("goal") == "speed":
        warnings.append("Speed uses observed response latency. New models may have no measurements yet; this is not a latency guarantee.")
    if intent.get("goal") == "cost":
        warnings.append("Price ranking assumes 80% input and 20% output tokens. Your actual token mix and provider discounts can change the cost.")
    task_policies, task_previews = {}, []
    for preference in intent.get('allowed_preferences', []):
        variant = host.normalize_policy(compile_intent({**intent, 'goal': preference,
                                                        'allowed_preferences': []}), admit=True)
        task_policies[preference] = {'policy_ir': variant['policy_ir'], 'policy_id': variant['policy_id']}
        candidates, _ = host.rank({'policy_ir': variant['policy_ir']})
        task_previews.append({'preference': preference, 'eligible': len(candidates),
            'first': (label(candidates[0]['candidate']['provider_id']) + ' · ' + candidates[0]['candidate']['model_family']) if candidates else None})
    return {**normalized, "ranked": rows, "excluded": exclusions, "warnings": warnings, 'task_previews': task_previews,
            "execution": {"timeout_ms": intent.get("timeout_seconds", 8) * 1000,
                          "first_token_timeout_ms": intent.get("timeout_seconds", 8) * 1000,
                          "task_policies": task_policies}}


def install(app, scoped, handle_chat, chat_request):
    @app.middleware("http")
    async def tenant_context(request: Request, call_next):
        raw = request.headers.get("x-llm-router-tenant")
        scope_headers = [request.headers.get(name) for name in (
            "x-unhardcoded-scope-version", "x-unhardcoded-project", "x-unhardcoded-environment")]
        if raw is None:
            if any(value is not None for value in scope_headers):
                return JSONResponse({"error": {"message": "Scope requires trusted tenant context"}}, status_code=403)
            return await call_next(request)
        if not cp.enabled() or not cp.internal_secret_ok(request.headers):
            return JSONResponse({"error": {"message": "Untrusted tenant context"}}, status_code=403)
        try:
            tenant_id = int(raw)
            if tenant_id <= 0:
                raise ValueError()
        except ValueError:
            return JSONResponse({"error": {"message": "Invalid tenant"}}, status_code=400)
        scope = {}
        if any(value is not None for value in scope_headers):
            try:
                version, project, environment = scope_headers
                if version != "2" or any(not value or not value.isascii() or not value.isdigit() for value in (project, environment)):
                    raise ValueError()
                scope = {"project_id": int(project), "environment_id": int(environment)}
                if min(scope.values()) <= 0:
                    raise ValueError()
            except (ValueError, TypeError):
                return JSONResponse({"error": {"message": "Invalid project/environment scope"}}, status_code=400)
        allowed = {"/v1/chat/completions", "/v1/responses", "/x/saas/catalog", "/x/saas/preview", "/x/saas/test", "/x/saas/connections"}
        if request.url.path not in allowed:
            return JSONResponse({"error": {"message": "Unsupported tenant endpoint"}}, status_code=404)
        # Read current credentials for each request: revocation/rotation is immediate.
        # The bounded request-local VM cannot retain stale tenant secrets/health.
        try:
            env, connections_config = await cp.tenant_connections(tenant_id, credential_names(scoped.base.catalog()), **scope)
        except cp.RouteUnavailable:
            return JSONResponse({"error": {"message": "Scoped credentials unavailable"}}, status_code=503)
        child = await asyncio.to_thread(scoped.base.for_tenant, tenant_id, env, (), connections_config, **scope)
        if request.url.path != '/x/saas/connections':
            from tenant_providers import prepare
            await prepare(child)
        env_token = cp.activate_tenant_env(child._env)
        try:
            host_token = _active.set(child)
            rev_token = _revision.set({
                "route": request.headers.get("x-unhardcoded-route"),
                "route_revision": request.headers.get("x-unhardcoded-revision"),
                "policy_id": request.headers.get("x-unhardcoded-policy-id"),
                'routing_preference': request.headers.get('x-unhardcoded-preference'),
                **scope,
            })
            try:
                return await call_next(request)
            finally:
                _revision.reset(rev_token)
                _active.reset(host_token)
        finally:
            cp.reset_tenant_env(env_token)

    def gate(request):
        return cp.internal_secret_ok(request.headers) and _active.get() is not None

    @app.get("/x/saas/catalog")
    def catalog(request: Request):
        if not gate(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        return {"models": choices(scoped), "connection_errors": getattr(scoped, '_connection_errors', {})}

    @app.get("/x/saas/connections")
    def connection_catalog(request: Request):
        if not gate(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        return {"connections": connections(scoped)}

    @app.post("/x/saas/preview")
    async def preview_route(request: Request):
        if not gate(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        try:
            return await asyncio.to_thread(preview, scoped, await request.json())
        except (ValueError, TypeError, KeyError) as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)

    @app.post("/x/saas/test")
    async def test_route(request: Request):
        if not gate(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        body = await request.json()
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
            return JSONResponse({"error": {"message": "Enter a test prompt of up to 8,000 characters."}}, status_code=400)
        try:
            built = await asyncio.to_thread(preview, scoped, body.get("intent"))
            req = chat_request(model="", policy_ir=built["policy_ir"], max_tokens=512,
                               messages=[{"role": "user", "content": prompt}],
                               **{k: v for k, v in built['execution'].items() if k != 'task_policies'})
            return await handle_chat(req)
        except (ValueError, TypeError, KeyError) as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
