"""
llm_router_host.py — reference Python embedding of router.lua via lupa.

Loads `router.lua` + `config.lua` (+ optional `metrics.lua`) into a Lua VM,
installs the `host` table the router needs for I/O, and exposes a small
Python API: init / info / rank / execute / dump_state.

`call_provider` defaults to a mock that returns canned responses keyed by
(provider_id, model_family). Tests inject responses via set_mock_response().
A real HTTP backend can be plugged in by passing call_provider=... .

Dependencies:
    pip install lupa>=2.0
    (real provider backends, optional: httpx boto3)
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Callable

import host_store
import route_reliability as _route_reliability
from provider_adapters.anthropic import make_anthropic_async_call_provider
from provider_adapters.bedrock import make_bedrock_async_call_provider
from provider_adapters.common import (
    AsyncCallProviderHook,
    CallProviderHook,
    TokenProvider,
    _cached_tokens,
    _classify_status,
    _elapsed_ms,
    _err,
    _provider_error_message,
)
from provider_adapters.dispatcher import make_api_kind_dispatcher
from provider_adapters.google import make_google_async_call_provider
from provider_adapters.openai_compatible import (
    _PEER_GATES,
    _classify_from_map,
    _parse_openai_response,
    _prepare_openai_call,
    _resolve_auth_headers,
    make_async_call_provider,
    make_http_call_provider,
)

import lupa
from lupa import LuaRuntime

__all__ = [
    "LLMRouterHost",
    "FlowAdmissionError",
    "CallProviderHook",
    "AsyncCallProviderHook",
    "DiscoverHook",
    "Logger",
    "Clock",
    "TokenProvider",
    "make_api_kind_dispatcher",
    "make_async_call_provider",
    "make_http_call_provider",
    "make_anthropic_async_call_provider",
    "make_bedrock_async_call_provider",
    "make_google_async_call_provider",
    "_PEER_GATES",
    "_cached_tokens",
    "_classify_from_map",
    "_classify_status",
    "_elapsed_ms",
    "_err",
    "_parse_openai_response",
    "_prepare_openai_call",
    "_provider_error_message",
    "_resolve_auth_headers",
]

DiscoverHook = Callable[[str], dict]
Logger = Callable[[str, str, dict], None]
Clock = Callable[[], int]

_AUTH_UNCONFIGURED = "auth_unconfigured"
_SECRET_PLACEHOLDERS = {"", "CHANGE_ME", "TODO", "TODO_CHANGE_ME", "PLACEHOLDER"}


def _secret_configured(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().upper() not in _SECRET_PLACEHOLDERS


class FlowAdmissionError(ValueError):
    """A Σ_flow term rejected at admission (flow.check). The message is prefixed
    'flow: ' so the shim can classify it to 400 invalid_flow without importing
    this module — the flow twin of the core's 'ir: ' policy-admission errors."""


def _last_user_text(messages: list) -> str:
    """The flow's input: the last user message's text (string or content
    parts). The DAG's input node carries this verbatim."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c
                                if isinstance(p, dict) and p.get("type") == "text")
    return ""


class LLMRouterHost:
    def __init__(
        self,
        router_path: str | Path,
        config_path: str | Path,
        metrics_path: str | Path | None = None,
        *,
        call_provider: CallProviderHook | None = None,
        call_provider_async: AsyncCallProviderHook | None = None,
        discover: DiscoverHook | None = None,
        env: dict[str, str] | None = None,
        now_ms: Clock | None = None,
        logger: Logger | None = None,
        enforce_provider_auth: bool | None = None,
    ):
        self.lua = LuaRuntime(unpack_returned_tuples=True)

        self._custom_call_hook = call_provider is not None
        self._custom_async_call_hook = call_provider_async is not None
        self._enforce_provider_auth = enforce_provider_auth
        self._call_hook: CallProviderHook = call_provider or _default_mock_call
        self._async_call_hook: AsyncCallProviderHook | None = call_provider_async
        self._discover_hook: DiscoverHook | None = discover
        self._env: dict[str, str] = env if env is not None else dict(os.environ)
        self._now_ms: Clock = now_ms or (lambda: int(time.time() * 1000))
        self._logger: Logger = logger or _noop_logger
        self._mock_responses: dict[tuple[str, str], dict] = {}
        self.log_records: list[tuple[str, str, dict]] = []

        # Install host table BEFORE loading router (router.init logs to host.log).
        self._install_host_table()

        # The core is the `llm_policy` package; make it require-able from the
        # directory holding router.lua (the compat shim does require("llm_policy")).
        self._add_to_lua_path(Path(router_path).resolve().parent)
        self.router = self._dofile(Path(router_path))
        self.config = self._dofile(Path(config_path))
        self.metrics = self._dofile(Path(metrics_path)) if metrics_path else None
        self._inject_host_fields()

    def _inject_host_fields(self) -> None:
        """Declare host-universal observation fields that every catalog gets for
        free because they denote a HOST measurement, not catalog data — currently
        the per-session cache-affinity Bool `cache_hot`. Done here (once, after the
        config loads, before router.init and the flow schema read cfg.fields) so no
        catalog .lua repeats the getter and the field exists for example/live/any
        config alike.

        Zero engine change: this is the fields.lua extension seam
        (`schema{ extensions }`). The getter builds each candidate's route key
        through the SAME `route_reliability.route_key` that `_fold_route_outcome`
        uses — bridged into Lua as the `host_route_key` global, so the
        serialization (`provider|family|peer`, peer falling back to provider for
        peerless routes) has exactly one source and cannot drift across the
        Python/Lua boundary. It compares that key to the hot route the host
        resolved into `ctx.request.cache_hot_route` per request (see
        `host_store.hot_route`). The algebra observes only the Bool; the route
        key never enters the signature."""
        # One source of truth for the route-key serialization: the getter must
        # build a candidate's key identically to the fold, or affinity is silently
        # lost. Bridge the host's route_key into Lua instead of re-serializing.
        self.lua.globals().host_route_key = _route_reliability.route_key
        self.lua.eval("""
function(cfg)
    cfg.fields = cfg.fields or {}
    cfg.fields.cache_hot = {
        sort = "Bool", default = false, group = "route",
        get = function(c, ctx)
            local hot = ctx and ctx.request and ctx.request.cache_hot_route
            if hot == nil then return false end
            local pid, fam = c.provider_id, c.model_family
            if pid == nil or fam == nil then return false end
            local peer = (c.offer and c.offer.peer_id) or pid
            return host_route_key(pid, fam, peer) == hot
        end,
    }
end
""")(self.config)

    # ---- public API -----------------------------------------------------

    def init(self) -> None:
        ok, err = self.router.init(self.config, self.metrics)
        if not ok:
            raise RuntimeError(f"router.init failed: {err}")

    def info(self) -> dict:
        return _to_py(self.router.info())

    def rank(self, contract: dict) -> tuple[list[dict], list[dict]]:
        """Return (ranked_survivors, rejected). Raises on error."""
        if self._should_enforce_provider_auth():
            self._sync_provider_auth_state()
        ranked, err, rejected = self.router.rank(_to_lua(self.lua, contract))
        if err:
            raise RuntimeError(f"rank failed: {err}")
        return _to_py(ranked) or [], _to_py(rejected) or []

    def build_policy(self, spec: dict) -> dict:
        """Lower a declarative policy spec to a Σ_pol term via the core's
        elaborate — the vocabulary's ONE compiler; nothing is lowered
        host-side. Returns {term, fingerprint, version}. The term here is
        normalized but NOT admitted: admission (with the live field schema
        and the host envelope) happens where the term is USED — rank
        preview or execution — so there is exactly one admission point."""
        # (sigma-pol/v2) weighted scoring was removed; a profile carries its
        # ranking as a raw Σ_pol `scorer` term (field/normalize/neg/scale/add).
        # A stray `weights` key is dropped — there is nothing to lower it to.
        profile = {k: v for k, v in spec.items() if k != "weights"}
        elaborate = self.router.ir.elaborate
        term = elaborate.profile(
            _to_lua(self.lua, profile),
            _to_lua(self.lua, {"retry_table": spec.get("retry_table") or {}}),
        )
        nf = self.router.ir.term.normalize(term)
        return {
            "policy_ir": _to_py(nf),
            "fingerprint": self.router.ir.term.fingerprint(nf),
            "version": self.router.ir.VERSION,
        }

    def normalize_policy(self, policy_ir: list) -> dict:
        """Normalize a raw Σ_pol term and stamp its identity — the builder's
        download/identify step when the frontend composes the IR directly
        (rather than via the declarative elaborate surface). Like build_policy,
        the term is canonicalized but NOT admitted; admission happens where it
        is used (rank preview / execution)."""
        nf = self.router.ir.term.normalize(_to_lua(self.lua, policy_ir))
        return {
            "policy_ir": _to_py(nf),
            "fingerprint": self.router.ir.term.fingerprint(nf),
            "version": self.router.ir.VERSION,
        }

    # ---- Σ_flow: composition over Σ_pol ---------------------------------

    def _flow_module(self):
        """The core's llm_policy.flow, plus the host's field schema (so each
        node's embedded policy admits against config.fields, exactly as a
        per-call policy_ir does). Built once, lazily."""
        if getattr(self, "_flow_mod", None) is None:
            # parens truncate require's (module, loaderdata) to one value
            self._flow_mod = self.lua.eval('(require("llm_policy.flow"))')
            self._flow_schema = self.lua.eval(
                "function(cfg, ir) return ir.fields.schema{"
                " extensions = cfg.fields, tier_order = cfg.tier_order } end"
            )(self.config, self.router.ir)
        return self._flow_mod

    def flow_admit(self, flow_ir) -> dict:
        """Admit a Σ_flow term: check (graph validity + every node's policy),
        normalize, and stamp identity. Raises FlowAdmissionError on refusal —
        the shim maps it to 400 invalid_flow, the flow twin of invalid_policy.
        Admission is the core's job (one boundary), like policy_ir."""
        F = self._flow_module()
        lf = _to_lua(self.lua, flow_ir)
        # flow.check returns `true` (one value) or `nil, err` (two); lupa hands
        # back a bare value or a tuple accordingly.
        res = F.check(lf, self._flow_schema)
        ok = res[0] if isinstance(res, tuple) else res
        if not ok:
            err = res[1] if isinstance(res, tuple) and len(res) > 1 else "invalid flow"
            raise FlowAdmissionError("flow: " + str(err))
        nf = F.normalize(lf)
        return {
            "flow_ir": _to_py(nf),
            "encoded": F.encode(nf),            # host hashes this for identity
            "fingerprint": F.fingerprint(nf),
            "version": F.VERSION,
        }

    async def execute_flow_async(self, flow_ir, base_contract,
                                 call_override=None) -> dict:
        """Run a Σ_flow: admit, then schedule the DAG (flow_runner), running
        each llm node as a normal routed call — its policy + system prompt —
        through execute_async, so a node inherits the whole catalog / cascade /
        pricing / trace machinery. The return is shaped like a router result so
        the shim translates it identically, with a per-node trace under
        trace.flow_nodes (the flow-level twin of decision_trace)."""
        from flow_runner import run_flow

        admitted = self.flow_admit(flow_ir)
        fp = admitted["fingerprint"]
        input_text = _last_user_text(base_contract.get("messages") or [])
        carry = {k: base_contract[k] for k in
                 ("max_tokens", "tools", "tool_choice", "response_format",
                  "temperature", "seed", "session", "cache_hot_route")
                 if k in base_contract}

        async def run_node(nid, node, prompt):
            # Give the node the FULL conversation (system, history, tool results)
            # so a flow can act in an agent loop — not just the last user text.
            # The node's own system is appended as an extra system turn; the
            # assembled prompt (input passthrough, or the template'd drafts for a
            # synthesizer) is the final user turn. Cost: each node sees the whole
            # conversation, so an N-node flow is ~N× the input tokens.
            msgs = list(base_contract.get("messages") or [])
            if node.get("system"):
                msgs.append({"role": "system", "content": node["system"]})
            msgs.append({"role": "user", "content": prompt})
            try:
                res = await self.execute_async(
                    {**carry, "messages": msgs, "policy_ir": node["policy"]},
                    call_override=call_override)
            except Exception as exc:
                # A node's routed call must NEVER crash the whole flow: an
                # unhandled exception here bubbles past the shim and surfaces as a
                # gateway 502. Degrade to a clean node failure instead.
                return {"ok": False, "text": None, "tool_calls": None,
                        "error": f"node_exception: {type(exc).__name__}: {exc}",
                        "node_trace": {"node": nid, "error": str(exc)}}
            resp, chosen, tr = (res.get("response") or {},
                                res.get("chosen") or {}, res.get("trace") or {})
            return {
                "ok": bool(res.get("ok")),
                "text": resp.get("text"),
                # Proposals from a non-terminal node travel as data to the
                # synthesizer; the terminal node's are emitted to the caller.
                "tool_calls": resp.get("tool_calls"),
                "error": res.get("error") or tr.get("exhausted_reason"),
                "node_trace": {
                    "policy_fingerprint": tr.get("policy_fingerprint"),
                    "provider": chosen.get("provider_id"),
                    "served_model_id": chosen.get("served_model_id"),
                    "price_in": chosen.get("price_in"),
                    "price_out": chosen.get("price_out"),
                    "raw_price_in": chosen.get("raw_price_in"),
                    "raw_price_out": chosen.get("raw_price_out"),
                    "price_multiplier": chosen.get("price_multiplier"),
                    "tokens_in": resp.get("tokens_in"),
                    "tokens_out": resp.get("tokens_out"),
                    "tokens_cached": resp.get("tokens_cached"),
                    "cost_reported": resp.get("cost_reported"),
                    # this node's own latency, so the dashboard shows WHICH node is
                    # the slow one in a flow (e.g. a 12s antseed glm-5.2 node vs a
                    # 0.9s gpt-5.5 node).
                    "latency_ms": tr.get("total_latency_ms"),
                    "decision_trace": tr or None,
                },
            }

        fr = await run_flow(admitted["flow_ir"], input_text, run_node)
        nodes = fr.get("trace") or []
        tok_in = sum((n.get("tokens_in") or 0) for n in nodes) or None
        tok_out = sum((n.get("tokens_out") or 0) for n in nodes) or None
        tok_cached = sum((n.get("tokens_cached") or 0) for n in nodes) or None
        # The flow's synthetic chosen ("flow") has no price, so the shim can't
        # compute cost — aggregate it here, per node, the same way shim's
        # _executed_cost_usd does: prefer the provider-reported cost, else compute
        # from the node's ranked price discounting cache-read tokens (~10x). Sum
        # is surfaced as the flow's cost_reported so the shim uses it verbatim.
        def _node_cost(n):
            rep = n.get("cost_reported")
            if isinstance(rep, (int, float)) and not isinstance(rep, bool) and rep >= 0:
                return float(rep)
            pin = n.get("raw_price_in", n.get("price_in"))
            pout = n.get("raw_price_out", n.get("price_out"))
            if pin is None and pout is None:
                return None
            if n.get("raw_price_in") is None and n.get("raw_price_out") is None:
                mult = n.get("price_multiplier")
                if isinstance(mult, (int, float)) and not isinstance(mult, bool) and mult > 0:
                    pin = pin / mult if pin is not None else pin
                    pout = pout / mult if pout is not None else pout
            tin = n.get("tokens_in") or 0
            cached = n.get("tokens_cached") or 0
            uncached = max(0, tin - cached)
            return (uncached / 1e6 * (pin or 0)
                    + cached / 1e6 * (pin or 0) * 0.1
                    + (n.get("tokens_out") or 0) / 1e6 * (pout or 0))
        _costs = [c for c in (_node_cost(n) for n in nodes) if c is not None]
        flow_cost = round(sum(_costs), 6) if _costs else None
        base_trace = {"policy_fingerprint": None, "flow_fingerprint": fp,
                      "flow_nodes": nodes}
        if not fr.get("ok"):
            # Surface the failed node's REAL error kind (e.g. "no_candidates",
            # "exhausted: <kind>") so the OpenAI error translator maps it like the
            # single-model path does (503/4xx). The generic "flow_node_failed"
            # string is unknown to _openai_error_from_router and falls through to
            # its 502 catch-all — THE actual cause of the observed flow+tools 502
            # (a node with no tool-capable candidate fails clean with
            # no_candidates, but the wrapper hid it behind a 502). Keep the flow
            # context in the trace.
            return {"ok": False, "error": fr.get("error") or "flow_node_failed",
                    # carry chosen + the per-node trace on FAILURE too, so a failed
                    # flow is visible in Activity (provider:"flow" + which node
                    # failed) instead of an empty row — the shim emits this as the
                    # final chunk's x_router even on the error path.
                    "chosen": {"provider_id": "flow", "model_family": "flow:" + fp,
                               "served_model_id": "flow:" + fp},
                    "trace": {**base_trace, "failed_node": fr.get("failed_node"),
                              "flow_error": "flow_node_failed"}}
        final_tool_calls = fr.get("tool_calls")
        return {
            "ok": True,
            "response": {"text": fr.get("text") or "",
                         "tool_calls": final_tool_calls or None,
                         "finish_reason": "tool_calls" if final_tool_calls else "stop",
                         "tokens_in": tok_in, "tokens_out": tok_out,
                         "tokens_cached": tok_cached, "cost_reported": flow_cost},
            "chosen": {"provider_id": "flow", "model_family": "flow:" + fp,
                       "served_model_id": "flow:" + fp},
            "trace": base_trace,
        }

    # The core observation vocabulary (fields.lua CORE) + which builder group
    # each belongs to (model property vs provider/pair property). Stable spec.
    # KNOWN DEBT: this is a hand-copy of `core/llm_policy/fields.lua` Fl.CORE
    # (names + sorts). It only de-drifts the low-churn core list — the part that
    # grows append-only (config.fields) is already read live below. Reading
    # Fl.CORE from the core would kill the copy but couples the host to a core
    # internal and still needs a host-side name->group map (group is a builder/UI
    # concept absent from Fl.CORE). Revisit if core fields start changing.
    _CORE_FIELDS = [
        ("price_in", "Num", "provider"), ("price_out", "Num", "provider"),
        ("latency_ms", "Num", "provider"), ("tok_s", "Num", "provider"),
        ("success_rate", "Num", "provider"), ("credits", "Num", "provider"),
        # (sigma-pol/v2) quality/quality_hint removed — neither denoted anything
        # observable (static hand-written placeholders); score on real fields.
        ("context", "Num", "model"),
        ("has_tee", "Bool", "provider"), ("no_log", "Bool", "provider"),
        ("breaker_open", "Bool", "provider"), ("disabled", "Bool", "provider"),
    ]

    def field_schema(self) -> list:
        """The observable fields a policy may gate/score over: the core
        vocabulary plus this host's config.fields extensions, each tagged with
        its sort and builder group (model vs provider). Drives the data-driven
        builder dropdowns (GET /x/fields)."""
        out = [{"name": n, "sort": s, "group": g, "core": True}
               for (n, s, g) in self._CORE_FIELDS]
        extract = self.lua.eval(
            "function(cfg) local o = {} for k, v in pairs(cfg.fields or {}) do "
            "o[#o + 1] = { name = k, sort = v.sort, group = v.group or 'model' } "
            "end return o end")
        for d in (_to_py(extract(self.config)) or []):
            out.append({"name": d["name"], "sort": d["sort"],
                        "group": d["group"], "core": False})
        return out

    def model_meta(self) -> dict:
        """Per-family values of the config.fields model-group traits — the
        registered model-level facts (OpenRouter benchmarks/modalities/
        capabilities) the builder gates on, surfaced for the dashboard Market
        view. Reads each field's own getter (the same source the policy sees),
        so it stays correct for any config.fields, not just model_meta.lua.
        Provider/pair-level fields are skipped (they have no per-family value)."""
        families = list((self.catalog().get("models") or {}).keys())
        extract = self.lua.eval(
            "function(cfg, families) local out = {} "
            "for _, fam in ipairs(families) do local row = {} "
            "local c = { model_family = fam } "
            "for k, v in pairs(cfg.fields or {}) do "
            "if (v.group or 'model') == 'model' and type(v.get) == 'function' then "
            "local ok, val = pcall(v.get, c) "
            "if ok and val ~= nil then row[k] = val end end end "
            "out[fam] = row end return out end")
        return _to_py(extract(self.config, _to_lua(self.lua, families))) or {}

    def execute(self, contract: dict) -> dict:
        if self._should_enforce_provider_auth():
            self._sync_provider_auth_state()
        return _to_py(self.router.execute(_to_lua(self.lua, contract)))

    async def execute_async(self, contract: dict, call_override=None) -> dict:
        """Drive router.execute_step cooperatively, awaiting provider HTTP off
        the Lua lock so one LuaRuntime can overlap many in-flight requests.

        The Lua VM is touched only inside each (synchronous) execute_step call;
        all waiting happens at `await` points, where other coroutines are free
        to step. Since asyncio is single-threaded, concurrent requests never
        run Lua simultaneously, so shared RUNTIME state stays race-free.

        `call_override` replaces the configured call hook for THIS run only
        (the streaming path uses it to thread a per-request delta channel);
        mock responses still take precedence per (provider, family) pair.
        """
        if self._should_enforce_provider_auth():
            self._sync_provider_auth_state()
        # Session id (if the caller named one) rides host-side from here to the
        # fold so route_cache learns which peer served this conversation. It is a
        # local of this coroutine, so concurrent executes never share it.
        session = contract.get("session")
        step = self.router.execute_step(None, _to_lua(self.lua, contract), None)
        while True:
            status = step["status"]
            if status == "done":
                return _to_py(step["result"])

            handle = step["state_handle"]
            if status == "call":
                req = _to_py(step["request"]) or {}
                if (contract.get("first_token_timeout_ms") is not None
                        and req.get("first_token_timeout_ms") is None):
                    req["first_token_timeout_ms"] = contract["first_token_timeout_ms"]
                resp = await self._resolve_call_async(req, call_override, session=session)
                step = self.router.execute_step(handle, None, _to_lua(self.lua, resp))
            elif status == "wait":
                until_ms = step["until_ms"] or 0
                delay_s = max(0.0, (until_ms - self._now_ms()) / 1000.0)
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
                step = self.router.execute_step(handle, None, None)
            else:
                return {"ok": False, "error": f"internal: bad step status {status}", "trace": {}}

    async def _resolve_call_async(self, request: dict, call_override=None,
                                  session: "str | None" = None) -> dict:
        """Resolve one provider call for the async driver: mock first (so the
        same set_mock_response works for sync and async), then a per-run
        override, then the async hook, then the sync hook as a last resort."""
        key = (request.get("provider_id"), request.get("model_family"))
        if key in self._mock_responses:
            result = self._mock_responses[key]
        elif call_override is not None:
            result = await call_override(request)
        elif self._async_call_hook is not None:
            result = await self._async_call_hook(request)
        else:
            result = self._call_hook(request)
        # Record the outcome here (not in the hook) so the streaming/override path —
        # all of opencode's traffic, and every flow node — writes a route
        # observation too, the host-owned perf the algebra reads (derived) and the
        # market view surfaces (#15/#4a). Mocks record as well, so a mocked call
        # is measured exactly like a live one.
        _fold_route_outcome(request, result, session=session)
        return result

    def dump_state(self) -> dict:
        return _to_py(self.router.dump_state())

    def restore_state(self, snapshot: dict) -> None:
        """Restore breakers/disabled/EMA after a re-init (hot config change)."""
        self.router.restore_state(_to_lua(self.lua, snapshot or {}))

    def set_env(self, name: str, value: str) -> None:
        """Inject a credential into the host's env view at runtime (the env
        dict is captured at construction, so hot-added provider keys must be
        pushed here as well as into os.environ)."""
        self._env[str(name)] = str(value)

    def _provider_auth_configured(self, provider_id: str, provider: dict) -> bool:
        """Whether this host has the credential material a provider declares.

        This is intentionally a pre-routing admission check, not a provider
        health signal: a missing env var is deterministic local config, so the
        router should not waste a request discovering it as an auth_error.
        OAuth providers that do not expose an env-backed token are left to their
        adapter because their readiness is backend-specific and refreshable.
        """
        auth = provider.get("auth") if isinstance(provider.get("auth"), dict) else None
        kind = auth.get("kind") if auth else None
        env = provider.get("auth_env") or (auth.get("env") if auth else None)

        if provider_id == "ollama" and self._env.get("OLLAMA_CLOUD") != "1":
            # Local Ollama deliberately runs without an API key. The cloud path
            # sets OLLAMA_CLOUD=1 and is gated by OLLAMA_API_KEY as usual.
            return True

        if kind in (None, "bearer") and env:
            return _secret_configured(self._env.get(str(env)))
        if kind in (None, "none") and not env:
            return True
        if kind == "none":
            return True
        if kind == "bearer":
            return _secret_configured(self._env.get(str(env))) if env else False
        if kind == "oauth":
            # Codex and future OAuth adapters can refresh/read outside the host
            # env snapshot. Do not pre-disable them unless they declare an env.
            return _secret_configured(self._env.get(str(env))) if env else True
        return False

    def _should_enforce_provider_auth(self) -> bool:
        if self._enforce_provider_auth is not None:
            return bool(self._enforce_provider_auth)
        return self._custom_call_hook or self._custom_async_call_hook

    def _sync_provider_auth_state(self) -> None:
        """Reflect missing provider credentials into the core's disabled set.

        The core already knows how to filter disabled providers and how to skip
        them during execution. We mark only deterministic local config misses
        with a distinct kind, and remove only that kind when a key appears. Real
        runtime auth_error disables, breakers, and EMAs are preserved.
        """
        catalog = self.catalog()
        providers = catalog.get("providers") or {}
        if not providers:
            return

        missing = {
            pid for pid, provider in providers.items()
            if isinstance(provider, dict) and not self._provider_auth_configured(pid, provider)
        }
        state = self.dump_state() or {}
        disabled = dict(state.get("disabled_providers") or {})
        changed = False
        now = self._now_ms()

        for pid in missing:
            cur = disabled.get(pid)
            if cur is None:
                disabled[pid] = {"kind": _AUTH_UNCONFIGURED, "at_ms": now}
                changed = True
            elif isinstance(cur, dict) and cur.get("kind") == _AUTH_UNCONFIGURED:
                # Keep it fresh so the normal disabled-provider TTL cannot make
                # an unset env var intermittently routable.
                cur["at_ms"] = now
                changed = True

        for pid, cur in list(disabled.items()):
            if pid in missing:
                continue
            if isinstance(cur, dict) and cur.get("kind") == _AUTH_UNCONFIGURED:
                del disabled[pid]
                changed = True

        if changed:
            state["disabled_providers"] = disabled
            self.restore_state(state)

    def catalog(self) -> dict:
        """The loaded config as plain Python (providers/models/profiles)."""
        return _to_py(self.config) or {}

    def set_async_call_hook(self, hook: "AsyncCallProviderHook | None") -> None:
        """Replace the async provider-call hook (e.g. once provider rules
        derived from the loaded catalog are available)."""
        self._async_call_hook = hook
        self._custom_async_call_hook = hook is not None

    def update_metrics(self, provider: str, model: str, delta: dict) -> None:
        self.router.update_metrics(provider, model, _to_lua(self.lua, delta))

    def invalidate_discovery(self, discovery_id: str) -> None:
        self.router.invalidate_discovery(discovery_id)

    # ---- mock control (for tests) --------------------------------------

    def set_mock_response(self, provider: str, model: str, response: dict) -> None:
        self._mock_responses[(provider, model)] = response

    def clear_mocks(self) -> None:
        self._mock_responses.clear()

    def set_discover_hook(self, hook: DiscoverHook | None) -> None:
        self._discover_hook = hook

    # ---- internals -----------------------------------------------------

    def _dofile(self, path: Path):
        # Pass the path through a Lua global to avoid quoting bugs.
        self.lua.globals()["__path"] = str(path.resolve())
        return self.lua.eval("dofile(__path)")

    def _add_to_lua_path(self, directory: Path):
        # Prepend a directory to package.path so require() resolves modules
        # there (both `?.lua` and `?/init.lua` forms).
        self.lua.globals()["__dir"] = str(directory)
        self.lua.execute(
            'package.path = __dir.."/?.lua;"..__dir.."/?/init.lua;"..package.path'
        )
        self.lua.globals()["__dir"] = None

    def _install_host_table(self):
        self.lua.globals()["host"] = self.lua.table_from({
            "now_ms":        self._h_now_ms,
            "log":           self._h_log,
            "env":           self._h_env,
            "call_provider": self._h_call_provider,
            "discover":      self._h_discover,
            "price_multiplier": self._h_price_multiplier,
            "sleep_ms":      self._h_sleep_ms,
        })

    def _h_now_ms(self) -> int:
        return self._now_ms()

    def _h_log(self, level, event, fields):
        py_fields = _to_py(fields) or {}
        self.log_records.append((level, event, py_fields))
        self._logger(level, event, py_fields)

    def _h_env(self, key):
        return self._env.get(key)

    def _h_call_provider(self, request):
        py_req = _to_py(request) or {}
        provider = py_req.get("provider_id")
        model = py_req.get("model_family")
        if (provider, model) in self._mock_responses:
            resp = self._mock_responses[(provider, model)]
        else:
            resp = self._call_hook(py_req)
        return _to_lua(self.lua, resp)

    def _h_discover(self, discovery_id):
        if not self._discover_hook:
            return _to_lua(self.lua, {"ok": False, "error": "no_discover_hook"})
        return _to_lua(self.lua, self._discover_hook(discovery_id))

    def _h_price_multiplier(self, provider_id, source_name=None) -> float:
        import settings

        for name in (provider_id, source_name):
            if not name:
                continue
            key = f"{name}.price_multiplier"
            if key in settings.SCHEMA:
                return float(settings.get(key))
        return 1.0

    def _h_sleep_ms(self, ms):
        time.sleep(float(ms) / 1000.0)


# ---- marshaling helpers -------------------------------------------------

def _to_py(obj):
    """Recursively convert lupa Lua tables to Python dicts/lists."""
    if obj is None:
        return None
    t = lupa.lua_type(obj)
    if t is None:
        return obj
    if t != "table":
        return obj
    keys = list(obj.keys())
    if keys and all(isinstance(k, int) for k in keys) \
            and set(keys) == set(range(1, len(keys) + 1)):
        return [_to_py(obj[i]) for i in range(1, len(keys) + 1)]
    return {k: _to_py(v) for k, v in obj.items()}


def _to_lua(lua: LuaRuntime, obj):
    if isinstance(obj, dict):
        return lua.table_from({k: _to_lua(lua, v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return lua.table_from([_to_lua(lua, x) for x in obj])
    return obj


def _default_mock_call(request: dict) -> dict:
    return {
        "ok": False,
        "error_kind": "no_mock_set",
        "http_status": 0,
        "latency_ms": 0,
    }


def _noop_logger(level, event, fields):
    pass


def _fold_route_outcome(request: dict, result: dict,
                        session: "str | None" = None) -> None:
    """Fold ONE call outcome into the host-side per-route measurements:
    reliability (success EMA), latency (EMA), and learned tool capability. Called
    from _resolve_call_async for BOTH the direct hook and the streaming/override
    path — the streaming result carries `ok` + `latency_ms` too — so opencode's
    streamed traffic and every Σ_flow node feed the same route_observations the
    algebra reads derived (offer.success_rate / offer.latency_ms). Previously this
    lived inside the direct hook only, so reliability/latency stayed empty for the
    streaming traffic that is, in practice, all of it."""
    # The route's identity is at the request TOP LEVEL (provider_id / model_family
    # / peer_id) — the core stamps it there for every call. The per-call `offer`
    # dict is only attached by some sources (antseed marketplace) and is None for
    # openrouter / static / partner routes. Reading family from `offer` alone meant
    # those routes NEVER folded, so latency_ms / success_rate stayed empty for them
    # and a policy could not rank them by speed. Read top level, fall back to offer.
    offer = request.get("offer") or {}
    fam = request.get("model_family") or offer.get("model_family")
    pid = request.get("provider_id")
    if not fam or not pid:
        return
    peer_id = request.get("peer_id") or offer.get("peer_id")
    ok = bool(result.get("ok"))
    # One route identity for reliability, latency and the call count: the peer for
    # marketplace routes, or the provider itself for partner/gateway routes (no
    # peer_id), so OpenRouter/OpenAI is comparable to a peer's. The engine no
    # longer folds reliability for ANY route (#15), so the host folds it for all
    # of them — not just marketplace — and the market perf view reads it back.
    rkey = _route_reliability.route_key(pid, fam, peer_id or pid)
    # #4a/#4c: record the per-ATTEMPT raw observation (every provider call,
    # including failed fallbacks — a grain `calls` lacks). reliability, latency AND
    # learned tool capability are derived on the fly from route_observations, not
    # folded in-process. served_by = the peer for marketplace routes, else the
    # provider — the identity route_stats / tool_incapable_routes key on. The tool
    # signals carry only on tools-requests (capability is a marketplace concern).
    host_store.observe_route_call_async({
        "ts": int(time.time() * 1000), "provider_id": pid, "model_family": fam,
        "served_by": peer_id or pid, "ok": ok, "latency_ms": result.get("latency_ms"),
        "tools_requested": bool(request.get("tools")),
        "tool_calls_emitted": bool((result.get("response") or {}).get("tool_calls"))})
    # Cache affinity + the per-session meter are DERIVED on the fly from `calls`
    # now (#4b): the route_observation above + the ingress's call ledger carry
    # everything (session, route, outcome, tokens, cost), so no per-session fold.
