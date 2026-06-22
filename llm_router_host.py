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
    (real-HTTP backend, optional: httpx)
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import route_reliability as _route_reliability
import route_latency as _route_latency
import route_tool_capability as _route_tool_capability

import lupa
from lupa import LuaRuntime

CallProviderHook = Callable[[dict], dict]
AsyncCallProviderHook = Callable[[dict], Awaitable[dict]]
DiscoverHook = Callable[[str], dict]
Logger = Callable[[str, str, dict], None]
Clock = Callable[[], int]


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
    ):
        self.lua = LuaRuntime(unpack_returned_tuples=True)

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

    # ---- public API -----------------------------------------------------

    def init(self) -> None:
        ok, err = self.router.init(self.config, self.metrics)
        if not ok:
            raise RuntimeError(f"router.init failed: {err}")

    def info(self) -> dict:
        return _to_py(self.router.info())

    def rank(self, contract: dict) -> tuple[list[dict], list[dict]]:
        """Return (ranked_survivors, rejected). Raises on error."""
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
                  "temperature", "seed") if k in base_contract}

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
                    "tokens_in": resp.get("tokens_in"),
                    "tokens_out": resp.get("tokens_out"),
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
                         "tokens_in": tok_in, "tokens_out": tok_out},
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
        step = self.router.execute_step(None, _to_lua(self.lua, contract), None)
        while True:
            status = step["status"]
            if status == "done":
                return _to_py(step["result"])

            handle = step["state_handle"]
            if status == "call":
                req = _to_py(step["request"]) or {}
                resp = await self._resolve_call_async(req, call_override)
                step = self.router.execute_step(handle, None, _to_lua(self.lua, resp))
            elif status == "wait":
                until_ms = step["until_ms"] or 0
                delay_s = max(0.0, (until_ms - self._now_ms()) / 1000.0)
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
                step = self.router.execute_step(handle, None, None)
            else:
                return {"ok": False, "error": f"internal: bad step status {status}", "trace": {}}

    async def _resolve_call_async(self, request: dict, call_override=None) -> dict:
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
        # Fold the outcome here (not in the hook) so the streaming/override path —
        # all of opencode's traffic, and every flow node — feeds route_latency /
        # reliability / the call count too, the host-owned perf the algebra reads
        # and the market view surfaces (#15). Mocks fold as well, so a mocked call
        # is measured exactly like a live one.
        _fold_route_outcome(request, result)
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

    def catalog(self) -> dict:
        """The loaded config as plain Python (providers/models/profiles)."""
        return _to_py(self.config) or {}

    def set_async_call_hook(self, hook: "AsyncCallProviderHook | None") -> None:
        """Replace the async provider-call hook (e.g. once provider rules
        derived from the loaded catalog are available)."""
        self._async_call_hook = hook

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


# ---- credential resolution + request prep (shared by sync/async HTTP) ---

TokenProvider = Callable[[], "str | None"]


def _resolve_auth_headers(
    request: dict,
    env_get: Callable[[str], str | None],
    token_providers: dict[str, TokenProvider] | None = None,
) -> tuple[dict | None, dict | None]:
    """Map a provider's auth descriptor to request headers.

    Supports `auth.kind` in {"none", "bearer", "oauth"}. For back-compat, a
    bare `auth_env` (no `auth` block) is treated as bearer. Returns
    (headers, error): on success `error` is None; on failure `headers` is None
    and `error` is a router error dict.
    """
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
    """Resolve auth headers for Ollama Cloud via OLLAMA_API_KEY.

    Args:
        env_get: Function to read environment variables
        url: Full URL for the request (unused; kept for call-site symmetry)
        method: HTTP method (unused; kept for call-site symmetry)
        body: Request body (unused; kept for call-site symmetry)

    Returns {"Authorization": "Bearer <key>"} if OLLAMA_API_KEY is set,
    else None (caller maps None to an auth_error).
    """
    api_key = env_get("OLLAMA_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return None


def _prepare_openai_call(
    request: dict,
    env_get: Callable[[str], str | None],
    extra: dict[str, str],
    timeout_s: float,
    token_providers: dict[str, TokenProvider] | None = None,
) -> tuple[tuple | None, dict | None]:
    """Build (url, body, headers, timeout_s) for an OpenAI-compatible call, or
    return (None, error). Shared by the sync and async HTTP backends."""
    auth_headers, err = _resolve_auth_headers(request, env_get, token_providers)
    if err is not None:
        return None, err

    offer = request.get("offer") or {}

    body: dict = {
        # marketplace candidates may serve a curated family under a different
        # wire name (service aliasing) — the offer's wire id wins.
        "model":    offer.get("wire_model_id") or request["served_model_id"],
        "messages": request.get("messages") or [],
    }
    for field in ("tools", "response_format", "temperature", "seed", "max_tokens"):
        v = request.get(field)
        if v is not None:
            body[field] = v

    url = (request.get("base_url") or "").rstrip("/") + "/chat/completions"

    # Ollama: override auth based on endpoint
    # Local endpoints (localhost, 127.0.0.1) require no auth
    # Cloud endpoints (ollama.com) require auth (OLLAMA_API_KEY Bearer)
    base_url = request.get("base_url") or ""
    provider_id = request.get("provider_id") or ""
    seller_endpoint = offer.get("seller_endpoint") or ""

    # Detect Ollama by provider_id, base_url, or seller_endpoint
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
            # Cloud: API key Bearer (OLLAMA_API_KEY)
            auth_headers = _resolve_ollama_cloud_auth(
                env_get, url, method="POST", body=b""
            )
            if auth_headers is None:
                return None, _err("auth_error", 0, 0,
                    "Ollama Cloud requires OLLAMA_API_KEY")
        else:
            # Local: no auth required
            auth_headers = {}

    headers = {"Content-Type": "application/json", **auth_headers, **extra}
    # Marketplace (AntSeed): the buyer proxy runs in browse mode and disables
    # auto-selection, so the host pins the offer's peer per request. The peer is
    # the one the policy actually priced/selected — keeping peer choice in Σ_pol
    # (deterministic) rather than an opaque buyer-side router.
    peer_id = offer.get("peer_id")
    if peer_id:
        headers["x-antseed-pin-peer"] = peer_id
    timeout = (request.get("timeout_ms") or int(timeout_s * 1000)) / 1000.0
    return (url, body, headers, timeout), None


def make_api_kind_dispatcher(
    default: AsyncCallProviderHook,
    handlers: dict[str, AsyncCallProviderHook] | None = None,
) -> AsyncCallProviderHook:
    """Route each call to a per-api_kind async handler, falling back to
    `default` (the OpenAI-compatible backend). Lets one host serve providers
    with different wire protocols (e.g. openai_codex) behind one router."""
    _handlers = dict(handlers or {})

    async def call(request: dict) -> dict:
        handler = _handlers.get(request.get("api_kind"), default)
        return await handler(request)

    return call


# ---- HTTP-real call_provider (OpenAI-compatible) ------------------------

def make_http_call_provider(
    env_get: Callable[[str], str | None] | None = None,
    timeout_s: float = 30.0,
    extra_headers: dict[str, str] | None = None,
    token_providers: dict[str, TokenProvider] | None = None,
    provider_rules: dict[str, dict] | None = None,
) -> CallProviderHook:
    """
    Return a call_provider that translates the router's request to an
    OpenAI-compatible /chat/completions POST, classifies the HTTP outcome
    to a canonical error_kind, and returns the shape router.lua expects.

    Auth is resolved from the provider's `auth` descriptor (kind none/bearer/
    oauth); a bare `auth_env` is treated as bearer. `env_get` reads bearer
    tokens (default `os.environ.get`); `token_providers` maps an oauth
    provider name to a token getter.

    Requires `httpx` (pip install httpx).
    """
    import time as _time
    import httpx

    _env_get = env_get or os.environ.get
    _extra = dict(extra_headers or {})

    def call(request: dict) -> dict:
        api_kind = request.get("api_kind", "openai_compatible")
        if api_kind != "openai_compatible":
            return _err("unsupported_api_kind", 0, 0,
                        f"api_kind={api_kind!r} not supported by HTTP backend")

        prep, err = _prepare_openai_call(request, _env_get, _extra, timeout_s, token_providers)
        if err is not None:
            return err
        url, body, headers, timeout = prep

        t0 = _time.monotonic()
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
        except httpx.TimeoutException:
            return _err("timeout", 0, _elapsed_ms(t0), f"POST {url} timed out")
        except (httpx.NetworkError, httpx.RequestError) as e:
            return _err("network_error", 0, _elapsed_ms(t0), str(e))

        rules = (provider_rules or {}).get(request.get("provider_id")) or {}
        return _parse_openai_response(resp, _elapsed_ms(t0), error_map=rules.get("error_map"))

    return call


# Per-peer in-flight gate for marketplace sellers. Each AntSeed peer advertises
# `maxConcurrency`; exceeding it earns a 429 "Max concurrency reached" that the
# router would (wastefully) treat as a failure and fall away from antseed. We cap
# in-flight calls per peer to its advertised limit — a bounded wait, then yield to
# the next candidate — so the router never trips the seller's own limiter.
_PEER_GATES: dict[str, asyncio.Semaphore] = {}


def _peer_gate(peer_id: str, cap: int) -> asyncio.Semaphore:
    """The per-peer semaphore (create-once; first advertised cap wins — caps are
    stable per seller and resizing a live semaphore races held slots)."""
    sem = _PEER_GATES.get(peer_id)
    if sem is None:
        sem = asyncio.Semaphore(cap)
        _PEER_GATES[peer_id] = sem
    return sem


def _fold_route_outcome(request: dict, result: dict) -> None:
    """Fold ONE call outcome into the host-side per-route measurements:
    reliability (success EMA), latency (EMA), and learned tool capability. Called
    from _resolve_call_async for BOTH the direct hook and the streaming/override
    path — the streaming result carries `ok` + `latency_ms` too — so opencode's
    streamed traffic and every Σ_flow node feed the same EMAs the algebra reads
    (offer.success_rate / offer.latency_ms). Previously this lived inside the
    direct hook only, so route_latency/reliability stayed empty for the streaming
    traffic that is, in practice, all of it."""
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
    _route_reliability.observe(rkey, ok)
    _route_latency.observe(rkey, result.get("latency_ms"), ok)
    # Learned tool capability is a marketplace concern only (static/partner routes
    # declare their capabilities in config), so keep it peer-scoped.
    if peer_id:
        _route_tool_capability.observe(
            rkey, bool(request.get("tools")),
            bool((result.get("response") or {}).get("tool_calls")))


def make_async_call_provider(
    env_get: Callable[[str], str | None] | None = None,
    timeout_s: float = 30.0,
    extra_headers: dict[str, str] | None = None,
    client: "Any" = None,
    token_providers: dict[str, TokenProvider] | None = None,
    provider_rules: dict[str, dict] | None = None,
) -> AsyncCallProviderHook:
    """Async twin of make_http_call_provider: same request translation, auth
    resolution and error classification, but non-blocking (httpx.AsyncClient)
    so the async shim can overlap many upstream calls on one event loop.

    Pass a shared `httpx.AsyncClient` to reuse connections; otherwise one is
    created per call. Requires `httpx`.
    """
    import httpx

    _env_get = env_get or os.environ.get
    _extra = dict(extra_headers or {})

    async def call(request: dict) -> dict:
        api_kind = request.get("api_kind", "openai_compatible")
        if api_kind != "openai_compatible":
            return _err("unsupported_api_kind", 0, 0,
                        f"api_kind={api_kind!r} not supported by HTTP backend")

        prep, err = _prepare_openai_call(request, _env_get, _extra, timeout_s, token_providers)
        if err is not None:
            return err
        url, body, headers, timeout = prep

        t0 = time.monotonic()

        # Per-seller concurrency gate (marketplace offers carry peer_id +
        # max_concurrency). Wait up to the call's own timeout for a slot; if the
        # peer stays saturated, yield to the next candidate as a rate_limit
        # rather than forcing the seller's 429.
        offer = request.get("offer") or {}
        peer_id = offer.get("peer_id")
        cap = offer.get("max_concurrency")
        gate = _peer_gate(peer_id, cap) if (peer_id and isinstance(cap, int) and cap > 0) else None
        if gate is not None:
            try:
                await asyncio.wait_for(gate.acquire(), timeout=timeout)
            except (asyncio.TimeoutError, TimeoutError):
                # our own backpressure, not the seller failing -> don't fold it
                # into the route's reliability.
                return _err("rate_limit", 0, _elapsed_ms(t0),
                            f"antseed peer {peer_id[:10]} in-flight cap {cap} saturated")
        try:
            try:
                if client is not None:
                    resp = await client.post(url, json=body, headers=headers, timeout=timeout)
                else:
                    async with httpx.AsyncClient() as c:
                        resp = await c.post(url, json=body, headers=headers, timeout=timeout)
            except httpx.TimeoutException:
                result = _err("timeout", 0, _elapsed_ms(t0), f"POST {url} timed out")
            except (httpx.NetworkError, httpx.RequestError) as e:
                result = _err("network_error", 0, _elapsed_ms(t0), str(e))
            else:
                rules = (provider_rules or {}).get(request.get("provider_id")) or {}
                result = _parse_openai_response(resp, _elapsed_ms(t0), error_map=rules.get("error_map"))
            # Per-route outcome (reliability / latency / learned tool capability)
            # is folded centrally in _resolve_call_async, so the streaming/override
            # path feeds the SAME EMAs as this direct hook — not just non-streaming
            # calls. (Was here; moved up so streamed traffic and flow nodes count.)
            return result
        finally:
            if gate is not None:
                gate.release()

    return call


def _classify_from_map(err_msg: str, error_map: dict | None) -> str | None:
    """Provider-declared body-substring -> canonical kind. First hit wins,
    case-insensitive, checked before the status-code fallback."""
    if not error_map:
        return None
    msg = (err_msg or "").lower()
    for needle, kind in error_map.items():
        if needle.lower() in msg:
            return str(kind)
    return None


def _parse_openai_response(resp: "Any", latency: int, error_map: dict | None = None) -> dict:
    """Translate an OpenAI-compatible HTTP response into the router's response
    shape. Shared by the sync and async HTTP backends."""
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
            return _err("content_filter", status, latency, "blocked by provider filter")

        msg   = choice.get("message") or {}
        usage = data.get("usage") or {}
        text = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")
        if not str(text).strip() and not tool_calls:
            return _err("bad_response", status, latency, "empty assistant content")
        return {
            "ok":         True,
            "latency_ms": latency,
            "response": {
                "text":          text,
                "tool_calls":    tool_calls,
                "finish_reason": finish,
                "tokens_in":     usage.get("prompt_tokens"),
                "tokens_out":    usage.get("completion_tokens"),
                "tokens_total":  usage.get("total_tokens"),
                "raw_model":     data.get("model"),
            },
        }

    try:
        err_body = resp.json()
        err_msg  = str(err_body)
    except Exception:
        err_msg = (resp.text or "")[:500]
    kind = _classify_from_map(err_msg, error_map) or _classify_status(status, err_msg)
    return _err(kind, status, latency, err_msg[:500])


def _classify_status(status: int, err_msg: str) -> str:
    if status in (401, 403):
        return "auth_error"
    if status == 429:
        return "rate_limit"
    if status == 402:
        return "payment_required"
    if status in (408, 504):
        return "timeout"
    if status == 404:
        return "model_unavailable"
    if status == 400:
        m = (err_msg or "").lower()
        if "context" in m or "token" in m or "length" in m or "maximum" in m:
            return "context_overflow"
        return "bad_request"
    if 500 <= status < 600:
        return "server_error"
    return "unknown"


def _err(kind: str, status: int, latency_ms: int, message: str) -> dict:
    return {
        "ok":            False,
        "error_kind":    kind,
        "http_status":   status,
        "latency_ms":    latency_ms,
        "error_message": message,
    }


def _elapsed_ms(t0: float) -> int:
    import time as _t
    return int((_t.monotonic() - t0) * 1000)
