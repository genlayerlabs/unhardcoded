"""
Σ_flow on the host: the pure scheduler (flow_runner) and the end-to-end wiring
through the shim — a flow_ir runs each node as a routed call and returns the
sink node's answer, with a per-node trace.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import flow_runner  # noqa: E402
from llm_router_host import LLMRouterHost, make_api_kind_dispatcher  # noqa: E402
from shim import create_app  # noqa: E402


def POLICY():
    return ["policy", ["meets_req"], ["field", "context"], ["argmax"],
            ["id"], ["always", {"action": "next_candidate"}]]


def MOA():
    """mixture-of-agents: u -> {a, b} -> f -> out."""
    return ["flow", {
        "u":   {"kind": "input"},
        "a":   {"kind": "llm", "system": "Answer concisely.",
                "policy": POLICY(), "inputs": ["u"]},
        "b":   {"kind": "llm", "system": "Answer rigorously.",
                "policy": POLICY(), "inputs": ["u"]},
        "f":   {"kind": "llm", "system": "Synthesize the best answer.",
                "policy": POLICY(), "inputs": ["a", "b"],
                "template": "Draft A:\n$1\n\nDraft B:\n$2"},
        "out": {"kind": "output", "inputs": ["f"]},
    }]


# ---- the pure scheduler -------------------------------------------------

def test_topo_order_and_template_assembly():
    nodes = MOA()[1]
    order = flow_runner.topo_order(nodes)
    assert order[0] == "u" and order[-1] == "out"
    assert order.index("a") < order.index("f")
    assert order.index("b") < order.index("f")

    node = nodes["f"]
    prompt = flow_runner.assemble(node, [{"id": "a", "text": "42"},
                                         {"id": "b", "text": "forty-two"}])
    assert prompt == "Draft A:\n42\n\nDraft B:\nforty-two"


def test_run_flow_threads_outputs_to_the_sink():
    async def run_node(nid, node, prompt):
        if node["system"].startswith("Synthesize"):
            return {"ok": True, "text": "FINAL(" + prompt.replace("\n", " ") + ")"}
        return {"ok": True, "text": node["system"][:6] + ":" + prompt}

    out = asyncio.run(flow_runner.run_flow(MOA(), "Q?", run_node))
    assert out["ok"]
    assert out["text"].startswith("FINAL(")
    assert "Draft A:" in out["text"] and "Draft B:" in out["text"]
    assert len(out["trace"]) == 3


def test_run_flow_short_circuits_on_node_failure():
    async def run_node(nid, node, prompt):
        return {"ok": False, "error": "exhausted"} if nid == "a" else {"ok": True, "text": "x"}

    out = asyncio.run(flow_runner.run_flow(MOA(), "Q?", run_node))
    assert not out["ok"] and out["failed_node"] == "a"


# ---- end-to-end through the shim ---------------------------------------

def _client():
    async def default(req):
        # echo the provider + the last user message so we can see assembly
        text = ""
        for m in reversed(req.get("messages") or []):
            if m.get("role") == "user":
                text = m.get("content") or ""
                break
        return {"ok": True, "latency_ms": 1,
                "response": {"text": f"[{req['provider_id']}] {text}",
                             "finish_reason": "stop", "tokens_in": 3, "tokens_out": 5}}

    async def codex(req):
        return {"ok": False, "error_kind": "server_error"}

    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        call_provider_async=make_api_kind_dispatcher(
            default=default, handlers={"openai_codex": codex}),
        now_ms=lambda: 1,
    )
    host.init()
    return TestClient(create_app(host, default_profile="default"))


def test_flow_ir_runs_the_dag_and_returns_the_sink_answer():
    r = _client().post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(),
        "messages": [{"role": "user", "content": "What is 17*23?"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    # the sink is the synthesizer, which saw both drafts assembled by its template
    assert "Draft A:" in content and "Draft B:" in content
    xr = body["x_router"]
    assert xr["provider"] == "flow"
    assert xr["model_family"].startswith("flow:")
    nodes = xr["decision_trace"]["flow_nodes"]
    assert len(nodes) == 3
    assert {n["node"] for n in nodes} == {"n1", "n2", "n3"}  # canonical ids; n0 is input
    assert all(n["provider"] for n in nodes), "each node recorded its routed provider"


def test_malformed_flow_is_rejected_400():
    bad = MOA()
    bad[1]["a"]["inputs"] = ["u", "f"]   # cycle a <-> f
    r = _client().post("/v1/chat/completions", json={
        "model": "", "flow_ir": bad,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_flow"


# ---- tool calls through a flow ------------------------------------------

def _tc(name, args):
    return {"id": "call_" + name, "type": "function",
            "function": {"name": name, "arguments": args}}


def test_run_flow_propagates_terminal_tool_calls_and_serializes_upstream():
    """Upstream nodes' PROPOSED tool calls reach the synthesizer as data in its
    prompt; the terminal node's tool calls are what the flow returns."""
    seen = {}

    async def run_node(nid, node, prompt):
        if node["system"].startswith("Synthesize"):
            seen["prompt"] = prompt
            return {"ok": True, "text": None, "tool_calls": [_tc("edit", '{"file":"main.py"}')]}
        # a and b each PROPOSE a (different) tool call, with some text too
        return {"ok": True, "text": "draft " + nid,
                "tool_calls": [_tc("read_file", '{"path":"' + nid + '"}')]}

    out = asyncio.run(flow_runner.run_flow(MOA(), "Q?", run_node))
    assert out["ok"]
    # the terminal (synthesizer) tool call is the one the flow emits
    assert out["tool_calls"] and out["tool_calls"][0]["function"]["name"] == "edit"
    # the upstream proposals were serialized into the synthesizer's prompt
    assert "proposed tool calls" in seen["prompt"]
    assert seen["prompt"].count("read_file") == 2  # both a and b proposed one


def _client_with_backend(backend):
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        call_provider_async=make_api_kind_dispatcher(default=backend),
        now_ms=lambda: 1,
    )
    host.init()
    return TestClient(create_app(host, default_profile="default"),
                      raise_server_exceptions=False)


def test_flow_with_tools_returns_terminal_tool_calls_end_to_end():
    async def backend(req):
        sys_txt = " ".join(m.get("content") or "" for m in (req.get("messages") or [])
                           if m.get("role") == "system")
        if "Synthesize" in sys_txt:
            return {"ok": True, "latency_ms": 1,
                    "response": {"text": None, "tool_calls": [_tc("do_it", '{"x":1}')],
                                 "finish_reason": "tool_calls", "tokens_in": 3, "tokens_out": 5}}
        return {"ok": True, "latency_ms": 1,
                "response": {"text": "draft", "finish_reason": "stop",
                             "tokens_in": 3, "tokens_out": 5}}

    r = _client_with_backend(backend).post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(),
        "tools": [{"type": "function", "function": {"name": "do_it",
                   "parameters": {"type": "object"}}}],
        "messages": [{"role": "user", "content": "go"}],
    })
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "do_it"


def test_flow_node_exception_is_a_clean_error_not_a_502():
    async def boom(req):
        raise RuntimeError("provider blew up")

    r = _client_with_backend(boom).post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(),
        "messages": [{"role": "user", "content": "hi"}],
    })
    # the node exception must be contained as a clean router error, never an
    # unhandled crash (which a gateway shows as 502).
    assert r.status_code != 500, r.text
    assert "error" in r.json()


def test_flow_node_no_candidates_surfaces_503_not_502():
    """THE actual cause of the observed flow+tools 502: a node that fails with a
    clean no_candidates (e.g. its family has no tool-capable route) must surface
    its real error kind (503), not the generic flow_node_failed → 502 catch-all."""
    impossible = ["policy", ["and", ["meets_req"], ["cmp", "price_in", "lt", -1]],
                  ["field", "context"], ["argmax"], ["id"],
                  ["always", {"action": "next_candidate"}]]
    flow = ["flow", {
        "u": {"kind": "input"},
        "a": {"kind": "llm", "system": "x", "policy": impossible, "inputs": ["u"]},
        "out": {"kind": "output", "inputs": ["a"]},
    }]
    r = _client().post("/v1/chat/completions", json={
        "model": "", "flow_ir": flow,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "no_candidates"


# ---- flow + streaming ---------------------------------------------------

def test_flow_with_stream_pseudo_streams_sse():
    """A streaming client (stream=true) gets the finished flow result as SSE,
    not a JSON body it can't read as a stream (the cause of the 60s/0-token
    hangs an agent SDK hit)."""
    r = _client().post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(), "stream": True,
        "messages": [{"role": "user", "content": "What is 17*23?"}],
    })
    assert r.status_code == 200, r.text
    assert "text/event-stream" in r.headers.get("content-type", "")
    body = r.text
    assert "data:" in body and "[DONE]" in body
    assert "chat.completion.chunk" in body
    assert "Draft A:" in body  # the sink synthesizer's assembled text streamed


def test_flow_with_tools_streams_terminal_tool_calls():
    async def backend(req):
        sys_txt = " ".join(m.get("content") or "" for m in (req.get("messages") or [])
                           if m.get("role") == "system")
        if "Synthesize" in sys_txt:
            return {"ok": True, "latency_ms": 1,
                    "response": {"text": None, "tool_calls": [_tc("do_it", '{"x":1}')],
                                 "finish_reason": "tool_calls", "tokens_in": 3, "tokens_out": 5}}
        return {"ok": True, "latency_ms": 1,
                "response": {"text": "draft", "finish_reason": "stop",
                             "tokens_in": 3, "tokens_out": 5}}

    r = _client_with_backend(backend).post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(), "stream": True,
        "tools": [{"type": "function", "function": {"name": "do_it",
                   "parameters": {"type": "object"}}}],
        "messages": [{"role": "user", "content": "go"}],
    })
    assert r.status_code == 200, r.text
    assert "text/event-stream" in r.headers.get("content-type", "")
    body = r.text
    assert "tool_calls" in body and "do_it" in body  # terminal tool_call streamed
    assert "[DONE]" in body


# ---- flow sees the full conversation (agent context), and node errors map ----

def test_flow_node_sees_full_conversation_context():
    """Each node must receive the whole conversation (system, history, tool
    results), not just the last user text — else a flow can't act in an agent
    loop. Here a secret lives ONLY in a system message; the node must see it."""
    async def backend(req):
        seen = " ".join(str(m.get("content") or "") for m in (req.get("messages") or []))
        return {"ok": True, "latency_ms": 1,
                "response": {"text": "HAS_SECRET" if "ZEBRA-9981" in seen else "NO_SECRET",
                             "finish_reason": "stop", "tokens_in": 3, "tokens_out": 5}}

    r = _client_with_backend(backend).post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(),
        "messages": [
            {"role": "system", "content": "Context: the code is ZEBRA-9981."},
            {"role": "user", "content": "What is the code?"}],
    })
    assert r.status_code == 200, r.text
    assert "HAS_SECRET" in r.json()["choices"][0]["message"]["content"]


def test_flow_node_timeout_maps_to_504_not_502():
    """A flow node whose seller times out must surface a clean 504, not the
    generic 502 catch-all (the observed flow 502 was a node timeout)."""
    async def backend(req):
        sys_txt = " ".join(m.get("content") or "" for m in (req.get("messages") or [])
                           if m.get("role") == "system")
        if "Synthesize" in sys_txt:
            return {"ok": False, "error_kind": "timeout", "latency_ms": 1}
        return {"ok": True, "latency_ms": 1,
                "response": {"text": "draft", "finish_reason": "stop",
                             "tokens_in": 3, "tokens_out": 5}}

    r = _client_with_backend(backend).post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(),
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 504, r.text


def test_flow_with_multi_turn_conversation_succeeds():
    """A flow over a multi-turn conversation (assistant + tool messages) must
    run cleanly — passing the full history to each node must not break it."""
    async def backend(req):
        return {"ok": True, "latency_ms": 1,
                "response": {"text": "ok", "finish_reason": "stop",
                             "tokens_in": 3, "tokens_out": 5}}

    r = _client_with_backend(backend).post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(),
        "messages": [
            {"role": "user", "content": "investigate the repo"},
            {"role": "assistant", "content": "running a command"},
            {"role": "tool", "tool_call_id": "c1", "content": "output: 1149 files"},
            {"role": "user", "content": "what did you find?"}],
    })
    assert r.status_code == 200, r.text


def test_flow_full_agent_case_context_plus_tools_emits_tool_calls():
    """THE real coding-agent case, end to end: a multi-turn conversation whose
    context (a secret in a system message + a tool result) the nodes must see,
    PLUS tools in the request, and the terminal node emits tool_calls — proving
    it both SAW the context and can act. Everything together, not in pieces."""
    async def backend(req):
        msgs = req.get("messages") or []
        seen = " ".join(str(m.get("content") or "") for m in msgs)
        sys_txt = " ".join(m.get("content") or "" for m in msgs if m.get("role") == "system")
        marker = "SAW-ZEBRA" if "ZEBRA-9981" in seen else "NO-CONTEXT"
        if "Synthesize" in sys_txt:  # terminal node: act, recording what context it saw
            return {"ok": True, "latency_ms": 1,
                    "response": {"text": None,
                                 "tool_calls": [_tc("act", '{"note":"%s"}' % marker)],
                                 "finish_reason": "tool_calls", "tokens_in": 3, "tokens_out": 5}}
        return {"ok": True, "latency_ms": 1,
                "response": {"text": "draft " + marker, "finish_reason": "stop",
                             "tokens_in": 3, "tokens_out": 5}}

    r = _client_with_backend(backend).post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(),
        "tools": [{"type": "function", "function": {"name": "act",
                   "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "system", "content": "Project context: the code is ZEBRA-9981."},
            {"role": "user", "content": "Use the code to do the task."},
            {"role": "assistant", "content": "reading the project"},
            {"role": "tool", "tool_call_id": "c1", "content": "prior step output"}],
    })
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"           # terminal emits the action
    tc = choice["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "act"
    assert "SAW-ZEBRA" in tc["function"]["arguments"]          # …having SEEN the context


def test_await_delta_or_beat_emits_heartbeat_while_task_runs():
    # The streaming loop's core: while the routed work is still running and no
    # delta has arrived, it must yield a 'beat' (-> a keepalive on the wire) so
    # the 60s ALB idle timeout can't cut a slow flow mid-run. A finished task
    # yields 'done'; a queued delta yields 'delta'.
    import asyncio
    import shim

    async def scenario():
        q = asyncio.Queue()
        slow = asyncio.create_task(asyncio.sleep(0.3))           # "still running"
        beat = await shim._await_delta_or_beat(asyncio.create_task(q.get()), slow, 0.02)
        await q.put("hello")
        delta = await shim._await_delta_or_beat(asyncio.create_task(q.get()), slow, 0.02)
        await slow                                                # let it finish
        done = await shim._await_delta_or_beat(asyncio.create_task(q.get()), slow, 0.02)
        return beat, delta, done

    beat, delta, done = asyncio.run(scenario())
    assert beat == ("beat", None)           # heartbeat while task runs, queue empty
    assert delta == ("delta", "hello")      # a real delta is delivered, not a beat
    assert done == ("done", None)           # task finished -> stop


def test_slow_flow_still_delivers_text_and_trace():
    # A flow that takes real time must still come back as a clean SSE stream with
    # the assembled text and the final chunk (trace) — the heartbeat path must not
    # drop or mangle the result. (The keepalive frames are timing-dependent and
    # unit-tested above.)
    import asyncio as _aio

    async def slow(req):
        await _aio.sleep(0.05)
        return {"ok": True, "latency_ms": 1,
                "response": {"text": "draft", "finish_reason": "stop",
                             "tokens_in": 3, "tokens_out": 5}}

    r = _client_with_backend(slow).post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(), "stream": True,
        "messages": [{"role": "user", "content": "go"}]})
    assert r.status_code == 200, r.text
    body = r.text
    assert "[DONE]" in body
    assert "x_router" in body      # the final chunk (trace) was delivered
    assert "draft" in body         # the assembled text was delivered


def test_failed_flow_streams_trace_and_error_not_empty(monkeypatch):
    # The "61s 200 with nothing" bug: a flow that fails AFTER committing to SSE
    # must still stream its decision_trace (provider:flow + which node) and a real
    # error event — not an empty 200 opencode shows as nothing.
    import asyncio as _aio
    import shim
    monkeypatch.setattr(shim, "_EARLY_FAIL_S", 0.02)   # commit to SSE fast
    monkeypatch.setattr(shim, "_HEARTBEAT_S", 0.02)

    async def backend(req):
        await _aio.sleep(0.15)                          # outlast EARLY_FAIL -> SSE path
        return {"ok": False, "error": "exhausted: timeout", "latency_ms": 150}

    r = _client_with_backend(backend).post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(), "stream": True,
        "messages": [{"role": "user", "content": "go"}]})
    assert r.status_code == 200, r.text
    body = r.text
    assert "x_router" in body              # the trace IS in the stream (not empty)
    assert "flow:" in body                 # recorded as a flow (provider visible)
    assert "exhausted" in body             # the real error kind is surfaced
    assert '"error"' in body


def test_flow_nodes_carry_topology_inputs():
    # The DAG topology (parallel branches -> merge) must be reconstructable from
    # the trace: every traced node carries its `inputs` (edges), and the merge
    # node lists >= 2 inputs (the parallel drafts).
    async def backend(req):
        return {"ok": True, "latency_ms": 1,
                "response": {"text": "x", "finish_reason": "stop",
                             "tokens_in": 1, "tokens_out": 1}}
    r = _client_with_backend(backend).post("/v1/chat/completions", json={
        "model": "", "flow_ir": MOA(),
        "messages": [{"role": "user", "content": "go"}]})
    assert r.status_code == 200, r.text
    nodes = r.json()["x_router"]["decision_trace"]["flow_nodes"]
    assert all(isinstance(n.get("inputs"), list) for n in nodes), "every node carries its edges"
    assert any(len(n["inputs"]) >= 2 for n in nodes), "the merge node lists the parallel branches"
