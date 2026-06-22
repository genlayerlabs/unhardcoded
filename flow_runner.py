"""
flow_runner.py — the host-side Σ_flow scheduler.

The core (`llm_policy.flow`) owns the flow IR: admission, canonical form,
identity, and a *reference* driver. A flow's driver is pure scheduling whose
only effect is `run_node`; we cannot run the reference driver inside the Lua
sandbox with a Python async callback (event-loop re-entrancy), so the host
mirrors the same scheduler here in Python and binds `run_node` to the existing
async router (`LLMRouterHost.execute_async`). This module is a faithful port of
`flow.lua`'s topological order + assembly — kept dependency-free and unit-tested
against the Lua reference so the two never drift.

A node call is just a normal router call with the node's policy and system
prompt, so every node inherits the whole catalog / cascade / pricing / trace
machinery for free.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable


def _nodes(flow: Any) -> dict:
    """Accept either { "flow", nodes } (list form) or {"flow":.., "nodes":..}."""
    if isinstance(flow, (list, tuple)):
        return flow[1] or {}
    return flow.get("nodes") or {}


def _endpoints(nodes: dict) -> tuple[str | None, str | None]:
    src = sink = None
    for nid, node in nodes.items():
        if node.get("kind") == "input":
            src = nid
        elif node.get("kind") == "output":
            sink = nid
    return src, sink


def topo_order(nodes: dict) -> list[str]:
    """Kahn's algorithm over pull-edges (pre -> id for pre in id.inputs).
    Deterministic: ready ids are taken in sorted order so the schedule is
    stable across runs (the flow is already admitted, so this never cycles)."""
    indeg = {nid: 0 for nid in nodes}
    succ: dict[str, list[str]] = {nid: [] for nid in nodes}
    for nid, node in nodes.items():
        for pre in node.get("inputs") or []:
            indeg[nid] += 1
            succ[pre].append(nid)
    queue = sorted(nid for nid, d in indeg.items() if d == 0)
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        newly = []
        for b in succ[nid]:
            indeg[b] -= 1
            if indeg[b] == 0:
                newly.append(b)
        if newly:
            queue = sorted(queue + newly)
    return order


def _part_view(part: dict) -> str:
    """What a downstream node sees from one predecessor: its text and — if it
    PROPOSED tool calls — a serialization of them. A node's tool calls are never
    executed inside the flow (nobody runs a node's tools; they are proposals), so
    they travel as data to the consuming node, letting a synthesizer/terminal node
    weigh the proposed actions before deciding the one it will actually emit."""
    text = part.get("text") or ""
    tcs = part.get("tool_calls")
    if not tcs:
        return text
    lines = []
    for tc in tcs:
        fn = tc.get("function") or {}
        lines.append(f"- {fn.get('name')}({fn.get('arguments')})")
    block = "[proposed tool calls]\n" + "\n".join(lines)
    return f"{text}\n\n{block}" if text else block


def assemble(node: dict, parts: list[dict]) -> str:
    """Build a node's user message from its predecessors' outputs. Mirrors
    flow.lua default_assemble: a `template` with $1,$2,… substitutes; a single
    input passes through; otherwise labeled sections in input order. A
    predecessor's proposed tool calls travel with its text (see _part_view)."""
    template = node.get("template")
    if template:
        import re
        return re.sub(
            r"\$(\d+)",
            lambda m: (_part_view(parts[int(m.group(1)) - 1])
                       if 0 < int(m.group(1)) <= len(parts) else ""),
            template,
        )
    if len(parts) == 1:
        return _part_view(parts[0])
    return "\n\n".join(f"[input {i + 1}]\n{_part_view(p)}" for i, p in enumerate(parts))


async def run_flow(
    flow: Any,
    input_text: str,
    run_node: Callable[[str, dict, str], Awaitable[dict]],
) -> dict:
    """Schedule the (already-admitted, normalized) flow.

    `run_node(node_id, node, prompt)` -> the node's result dict; it must carry
    `{"ok": bool, "text": str, "tool_calls": list|None, ...}` plus whatever the
    trace wants. Returns `{ok, text, tool_calls, trace:[per-node], failed_node?}`
    where `tool_calls` are the TERMINAL (sink-feeding) node's — they alone are
    emitted to the caller; upstream nodes' tool calls travel as proposals in the
    assembled prompt (see _part_view). A node that fails short-circuits the flow
    (the rest of the DAG can't proceed without its output)."""
    nodes = _nodes(flow)
    src, sink = _endpoints(nodes)
    _EMPTY = {"text": "", "tool_calls": None}
    out: dict[str, dict] = {src: {"text": input_text, "tool_calls": None}}
    trace: list[dict] = []

    for nid in topo_order(nodes):
        node = nodes[nid]
        kind = node.get("kind")
        if kind == "input":
            continue
        if kind == "output":
            out[nid] = out.get(node["inputs"][0], _EMPTY)
            continue
        # llm node
        parts = [{"id": pre, **out.get(pre, _EMPTY)} for pre in node.get("inputs") or []]
        prompt = assemble(node, parts)
        result = await run_node(nid, node, prompt)
        # Carry the node's edges (its inputs) so the dashboard can reconstruct the
        # DAG topology — parallel branches and where they merge — not just a flat
        # per-node list.
        trace.append({"node": nid, "inputs": list(node.get("inputs") or []),
                      **(result.get("node_trace") or {})})
        if not result.get("ok"):
            return {"ok": False, "failed_node": nid, "text": "", "tool_calls": None,
                    "error": result.get("error"), "trace": trace}
        out[nid] = {"text": result.get("text") or "",
                    "tool_calls": result.get("tool_calls") or None}

    final = out.get(sink, _EMPTY)
    return {"ok": True, "text": final["text"], "tool_calls": final["tool_calls"],
            "trace": trace}
