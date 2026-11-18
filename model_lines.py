"""Model lines: a named pointer to the family currently qualified as the head of
a model series.

A policy that names an exact family (`family_eq "gpt-6.1-sol"`) goes stale the
day its successor ships, and every caller that copied the term has to be edited
and redeployed. A line moves that one fact into the host: the policy asks for
`is("line_openai_sol")` and the host answers which family that is today. The
policy bytes — and so the policy id — never change when a line moves.

Two halves, kept apart on purpose:

- DEFINITIONS (`model_lines.json`, reviewed in git): which lines exist, the seed
  head of each, and the family-name pattern a successor must match. The set of
  lines is static because a line is an observation field, and a policy naming an
  undeclared field must fail admission, not resolve to "no candidate".
- HEADS (the `model_line_heads` table): which family each line points at now.
  Moved by `promote`/`rollback`, written by the qualifier
  (`scripts/qualify_model_lines.py`) or by an operator, and re-read by every
  router replica on a short timer. A line with no row serves its seed.

Fail-safe direction: an unreachable store keeps the last-known heads; a missing
or invalid definitions file declares no lines at all (the feature is off) rather
than a half-read set.

This is the `fields.lua` extension seam, like `cache_hot`: the engine observes a
Bool per candidate and learns nothing about lines. Each engine keeps its own
copy of the heads as a Lua table (the getter runs per candidate per request and
must stay native); `subscribe` hands it over and every move republishes it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import weakref
from pathlib import Path
from typing import Any

import host_store

_log = logging.getLogger("llm-router.model-lines")

ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = ROOT / "model_lines.json"
FIELD_PREFIX = "line_"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_FAMILY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$")

# The knobs a line may override; the values are the built-in defaults, used when
# neither the line nor the file's `defaults` block sets one.
QUALIFY_DEFAULTS: dict[str, float] = {
    # A candidate's measured mean cost per probe may be at most this multiple of
    # the head's, measured on the same prompts in the same run.
    "max_cost_ratio": 1.1,
    # Latency passes when the candidate's p90 over successful probes is under
    # this absolute bound ...
    "max_latency_ms": 15000,
    # ... or within this multiple of the head's p90 on the same prompts: a line
    # whose head is already slow is not held to a bar its own head misses.
    "max_latency_ratio": 1.25,
    # Share of probes that must return a valid answer inside the deadlines. Also
    # the bar a head must clear to count as healthy. Below 1 on purpose: one
    # transient provider error in a run is noise, not a regression.
    "min_pass_rate": 0.9,
    # Qualifying runs in a row (on separate runs) before a promotion.
    "consecutive_passes": 2,
    # Newest candidates probed per line per run.
    "max_candidates": 2,
}


class ModelLineError(ValueError):
    """A definitions file or a head move is not acceptable."""


_lock = threading.Lock()
_definitions: "dict[str, dict[str, Any]] | None" = None
_heads: dict[str, str] = {}
_rows: dict[str, dict[str, Any]] = {}
_store_ok: "bool | None" = None
# Engines holding a copy of the heads (LLMRouterHost.sync_model_line_heads).
# Weak: a request-scoped tenant engine drops out when the request ends.
_subscribers: "weakref.WeakSet[Any]" = weakref.WeakSet()


# ---- definitions ---------------------------------------------------------------

def _definitions_path() -> Path:
    raw = os.getenv("MODEL_LINES_FILE", "").strip()
    return Path(raw) if raw else DEFAULT_PATH


def parse_definitions(data: Any) -> dict[str, dict[str, Any]]:
    """Validate a definitions document and return {line: definition} with the
    qualification knobs resolved (line > file defaults > built-in defaults) and
    the pattern compiled. Raises ModelLineError naming the first problem."""
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ModelLineError("model lines file must be an object with version 1")
    file_defaults = data.get("defaults") or {}
    lines = data.get("lines")
    if not isinstance(file_defaults, dict) or not isinstance(lines, dict):
        raise ModelLineError("`defaults` and `lines` must be objects")
    unknown = set(file_defaults) - set(QUALIFY_DEFAULTS)
    if unknown:
        raise ModelLineError(f"unknown default knob(s): {sorted(unknown)}")
    out: dict[str, dict[str, Any]] = {}
    for name, raw in lines.items():
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ModelLineError(f"invalid line name {name!r}")
        if not isinstance(raw, dict):
            raise ModelLineError(f"line {name}: definition must be an object")
        head = raw.get("head")
        if not isinstance(head, str) or not _FAMILY_RE.match(head):
            raise ModelLineError(f"line {name}: `head` must be a family name")
        pattern = raw.get("match")
        compiled = None
        if pattern is not None:
            if not isinstance(pattern, str):
                raise ModelLineError(f"line {name}: `match` must be a string or null")
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ModelLineError(f"line {name}: bad `match`: {exc}") from exc
            if compiled.groups < 1:
                raise ModelLineError(
                    f"line {name}: `match` needs a capture group for the version")
            if version_key(compiled, head) is None:
                raise ModelLineError(
                    f"line {name}: `head` {head!r} does not match its own `match`")
        knobs: dict[str, float] = {}
        for knob, builtin in QUALIFY_DEFAULTS.items():
            value = raw.get(knob, file_defaults.get(knob, builtin))
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ModelLineError(f"line {name}: `{knob}` must be a positive number")
            knobs[knob] = value
        if knobs["min_pass_rate"] > 1:
            raise ModelLineError(f"line {name}: `min_pass_rate` must be <= 1")
        extra = set(raw) - {"head", "match", "description"} - set(QUALIFY_DEFAULTS)
        if extra:
            raise ModelLineError(f"line {name}: unknown key(s) {sorted(extra)}")
        out[name] = {"name": name, "head": head, "match": pattern,
                     "pattern": compiled,
                     "description": str(raw.get("description") or ""), **knobs}
    return out


def load_definitions(path: "str | Path | None" = None) -> dict[str, dict[str, Any]]:
    """Read and validate a definitions file. A missing file is an empty set."""
    p = Path(path) if path is not None else _definitions_path()
    if not p.exists():
        return {}
    return parse_definitions(json.loads(p.read_text(encoding="utf-8")))


def configure(path: "str | Path | None" = None) -> dict[str, dict[str, Any]]:
    """(Re)load the definitions and reset every head to its seed. An invalid file
    declares NO lines and logs why — a policy naming one then fails admission
    loudly instead of the host serving a partial set."""
    global _definitions, _heads, _rows, _store_ok
    try:
        defs = load_definitions(path)
    except (OSError, ValueError) as exc:
        _log.error("model lines disabled: %s", exc)
        defs = {}
    with _lock:
        _definitions = defs
        _heads = {name: d["head"] for name, d in defs.items()}
        _rows = {}
        _store_ok = None
    _publish()
    return defs


def definitions() -> dict[str, dict[str, Any]]:
    """The declared lines, loading them on first use."""
    with _lock:
        defs = _definitions
    return defs if defs is not None else configure()


def field_name(line: str) -> str:
    return FIELD_PREFIX + line


# ---- heads ---------------------------------------------------------------------

def heads() -> dict[str, str]:
    definitions()
    with _lock:
        return dict(_heads)


def head_of(line: str) -> "str | None":
    definitions()
    with _lock:
        return _heads.get(line)


def subscribe(engine: Any) -> None:
    """Give `engine` the current heads now and on every later move. The engine
    exposes `sync_model_line_heads(dict)`; it is held weakly."""
    with _lock:
        _subscribers.add(engine)
        current = dict(_heads)
    engine.sync_model_line_heads(current)


def _publish() -> None:
    """Push the current heads to every live engine. Never raises: one engine
    that cannot be updated must not stop the others, and it keeps its last
    table — the same fail-safe direction as an unreachable store."""
    with _lock:
        engines = list(_subscribers)
        current = dict(_heads)
    for engine in engines:
        try:
            engine.sync_model_line_heads(current)
        except Exception as exc:  # noqa: BLE001
            _log.warning("model lines: engine sync failed: %s", exc)


def reload() -> bool:
    """Re-read the heads from the store. Returns False — and changes nothing —
    when the store is unreachable. A stored head for an undeclared line is
    ignored, and one that is not a plausible family name falls back to the seed."""
    global _heads, _rows, _store_ok
    defs = definitions()
    stored = host_store.get_model_line_heads()
    if stored is None:
        with _lock:
            _store_ok = False
        return False
    new_heads: dict[str, str] = {}
    rows: dict[str, dict[str, Any]] = {}
    for name, d in defs.items():
        row = stored.get(name)
        family = (row or {}).get("family")
        if isinstance(family, str) and _FAMILY_RE.match(family):
            new_heads[name] = family
            rows[name] = row
        else:
            new_heads[name] = d["head"]
    with _lock:
        moved = {n: (old, new_heads[n]) for n, old in _heads.items()
                 if n in new_heads and new_heads[n] != old}
        _heads = new_heads
        _rows = rows
        _store_ok = True
    for name, (old, new) in moved.items():
        _log.info("model line %s head: %s -> %s", name, old, new)
    if moved:
        _publish()
    return True


def _check_move(line: str, family: str, *, force: bool) -> dict[str, Any]:
    d = definitions().get(line)
    if d is None:
        raise ModelLineError(f"unknown line {line!r}")
    if not isinstance(family, str) or not _FAMILY_RE.match(family):
        raise ModelLineError("family must be a family name")
    if not force and d["pattern"] is not None and version_key(d["pattern"], family) is None:
        raise ModelLineError(
            f"{family!r} does not match line {line}'s pattern {d['match']!r}")
    return d


def promote(line: str, family: str, *, actor: str,
            detail: "dict[str, Any] | None" = None,
            expected_current: "str | None" = None,
            force: bool = False, kind: str = "promoted") -> dict[str, Any]:
    """Point `line` at `family`. Reads the store first so the recorded previous
    head is the real one even on a replica that has not refreshed yet."""
    d = _check_move(line, family, force=force)
    reload()
    result = host_store.set_model_line_head(
        line, family, seed_family=d["head"], actor=actor, kind=kind,
        detail=detail, expected_current=expected_current)
    if result.get("ok"):
        reload()
    return result


def rollback(line: str, *, actor: str,
             detail: "dict[str, Any] | None" = None) -> dict[str, Any]:
    """Point `line` back at the head it had before its last move."""
    d = definitions().get(line)
    if d is None:
        raise ModelLineError(f"unknown line {line!r}")
    reload()
    with _lock:
        row = _rows.get(line) or {}
        current = _heads.get(line)
    previous = row.get("previous_family")
    if not previous:
        return {"ok": False, "reason": "no_previous", "family": current}
    result = host_store.set_model_line_head(
        line, previous, seed_family=d["head"], actor=actor, kind="rolled_back",
        detail=detail, expected_current=current)
    if result.get("ok"):
        reload()
    return result


def snapshot(events: int = 20) -> dict[str, Any]:
    """Operator view: every line, its head, where that head came from, and the
    latest audit events."""
    defs = definitions()
    with _lock:
        current = dict(_heads)
        rows = dict(_rows)
        store_ok = _store_ok
    lines = []
    for name, d in defs.items():
        row = rows.get(name) or {}
        lines.append({
            "line": name, "field": field_name(name),
            "description": d["description"], "match": d["match"],
            "head": current.get(name), "seed": d["head"],
            "source": "store" if row else "seed",
            "previous": row.get("previous_family"),
            "updated_at": row.get("updated_at"),
            "updated_by": row.get("updated_by"),
            "qualify": {k: d[k] for k in QUALIFY_DEFAULTS},
        })
    return {"lines": lines, "store_ok": store_ok,
            "events": host_store.model_line_events(limit=events) if events else []}


# ---- successor discovery (pure) -------------------------------------------------

def version_key(pattern: "re.Pattern[str]", family: str) -> "tuple[tuple[int, ...], ...] | None":
    """The ordering key of `family` inside a line, or None when the family is
    not in the line: one integer tuple per capture group, each group split on
    `.`/`-`. Groups compare in order, so `v4.1 / 1017` is ((4, 1), (1017,)) and
    sorts after `v4 / 0731` ((4,), (731,)) — the version decides before the
    snapshot date does, however many parts either has."""
    m = pattern.fullmatch(family)
    if m is None:
        return None
    key: list[tuple[int, ...]] = []
    for group in m.groups():
        if group is None:
            key.append(())
            continue
        parts = re.split(r"[.\-]", group)
        if not all(part.isdigit() for part in parts):
            return None
        key.append(tuple(int(part) for part in parts))
    return tuple(key) if any(key) else None


def successors(definition: dict[str, Any], head: str, families: "list[str]",
               limit: "int | None" = None) -> list[str]:
    """Families of this line that sort strictly after `head`, newest first.
    A line without a pattern, or a head that is not in its own line (an operator
    forced it), has no automatic successors."""
    pattern = definition.get("pattern")
    if pattern is None:
        return []
    head_key = version_key(pattern, head)
    if head_key is None:
        return []
    keyed = []
    for family in set(families):
        key = version_key(pattern, family)
        if key is not None and key > head_key:
            keyed.append((key, family))
    keyed.sort(reverse=True)
    out = [family for _, family in keyed]
    return out[:int(limit)] if limit else out
