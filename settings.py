"""Operator-tunable runtime knobs, overridable from the dashboard Config tab.

Each knob has an ENV default (back-compat) plus an optional override persisted in
the host store (`settings_overrides` table). Reads go through get() so an override
applies live: the writer (ingress dashboard) and the readers (router sources,
ingress runway) share the store; the ingress writes it, the router re-reads it on
/x/config/reload. Unknown keys and out-of-range values are ignored — the default
always wins, so a bad override can never break ranking.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import host_store
from env_coerce import env_int


# key -> declarative knob. `provider` groups it in the UI; type/min/max validate.
SCHEMA: dict[str, dict[str, Any]] = {
    "compaction.at_tokens": {
        "provider": "compaction", "type": "int", "default": env_int("COMPACT_AT_TOKENS", 24000),
        "min": 1000, "max": 2000000, "label": "Suggest compaction at (input tokens)",
        "help": "When a call's input exceeds this, the response carries "
                "x_router.compact=true so the agent knows to POST /v1/compact."},
}
# Per-provider knobs are declared next to each provider in `providers.py` (the
# modular 4th aspect) and merged here, so `settings.get()` and the Config tab keep
# the same flat `<provider>.<knob>` schema while the definitions live with the
# provider rather than scattered in this file.
import providers as _providers  # noqa: E402

SCHEMA.update(_providers.provider_knob_schema())

_lock = threading.Lock()
_overrides: dict[str, Any] = {}


def reload() -> dict[str, Any]:
    """Re-read the overrides from the host store; keep only schema-known keys."""
    global _overrides
    raw = host_store.get_overrides()
    kept = {k: v for k, v in raw.items() if k in SCHEMA}
    with _lock:
        _overrides = kept
    return kept


def _coerce_list(value: Any, d: dict[str, Any]) -> list[str] | None:
    """A CSV string or a JSON array -> a clean, de-duplicated list of tokens
    (order preserved). None when the shape is wrong or it overflows `max` items."""
    if isinstance(value, str):
        items: Any = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return None
    out: list[str] = []
    for it in items:
        s = str(it).strip()
        if s and s not in out:
            out.append(s)
    return out if d["min"] <= len(out) <= d["max"] else None


def _coerce(key: str, value: Any) -> Any:
    d = SCHEMA[key]
    if d["type"] == "list":
        return _coerce_list(value, d)
    try:
        v = int(value) if d["type"] == "int" else float(value)
    except (TypeError, ValueError):
        return None
    return v if d["min"] <= v <= d["max"] else None


def get(key: str) -> Any:
    d = SCHEMA[key]
    with _lock:
        ov = _overrides.get(key)
    if ov is not None:
        c = _coerce(key, ov)
        if c is not None:
            return c
    return d["default"]


def current() -> list[dict[str, Any]]:
    """Every knob with value/default/schema for the dashboard Config tab."""
    with _lock:
        ov = dict(_overrides)
    return [{"key": key, "provider": d["provider"], "type": d["type"],
             "label": d["label"], "help": d["help"],
             "min": d["min"], "max": d["max"], "default": d["default"],
             "value": get(key), "overridden": key in ov}
            for key, d in SCHEMA.items()]


def validate_and_write(updates: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate updates against the schema and merge into the override file.
    A null value clears that override (back to default). Returns
    (new_overrides, errors); writes nothing if any error."""
    errors: list[str] = []
    new = dict(reload())
    for key, value in (updates or {}).items():
        if key not in SCHEMA:
            errors.append(f"unknown knob {key!r}")
            continue
        if value is None:
            new.pop(key, None)
            continue
        c = _coerce(key, value)
        if c is None:
            d = SCHEMA[key]
            errors.append(
                f"{key}: must be a list of <= {d['max']} items" if d["type"] == "list"
                else f"{key}: must be {d['type']} in [{d['min']}, {d['max']}]")
            continue
        new[key] = c
    if errors:
        return {}, errors
    if not host_store.set_overrides(new):
        return {}, ["failed to persist config overrides"]
    reload()
    return new, []


if not os.getenv("ROUTER_SKIP_SETTINGS_IMPORT_RELOAD"):
    reload()  # load overrides at import
