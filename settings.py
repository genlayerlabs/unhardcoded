"""Operator-tunable runtime knobs, overridable from the dashboard Config tab.

Each knob has an ENV default (back-compat) plus an optional override persisted to
a JSON file on the PVC. Reads go through get() so an override applies live: the
writer (ingress dashboard) and the readers (router sources, ingress runway) share
the PVC file; the ingress writes it, the router re-reads it on /x/config/reload.
Unknown keys and out-of-range values are ignored — the default always wins, so a
bad override can never break ranking.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

OVERRIDES_PATH = os.getenv("LLM_ROUTER_CONFIG_OVERRIDES",
                           "/run/llm-router/secrets/config-overrides.json")


def _f(env: str, d: float) -> float:
    try:
        return float(os.getenv(env, str(d)))
    except ValueError:
        return d


def _i(env: str, d: int) -> int:
    try:
        return int(os.getenv(env, str(d)))
    except ValueError:
        return d


# key -> declarative knob. `provider` groups it in the UI; type/min/max validate.
SCHEMA: dict[str, dict[str, Any]] = {
    "compaction.at_tokens": {
        "provider": "compaction", "type": "int", "default": _i("COMPACT_AT_TOKENS", 24000),
        "min": 1000, "max": 2000000, "label": "Suggest compaction at (input tokens)",
        "help": "When a call's input exceeds this, the response carries "
                "x_router.compact=true so the agent knows to POST /v1/compact."},
    "antseed.offers_top_n": {
        "provider": "antseed", "type": "int", "default": _i("ANTSEED_OFFERS_TOP_N", 3),
        "min": 1, "max": 10, "label": "Offers per family (top-N peers)",
        "help": "Cheapest distinct seller peers surfaced per family to rotate between on failure."},
    "antseed.reputation_min": {
        "provider": "antseed", "type": "float", "default": _f("ANTSEED_REPUTATION_MIN", 0),
        "min": 0, "max": 100, "label": "Min peer on-chain reputation",
        "help": "Drop AntSeed peers whose on-chain reputation score (0-100) is below "
                "this. 0 = off. Peers that report no reputation are kept (cold-start safe)."},
    "antseed.peer_allowlist": {
        "provider": "antseed", "type": "list", "default": [],
        "min": 0, "max": 500, "label": "Peer allowlist (peer IDs)",
        "help": "If non-empty, ONLY these AntSeed peer IDs are offered. "
                "Comma-separated. Empty (default) = every peer is eligible."},
    "antseed.peer_denylist": {
        "provider": "antseed", "type": "list", "default": [],
        "min": 0, "max": 500, "label": "Peer denylist (peer IDs)",
        "help": "AntSeed peer IDs that are never offered. Comma-separated. "
                "Takes precedence over the allowlist. Empty (default) = none denied."},
    "antseed.runway_deposits_low_usdc": {
        "provider": "antseed", "type": "float", "default": _f("RUNWAY_DEPOSITS_LOW_USDC", 2),
        "min": 0, "max": 100000, "label": "Wallet runway: low (USDC)",
        "help": "Deposits below this read as 'low · top up'."},
    "antseed.runway_deposits_empty_usdc": {
        "provider": "antseed", "type": "float", "default": _f("RUNWAY_DEPOSITS_EMPTY_USDC", 0.01),
        "min": 0, "max": 100000, "label": "Wallet runway: empty (USDC)",
        "help": "Deposits at/below this read as 'empty'."},
    "codex.imputed_price_in": {
        "provider": "codex", "type": "float", "default": _f("CODEX_IMPUTED_PRICE_IN", 5),
        "min": 0, "max": 1000, "label": "Scarcity price in ($/Mtok at full demote)",
        "help": "Imputed input price when the subscription quota is fully strained."},
    "codex.imputed_price_out": {
        "provider": "codex", "type": "float", "default": _f("CODEX_IMPUTED_PRICE_OUT", 25),
        "min": 0, "max": 1000, "label": "Scarcity price out ($/Mtok at full demote)",
        "help": "Imputed output price at full demote."},
    "codex.quota_demote_start": {
        "provider": "codex", "type": "float", "default": _f("CODEX_QUOTA_DEMOTE_START", 0.5),
        "min": 0, "max": 1, "label": "Quota demote start (fraction)",
        "help": "Quota-used fraction at which the scarcity price ramp begins."},
    "codex.quota_429_window_s": {
        "provider": "codex", "type": "float", "default": _f("CODEX_QUOTA_429_WINDOW_S", 120),
        "min": 1, "max": 3600, "label": "429 window (s)",
        "help": "How long an observed 429 counts toward the scarcity ramp."},
    "codex.quota_429_shed": {
        "provider": "codex", "type": "float", "default": _f("CODEX_QUOTA_429_SHED", 3),
        "min": 1, "max": 100, "label": "429s to full demote",
        "help": "Recent 429s within the window that ramp the price to full."},
    "codex.runway_quota_low_fraction": {
        "provider": "codex", "type": "float", "default": _f("RUNWAY_QUOTA_LOW_FRACTION", 0.8),
        "min": 0, "max": 1, "label": "Quota runway: low (fraction)",
        "help": "Quota-used above this reads as 'low'."},
    "openrouter.runway_credits_low_usd": {
        "provider": "openrouter", "type": "float", "default": _f("RUNWAY_CREDITS_LOW_USD", 25),
        "min": 0, "max": 1000000, "label": "Credits runway: low (USD)",
        "help": "Credits below this read as 'low'."},
    "openrouter.runway_credits_empty_usd": {
        "provider": "openrouter", "type": "float", "default": _f("RUNWAY_CREDITS_EMPTY_USD", 1),
        "min": 0, "max": 1000000, "label": "Credits runway: empty (USD)",
        "help": "Credits at/below this read as 'empty'."},
}

_lock = threading.Lock()
_overrides: dict[str, Any] = {}


def reload() -> dict[str, Any]:
    """Re-read the override file; keep only schema-known keys."""
    global _overrides
    try:
        with open(OVERRIDES_PATH) as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raw = {}
    except (OSError, ValueError):
        raw = {}
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
    tmp = OVERRIDES_PATH + ".tmp"
    try:
        os.makedirs(os.path.dirname(OVERRIDES_PATH), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(new, f)
        os.replace(tmp, OVERRIDES_PATH)
    except OSError as exc:
        return {}, [str(exc)]
    reload()
    return new, []


reload()  # load overrides at import
