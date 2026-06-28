"""
Operator-added providers: an overlay persisted by the dashboard in the host
store (the provider_overlays table), merged into the Lua catalog at startup and
hot-applied at runtime via the shim's POST /x/providers.

Overlay shape:
    {"providers": {
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_kind": "openai_compatible",
            "tier": "partner",
            "auth_env": "GROQ_API_KEY",
            "added_at": 1781112000,
            "served_models": [
                {"family": "llama-3.3-70b",
                 "provider_model_id": "llama-3.3-70b-versatile"}
            ]
        }
    }}

Keys are NEVER stored here — they live in .env.secrets (auth_env indirection,
same as hand-configured providers). The overlay never overwrites a provider
that exists in config.live.lua.
"""
from __future__ import annotations

import re

import host_store

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,39}$")   # 2-40 chars (1 + up to 39)
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,60}$")


def load_overlay() -> dict:
    """The operator-added provider overlay, {'providers': {pid: entry}}, from the
    host store."""
    return host_store.get_provider_overlays()


def save_overlay(overlay: dict) -> bool:
    """Persist the full overlay to the host store; returns True on success, False
    on a persistence failure (so the caller doesn't report a provider added when it
    wasn't durable). Keys are never stored here (they live in .env.secrets)."""
    return host_store.set_provider_overlays((overlay or {}).get("providers") or {})


def validate_entry(pid: str, entry: dict, catalog: dict) -> list[str]:
    """Validation errors for one overlay provider against the loaded catalog."""
    errors: list[str] = []
    if not _ID_RE.match(pid or ""):
        errors.append("id must be lowercase [a-z0-9_-], 2-40 chars")
    if pid in (catalog.get("providers") or {}):
        errors.append(f"provider {pid!r} already exists")
    base_url = str(entry.get("base_url") or "")
    if not base_url.startswith(("http://", "https://")):
        errors.append("base_url must be http(s)")
    if entry.get("api_kind", "openai_compatible") != "openai_compatible":
        errors.append("only api_kind=openai_compatible can be added at runtime")
    if not _ENV_RE.match(str(entry.get("auth_env") or "")):
        errors.append("auth_env must be UPPER_SNAKE_CASE")
    served = entry.get("served_models")
    if not isinstance(served, list) or not served:
        errors.append("served_models must be a non-empty list")
    else:
        families = catalog.get("models") or {}
        for sm in served:
            fam = (sm or {}).get("family")
            if fam not in families:
                errors.append(f"unknown model family {fam!r}")
    return errors


def apply_to_host(host, overlay: dict) -> list[str]:
    """Merge overlay providers into the host's LIVE Lua config table.
    Existing providers are never overwritten; served_by entries are appended
    once. The caller re-runs host.init() afterwards (with dump/restore_state
    around it when the router is already serving)."""
    lua = host.lua
    append = lua.eval("function(t, e) t[#t + 1] = e end")
    new_table = lua.eval("function() return {} end")
    cfg = host.config
    applied: list[str] = []
    for pid, entry in (overlay.get("providers") or {}).items():
        if cfg["providers"][pid] is not None:
            continue
        tbl = new_table()
        for k, v in entry.items():
            if k in ("served_models", "added_at"):
                continue
            tbl[k] = v
        if tbl["api_kind"] is None:
            tbl["api_kind"] = "openai_compatible"
        if tbl["discovery"] is None:
            tbl["discovery"] = "static"
        cfg["providers"][pid] = tbl
        for sm in entry.get("served_models") or []:
            model = cfg["models"][sm.get("family")]
            if model is None:
                continue
            if model["served_by"] is None:
                model["served_by"] = new_table()
            exists = any(row["provider"] == pid for row in model["served_by"].values())
            if exists:
                continue
            row = new_table()
            row["provider"] = pid
            if sm.get("provider_model_id"):
                row["provider_model_id"] = sm["provider_model_id"]
            append(model["served_by"], row)
        applied.append(pid)
    return applied
