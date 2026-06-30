"""Modular provider registry — every provider is a COMPOSITION of up to four
aspects, declared in ONE place:

  1. declaration — the structural catalog entry (base_url, auth_env, tier,
     api_kind, capabilities). Stays in `config.live.lua` because the CORE reads
     the catalog; this registry references the provider by `id`.
  2. source      — the catalog/pricing/discovery builder (a `ProviderSource`:
     `pricing()`/`balances()`/`offers_sync`). Optional: a provider with no live
     catalog of its own (e.g. a pure direct wire) declares `source=None`.
  3. adapter     — the wire backend for an `api_kind` (an AsyncCallProviderHook).
     Optional: `adapter=None` ⇒ the default `openai_compatible` backend.
  4. knobs       — operator-tunable settings (merged into `settings.SCHEMA`,
     persisted via the host store), declared next to the provider, not scattered.

`build_source_registry()` (the source list), the api_kind dispatcher handlers,
and the settings SCHEMA all DERIVE from `PROVIDERS` — one source of truth.
Adding a provider = one `config.live.lua` entry + one `Provider` record here.

Behaviour note (`special`): `codex` is the one provider whose source OBSERVES its
own wire backend (the scarcity-price source ingests the backend's quota traffic),
so its source/adapter/streaming are wired imperatively in `serve.py`. The registry
still lists it (for its knobs and for the Config view) but skips building its
source/adapter here — generalising that one coupling into the record would be
mechanism for a single case (act over potency)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable


def _has(catalog: dict, pred: Callable[[str, dict], bool]) -> bool:
    return any(isinstance(p, dict) and pred(pid, p)
               for pid, p in (catalog.get("providers") or {}).items())


@dataclass(frozen=True)
class Provider:
    id: str
    # SOURCE — (catalog, env_get) -> ProviderSource. None = no catalog source.
    source: "Callable[..., Any] | None" = None
    # ENABLED — build the source only when the loaded catalog actually has this
    # provider (preserves build_registry's per-provider gating). None = always.
    enabled: "Callable[[dict], bool] | None" = None
    # ADAPTER (wire) — the api_kind this provider's backend serves + a factory
    # (timeout_s) -> AsyncCallProviderHook. api_kind=None ⇒ default
    # openai_compatible backend (no dedicated adapter).
    api_kind: "str | None" = None
    adapter: "Callable[..., Any] | None" = None
    # streaming twin for the api_kind: a factory () -> stream hook, or None ⇒
    # `stream_unsupported_api_kind` (native non-streaming backends fall back).
    stream_adapter: "Callable[..., Any] | None" = None
    # KNOBS — operator-tunable settings.SCHEMA entries (keyed bare; namespaced
    # `<id>.<knob>` at registration). Declared next to the provider.
    knobs: "dict[str, dict]" = field(default_factory=dict)
    # codex's source↔backend coupling is wired in serve.py (see module docstring).
    special: bool = False


def _openrouter_source(catalog, env_get):
    from sources.openrouter import OpenRouterSource
    return OpenRouterSource(catalog, env_get=env_get)


def _antseed_source(catalog, env_get):
    from sources.antseed import AntSeedSource
    return AntSeedSource(catalog)


def _ollama_source(catalog, env_get):
    from sources.ollama import OllamaSource
    return OllamaSource(catalog, env_get=env_get)


def _bedrock_source(catalog, env_get):
    from sources.bedrock import BedrockSource
    return BedrockSource(catalog, env_get=env_get)


def _anthropic_adapter(timeout_s):
    from provider_adapters.anthropic import make_anthropic_async_call_provider
    return make_anthropic_async_call_provider(timeout_s=timeout_s)


def _bedrock_adapter(timeout_s):
    from provider_adapters.bedrock import make_bedrock_async_call_provider
    return make_bedrock_async_call_provider(timeout_s=timeout_s)


def _google_adapter(timeout_s):
    from provider_adapters.google import make_google_async_call_provider
    return make_google_async_call_provider(timeout_s=timeout_s)


# --- helpers for declaring knobs (mirror settings._i/_f without importing it) --
def _i(env, d):
    try:
        return int(os.getenv(env, str(d)))
    except (TypeError, ValueError):
        return d


def _f(env, d):
    try:
        return float(os.getenv(env, str(d)))
    except (TypeError, ValueError):
        return d


# The provider registry. Each entry composes the aspects above; absent aspects
# are simply None/empty (composition, not inheritance — nothing is forced).
PROVIDERS: "list[Provider]" = [
    Provider(
        "openrouter",
        source=_openrouter_source,
        enabled=lambda c: "openrouter" in (c.get("providers") or {}),
        knobs={
            "runway_credits_low_usd": {
                "type": "float", "default": _f("RUNWAY_CREDITS_LOW_USD", 25),
                "min": 0, "max": 1000000, "label": "Credits runway: low (USD)",
                "help": "Credits below this read as 'low'."},
            "runway_credits_empty_usd": {
                "type": "float", "default": _f("RUNWAY_CREDITS_EMPTY_USD", 1),
                "min": 0, "max": 1000000, "label": "Credits runway: empty (USD)",
                "help": "Credits at/below this read as 'empty'."},
        },
    ),
    Provider(
        "antseed",
        source=_antseed_source,
        enabled=lambda c: _has(c, lambda pid, p: p.get("discovery") == "marketplace"
                               and str(p.get("discovery_id", "")).startswith("antseed")),
        knobs={
            "offers_top_n": {
                "type": "int", "default": _i("ANTSEED_OFFERS_TOP_N", 3),
                "min": 1, "max": 10, "label": "Offers per family (top-N peers)",
                "help": "Cheapest distinct seller peers surfaced per family to rotate between on failure."},
            "reputation_min": {
                "type": "float", "default": _f("ANTSEED_REPUTATION_MIN", 0),
                "min": 0, "max": 100, "label": "Min peer on-chain reputation",
                "help": "Drop AntSeed peers whose on-chain reputation score (0-100) is below "
                        "this. 0 = off. Peers that report no reputation are kept (cold-start safe)."},
            "peer_allowlist": {
                "type": "list", "default": [],
                "min": 0, "max": 500, "label": "Peer allowlist (peer IDs)",
                "help": "If non-empty, ONLY these AntSeed peer IDs are offered. "
                        "Comma-separated. Empty (default) = every peer is eligible."},
            "peer_denylist": {
                "type": "list", "default": [],
                "min": 0, "max": 500, "label": "Peer denylist (peer IDs)",
                "help": "AntSeed peer IDs that are never offered. Comma-separated. "
                        "Takes precedence over the allowlist. Empty (default) = none denied."},
            "runway_deposits_low_usdc": {
                "type": "float", "default": _f("RUNWAY_DEPOSITS_LOW_USDC", 2),
                "min": 0, "max": 100000, "label": "Wallet runway: low (USDC)",
                "help": "Deposits below this read as 'low · top up'."},
            "runway_deposits_empty_usdc": {
                "type": "float", "default": _f("RUNWAY_DEPOSITS_EMPTY_USDC", 0.01),
                "min": 0, "max": 100000, "label": "Wallet runway: empty (USDC)",
                "help": "Deposits at/below this read as 'empty'."},
        },
    ),
    Provider(
        "ollama",
        source=_ollama_source,
        enabled=lambda c: "ollama" in (c.get("providers") or {}),
    ),
    Provider(
        "bedrock",
        source=_bedrock_source,
        enabled=lambda c: _has(c, lambda pid, p: p.get("source") == "bedrock"
                               or str(pid).startswith("bedrock")),
        api_kind="bedrock", adapter=_bedrock_adapter,
    ),
    Provider("anthropic", api_kind="anthropic", adapter=_anthropic_adapter),
    Provider("google", api_kind="google", adapter=_google_adapter),
    # codex: source ↔ backend coupling (observe) wired in serve.py; listed here
    # for its knobs and the Config view. Its source/adapter are skipped below.
    Provider(
        "codex", special=True,
        knobs={
            "imputed_price_in": {
                "type": "float", "default": _f("CODEX_IMPUTED_PRICE_IN", 5),
                "min": 0, "max": 1000, "label": "Scarcity price in ($/Mtok at full demote)",
                "help": "Imputed input price when the subscription quota is fully strained."},
            "imputed_price_out": {
                "type": "float", "default": _f("CODEX_IMPUTED_PRICE_OUT", 25),
                "min": 0, "max": 1000, "label": "Scarcity price out ($/Mtok at full demote)",
                "help": "Imputed output price at full demote."},
            "quota_demote_start": {
                "type": "float", "default": _f("CODEX_QUOTA_DEMOTE_START", 0.5),
                "min": 0, "max": 1, "label": "Quota demote start (fraction)",
                "help": "Quota-used fraction at which the scarcity price ramp begins."},
            "quota_429_window_s": {
                "type": "float", "default": _f("CODEX_QUOTA_429_WINDOW_S", 120),
                "min": 1, "max": 3600, "label": "429 window (s)",
                "help": "How long an observed 429 counts toward the scarcity ramp."},
            "quota_429_shed": {
                "type": "float", "default": _f("CODEX_QUOTA_429_SHED", 3),
                "min": 1, "max": 100, "label": "429s to full demote",
                "help": "Recent 429s within the window that ramp the price to full."},
            "runway_quota_low_fraction": {
                "type": "float", "default": _f("RUNWAY_QUOTA_LOW_FRACTION", 0.8),
                "min": 0, "max": 1, "label": "Quota runway: low (fraction)",
                "help": "Quota-used above this reads as 'low'."},
        },
    ),
]


def build_source_registry(catalog: dict, env_get=os.environ.get) -> list:
    """The ProviderSource list, derived from PROVIDERS: build each provider's
    source when it has one, is enabled for the loaded catalog, and is not the
    specially-wired codex (built in serve.py). Replaces the old conditional
    build_registry — same providers, same gating."""
    out = []
    for p in PROVIDERS:
        if p.special or p.source is None:
            continue
        if p.enabled is not None and not p.enabled(catalog):
            continue
        out.append(p.source(catalog, env_get))
    return out


def native_adapter_handlers(timeout_s: float) -> "dict[str, Any]":
    """api_kind -> wire backend for the providers with a dedicated adapter
    (codex is wired separately in serve.py because it needs `observe`)."""
    return {p.api_kind: p.adapter(timeout_s)
            for p in PROVIDERS
            if not p.special and p.api_kind and p.adapter}


def provider_knob_schema() -> "dict[str, dict]":
    """The provider knobs, namespaced `<id>.<knob>` and stamped with the provider
    group — the per-provider half of settings.SCHEMA, derived from PROVIDERS."""
    schema: dict[str, dict] = {}
    for p in PROVIDERS:
        for name, spec in p.knobs.items():
            schema[f"{p.id}.{name}"] = {**spec, "provider": p.id}
    return schema
