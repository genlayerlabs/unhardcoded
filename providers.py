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
and the settings SCHEMA all DERIVE from `PROVIDERS` — one source of truth, and the
single composition root: this module depends DOWN on `sources/*` and
`provider_adapters/*`, which never import back (no cycle).
Adding a provider = one `config.live.lua` entry + one `Provider` record here.

Behaviour note (`special`): `codex` is the one provider whose source OBSERVES its
own wire backend (the scarcity-price source ingests the backend's quota traffic).
Its *source* is built here like any other (it only needs the codex provider id);
what `special` marks is that its ADAPTER and that observe/bind coupling are wired
imperatively in `serve.py` — generalising that one coupling into the record would
be mechanism for a single case (act over potency)."""
from __future__ import annotations

import os
import functools
from dataclasses import dataclass, field
from typing import Any, Callable

from env_coerce import env_float, env_int


def _has(catalog: dict, pred: Callable[[str, dict], bool]) -> bool:
    return any(isinstance(p, dict) and pred(pid, p)
               for pid, p in (catalog.get("providers") or {}).items())


@dataclass(frozen=True)
class Provider:
    id: str
    # SOURCE — (catalog, env_get) -> ProviderSource. None = no catalog source.
    source: "Callable[..., Any] | None" = None
    # ENABLED — build the source only when the loaded catalog actually has this
    # provider (preserves build_source_registry's per-provider gating). None = always.
    enabled: "Callable[[dict], bool] | None" = None
    # ADAPTER (wire) — the api_kind this provider's backend serves + a factory
    # (timeout_s) -> AsyncCallProviderHook. api_kind=None ⇒ default
    # openai_compatible backend (no dedicated adapter).
    api_kind: "str | None" = None
    adapter: "Callable[..., Any] | None" = None
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


def _anthropic_stream_adapter(timeout_s):
    from provider_adapters.anthropic import stream_anthropic
    return functools.partial(stream_anthropic, timeout_s=timeout_s)


def _bedrock_adapter(timeout_s):
    from provider_adapters.bedrock import make_bedrock_async_call_provider
    return make_bedrock_async_call_provider(timeout_s=timeout_s)


def _bedrock_stream_adapter(timeout_s):
    from provider_adapters.bedrock import stream_bedrock
    return functools.partial(stream_bedrock, timeout_s=timeout_s)


def _google_adapter(timeout_s):
    from provider_adapters.google import make_google_async_call_provider
    return make_google_async_call_provider(timeout_s=timeout_s)


def _google_stream_adapter(timeout_s):
    from provider_adapters.google import stream_google
    return functools.partial(stream_google, timeout_s=timeout_s)


def _codex_source(catalog, env_get):
    # codex's source just needs the codex provider id from the loaded catalog;
    # the OBSERVE/bind coupling (the source watching the backend's quota traffic)
    # is wired in serve.py. Constructing the source is not itself special.
    from sources.codex import CodexSource
    pid = next((pid for pid, p in (catalog.get("providers") or {}).items()
                if isinstance(p, dict) and p.get("api_kind") == "openai_codex"), None)
    return CodexSource(pid) if pid else None


def _official_price_source(provider_id):
    # a direct provider (openai/anthropic/google) priced from its OFFICIAL pricing
    # page into host_store.provider_prices — see sources/official_pricing.
    def factory(catalog, env_get):
        from sources.official_pricing import OfficialPriceSource
        return OfficialPriceSource(catalog, provider_id, env_get=env_get)
    return factory


def _present(provider_id):
    return lambda c: provider_id in (c.get("providers") or {})


# The provider registry. Each entry composes the aspects above; absent aspects
# are simply None/empty (composition, not inheritance — nothing is forced).
PROVIDERS: "list[Provider]" = [
    Provider(
        "openrouter",
        source=_openrouter_source,
        enabled=lambda c: "openrouter" in (c.get("providers") or {}),
        knobs={
            "runway_credits_low_usd": {
                "type": "float", "default": env_float("RUNWAY_CREDITS_LOW_USD", 25),
                "min": 0, "max": 1000000, "label": "Credits runway: low (USD)",
                "help": "Credits below this read as 'low'."},
            "runway_credits_empty_usd": {
                "type": "float", "default": env_float("RUNWAY_CREDITS_EMPTY_USD", 1),
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
                "type": "int", "default": env_int("ANTSEED_OFFERS_TOP_N", 3),
                "min": 1, "max": 10, "label": "Offers per family (top-N peers)",
                "help": "Cheapest distinct seller peers surfaced per family to rotate between on failure."},
            "reputation_min": {
                "type": "float", "default": env_float("ANTSEED_REPUTATION_MIN", 0),
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
                "type": "float", "default": env_float("RUNWAY_DEPOSITS_LOW_USDC", 2),
                "min": 0, "max": 100000, "label": "Wallet runway: low (USDC)",
                "help": "Deposits below this read as 'low · top up'."},
            "runway_deposits_empty_usdc": {
                "type": "float", "default": env_float("RUNWAY_DEPOSITS_EMPTY_USDC", 0.01),
                "min": 0, "max": 100000, "label": "Wallet runway: empty (USDC)",
                "help": "Deposits at/below this read as 'empty'."},
            # ---- funding autonomy (the wallet keeper, `wallet_keeper.py`) ------
            # These knobs govern code that moves REAL USDC on Base mainnet. The
            # keeper cross-validates them on every load (see
            # wallet_keeper._load_knobs) and refuses to act on an inconsistent
            # set — settings.SCHEMA validates each knob in isolation, so the
            # RELATIONS between them (trigger above the tourniquet, one top-up
            # within the daily cap) have to be checked where they are read.
            "keeper_enabled": {
                "type": "int", "default": env_int("ANTSEED_KEEPER_ENABLED", 0),
                "min": 0, "max": 1, "label": "Wallet keeper: enabled",
                "help": "Global kill switch for the autonomous funding loop "
                        "(reclaim + top-up). 0 = OFF (default: the loop ships "
                        "dark and is armed deliberately), 1 = ON. Off means the "
                        "keeper reads nothing and fires nothing."},
            # The minimum is NOT cosmetic: config.live.lua's policy_envelope
            # hardcodes the same floor as `credits >= 1.0` and cannot read a
            # knob (it is static Lua evaluated by the core). Allowing this below
            # 1.0 made the knob a silent no-op — offers would rank and the
            # envelope would then reject every one of them, with nothing
            # anywhere explaining the empty ranking. Two definitions of "the
            # minimum spendable escrow" is one too many; this is the tie.
            "min_available_usdc": {
                "type": "float", "default": env_float("ANTSEED_MIN_AVAILABLE_USDC", 1.1),
                "min": 1.0, "max": 100000, "label": "Offer tourniquet: min available (USDC)",
                "help": "Stop offering AntSeed routes at all when escrow "
                        "deposits_available falls below this. Opening a payment "
                        "channel RESERVES ~1 USDC, so below one reserve every "
                        "call 402s insufficient_deposits. Default 1.1 = just "
                        "above one reserve. Cannot go below 1.0: the host "
                        "policy envelope rejects AntSeed below that regardless, "
                        "so a lower value here would suppress nothing and "
                        "explain nothing."},
            "topup_trigger_usdc": {
                "type": "float", "default": env_float("ANTSEED_TOPUP_TRIGGER_USDC", 2.0),
                "min": 0, "max": 100000, "label": "Top-up trigger (USDC)",
                "help": "Deposit when escrow deposits_available falls below "
                        "this. Must be ABOVE the offer tourniquet so funding "
                        "reacts before routing is suppressed."},
            "topup_amount_usdc": {
                "type": "float", "default": env_float("ANTSEED_TOPUP_AMOUNT_USDC", 5),
                "min": 0.01, "max": 50, "label": "Top-up amount (USDC)",
                "help": "USDC moved wallet->escrow per top-up. Hard-capped at 50 "
                        "here AND server-side in antseed/control.js."},
            "topup_wallet_floor_usdc": {
                "type": "float", "default": env_float("ANTSEED_TOPUP_WALLET_FLOOR_USDC", 1),
                "min": 0, "max": 100000, "label": "Top-up: hot-wallet floor (USDC)",
                "help": "Never deposit if it would leave the hot wallet below "
                        "this. VETO-ONLY: the wallet balance comes from an "
                        "untrusted public RPC, so a low read blocks a top-up but "
                        "a high read never authorizes one. An ABSENT or stale "
                        "read blocks too — a floor that is skipped whenever the "
                        "RPC declines to answer is not a floor — so setting "
                        "ANTSEED_WALLET_RPC_URL=off disables top-ups entirely "
                        "(reported as `wallet_unreadable` on /x/runtime). "
                        "Reclaim is deliberately unaffected: failing closed on "
                        "the one action that recovers money would let an RPC "
                        "outage strand the escrow."},
            "topup_daily_cap_usdc": {
                "type": "float", "default": env_float("ANTSEED_TOPUP_DAILY_CAP_USDC", 10),
                "min": 0, "max": 100000, "label": "Top-up: daily cap (USDC)",
                "help": "Maximum USDC the keeper may deposit in any rolling 24h, "
                        "derived from the durable wallet_ops ledger (a pod "
                        "restart does not reset it)."},
            "topup_cooldown_s": {
                "type": "int", "default": env_int("ANTSEED_TOPUP_COOLDOWN_S", 900),
                "min": 0, "max": 86400, "label": "Top-up: cooldown (s)",
                "help": "Minimum seconds between top-ups, also derived from the "
                        "durable ledger."},
            "reclaim_min_usdc": {
                "type": "float", "default": env_float("ANTSEED_RECLAIM_MIN_USDC", 3),
                "min": 0, "max": 100000, "label": "Reclaim: min reserved (USDC)",
                "help": "Only force-close idle payment channels when escrow "
                        "deposits_reserved exceeds this AND the provider has "
                        "served zero successful calls for an hour."},
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
        stream_adapter=_bedrock_stream_adapter,
    ),
    # Direct first-party providers: no marketplace catalog of their own, but a
    # price source feeding host_store.provider_prices from each official pricing
    # page, so they can compete on cost instead of defaulting to +inf.
    Provider("openai", source=_official_price_source("openai"),
             enabled=_present("openai")),
    Provider("anthropic", api_kind="anthropic", adapter=_anthropic_adapter,
             stream_adapter=_anthropic_stream_adapter,
             source=_official_price_source("anthropic"), enabled=_present("anthropic")),
    Provider("google", api_kind="google", adapter=_google_adapter,
             stream_adapter=_google_stream_adapter,
             source=_official_price_source("google"), enabled=_present("google")),
    # codex: its source is built like any provider's (via _codex_source); only the
    # ADAPTER and the observe/bind coupling (the source watching its own backend's
    # quota traffic) are wired imperatively in serve.py — that, not the source, is
    # what `special` marks.
    Provider(
        "codex", special=True,
        source=_codex_source,
        enabled=lambda c: _has(c, lambda pid, p: p.get("api_kind") == "openai_codex"),
        knobs={
            "imputed_price_in": {
                "type": "float", "default": env_float("CODEX_IMPUTED_PRICE_IN", 5),
                "min": 0, "max": 1000, "label": "Scarcity price in ($/Mtok at full demote)",
                "help": "Imputed input price when the subscription quota is fully strained."},
            "imputed_price_out": {
                "type": "float", "default": env_float("CODEX_IMPUTED_PRICE_OUT", 25),
                "min": 0, "max": 1000, "label": "Scarcity price out ($/Mtok at full demote)",
                "help": "Imputed output price at full demote."},
            "quota_demote_start": {
                "type": "float", "default": env_float("CODEX_QUOTA_DEMOTE_START", 0.5),
                "min": 0, "max": 1, "label": "Quota demote start (fraction)",
                "help": "Quota-used fraction at which the scarcity price ramp begins."},
            "quota_429_window_s": {
                "type": "float", "default": env_float("CODEX_QUOTA_429_WINDOW_S", 120),
                "min": 1, "max": 3600, "label": "429 window (s)",
                "help": "How long an observed 429 counts toward the scarcity ramp."},
            "quota_429_shed": {
                "type": "float", "default": env_float("CODEX_QUOTA_429_SHED", 3),
                "min": 1, "max": 100, "label": "429s to full demote",
                "help": "Recent 429s within the window that ramp the price to full."},
            "runway_quota_low_fraction": {
                "type": "float", "default": env_float("RUNWAY_QUOTA_LOW_FRACTION", 0.8),
                "min": 0, "max": 1, "label": "Quota runway: low (fraction)",
                "help": "Quota-used above this reads as 'low'."},
        },
    ),
]


def build_source_registry(catalog: dict, env_get=os.environ.get) -> list:
    """The complete ProviderSource list, derived from PROVIDERS: every provider
    with a source factory, enabled for the loaded catalog, built once. This is the
    single composition root for the registry — serve.py and settings depend DOWN
    on it, and `sources/*` stay leaves (they never import this module)."""
    out = []
    for p in PROVIDERS:
        if p.source is None:
            continue
        if p.enabled is not None and not p.enabled(catalog):
            continue
        s = p.source(catalog, env_get)
        if s is not None:
            out.append(s)
    # AntSeed market rows carry no model metadata at all, so an uncurated peer
    # service would have no context and the core would reject it for any
    # min_context request. When both sources are live, hand the OpenRouter source
    # over as a trait oracle. Wired HERE, at the composition root, so `sources/*`
    # stay leaves that never import one another.
    by_name = {getattr(s, "name", None): s for s in out}
    antseed, openrouter = by_name.get("antseed"), by_name.get("openrouter")
    if antseed is not None and openrouter is not None:
        antseed.bind_trait_source(openrouter)
    return out


def native_adapter_handlers(timeout_s: float) -> "dict[str, Any]":
    """api_kind -> wire backend for the providers with a dedicated adapter
    (codex is wired separately in serve.py because it needs `observe`)."""
    return {p.api_kind: p.adapter(timeout_s)
            for p in PROVIDERS
            if not p.special and p.api_kind and p.adapter}


def native_streaming_adapter_handlers(timeout_s: float) -> "dict[str, Any]":
    """api_kind -> true streaming backend for native providers that support it
    (codex remains wired separately in serve.py because it needs `observe`)."""
    return {p.api_kind: p.stream_adapter(timeout_s)
            for p in PROVIDERS
            if not p.special and p.api_kind and p.stream_adapter}


def _price_multiplier_knob(provider_id: str) -> dict:
    # Default 1.0 (no nudge) for every provider: a routing preference is an
    # operator decision, set + persisted from the Config tab, not hardcoded here
    # (e.g. bedrock < 1.0 to prefer prepaid credits). A REAL per-call surcharge is
    # NOT a multiplier — it belongs in the provider's reported/list price so it
    # ranks AND bills; this lever is ranking-only (billing divides it back out).
    return {
        "provider": provider_id, "type": "float", "default": 1.0,
        "min": 0.1, "max": 100.0, "label": "Ranking price multiplier",
        "help": "A FICTITIOUS routing lever: scales this provider's price for "
                "RANKING only (< 1 = prefer it, > 1 = avoid it). It does NOT change "
                "billing — cost_usd always settles at the real reported cost or the "
                "raw list price. 1.0 = no nudge. Raw source prices stay raw; the "
                "ranking price is computed from the current multiplier at selection time."}


def provider_knob_schema() -> "dict[str, dict]":
    """The provider knobs, namespaced `<id>.<knob>` and stamped with the provider
    group — the per-provider half of settings.SCHEMA, derived from PROVIDERS. Every
    provider that contributes a price (has a source) also gets a ranking-price
    multiplier knob, applied at selection time by the host/core boundary."""
    schema: dict[str, dict] = {}
    for p in PROVIDERS:
        if p.source is not None:
            schema[f"{p.id}.price_multiplier"] = _price_multiplier_knob(p.id)
        for name, spec in p.knobs.items():
            schema[f"{p.id}.{name}"] = {**spec, "provider": p.id}
    return schema
