"""
AntSeed source: offers, prices and wallet balances for the AntSeed buyer
proxies.

Both the marketplace book (`peer_offers`) and the buyer status (`buyer_status`:
session pin + escrow + wallet) are read from the host store, which the antseed
sidecar writes — the book from `antseed network browse --services --json`, the
status from `antseed buyer status --json`. The source no longer touches the
filesystem; the buyer daemon's control API stays a unix socket inside the
antseed containers (only the proxy ports are shared with the router's netns).
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import host_store
import route_reliability as _route_reliability
import settings
from sources import Balance, Price

_log = logging.getLogger("unhardcoded.sources.antseed")

STALE_AFTER_S = 900

# Wedge detector: this many consecutive FAILED attempts, while offers were
# actually being ranked, reads as "the provider is offering routes it cannot
# serve" — the exact 24h/5,432-attempts/zero-successes shape the funding gate
# exists to prevent. Surfaced as `wallet_health` on the source stats.
WEDGE_CONSECUTIVE_FAILURES = 10

# How many unbound wire names snapshot_stats() ranks. The point of the list is a
# curation QUEUE an operator works top-down, not an inventory.
UNBOUND_TOP_N = 20

# Leading vendor token -> the ONE vendor it names. `claude`/`anthropic` and
# `gemini`/`google` are the same vendor spelled two ways, so an `anthropic-`
# prefixed wire name still reaches a `claude-` prefixed curated family — while a
# peer that names a DIFFERENT vendor than the family's own can never reach it.
# Only brand tokens belong here: a token that is also the head of a real model
# name (`gemma`, `mistral`) would leave a residue like `large` that is far too
# generic to be a safe route target.
_VENDOR_PREFIXES: dict[str, str] = {
    "anthropic": "anthropic", "claude": "anthropic",
    "google": "google", "gemini": "google",
    "openai": "openai",
    "meta": "meta", "meta-llama": "meta",
    "qwen": "qwen",
    "x-ai": "x-ai", "xai": "x-ai",
    "deepseek": "deepseek",
    "moonshot": "moonshot", "moonshotai": "moonshot",
    "z-ai": "z-ai",
    "minimax": "minimax",
    "mistralai": "mistral",
}
# longest first, so `meta-llama-…` is not read as `meta-` + `llama-…`
_VENDOR_TOKENS = tuple(sorted(_VENDOR_PREFIXES, key=len, reverse=True))
# The canonical vendor ids the table resolves to. A catalog `vendor = "…"`
# annotation may be written either as a wire token (`mistralai`, `xai`) or as the
# canonical id it maps to (`mistral`, `x-ai`) — both name the same vendor.
_VENDOR_IDS = frozenset(_VENDOR_PREFIXES.values())

# A residue whose FIRST segment is a bare version token names no model: `v4-pro`
# (from `deepseek-v4-pro`), `v3-2`, `m2-7` (`minimax-m2.7`), `3-1-pro-preview`
# (`gemini-3.1-pro-preview`). `v4-pro` is the residue from the original
# production incident — every vendor ships a "v4 pro", so the bare form is a
# coin flip between them, not a name. Matching the leading segment (optional
# one-or-two letter prefix, then digits only) refuses all of those while leaving
# every residue that leads with a real name word — `opus-4-8`, `sonnet-4-6`,
# `haiku-4-5`, `fable-5` — reachable.
#
# It is deliberately blunt: a future `openai-o3-mini` would be refused too, since
# `o3` is structurally indistinguishable from `v4`. Refusing costs a raw-name
# offer the operator can re-bind with one `service_aliases` entry; admitting
# costs a silently wrong model. The `service_aliases` bridge is the sanctioned
# escape hatch, exactly as for the letter/digit boundary `_canon_service` leaves
# alone.
_VERSIONISH_HEAD = re.compile(r"^[a-z]{0,2}[0-9]+$")

# Serving-MODE markers: the same weights behind a different serving switch. Each
# gets its OWN family (`glm-5.1:web` -> `glm-5.1@web`) and is never folded into the
# base — folding would silently land an existing `family_eq("glm-5.1")` policy on
# the web-search product. Checked once, longest-first where they overlap
# (`-non-thinking` before `-thinking`).
#
# NEVER listed here, deliberately: `-uncensored` and `-it` are different WEIGHTS,
# not a serving mode (and prod policies name `gemma-4-31b-it` *with* the suffix),
# and a bare `-p` has no known meaning — a one-letter rule would eat real name
# segments. Those stay separate families; see the tests that pin them.
_VARIANT_SUFFIXES = (
    (":web", "web"), ("@web", "web"),
    ("-non-thinking", "non-thinking"),
    ("-thinking", "thinking"),
    ("-fast", "fast"),
)
_VARIANT_PREFIXES = (("e2ee-", "e2ee"),)


def _canon_service(name: str) -> str:
    """Canonical key for a model name: lowercase, with every run of non-alphanumerics
    folded to a single `-`. The same shape as sources/bedrock.py's `_norm`, so
    `Anthropic/Claude-Opus-4.8`, `anthropic:claude_opus_4.8` and
    `anthropic-claude-opus-4-8` are one key.

    Separators are FOLDED, never REMOVED: `gpt55` and `gpt-5.5` canonicalize to
    `gpt55` and `gpt-5-5`, two different models, and stay that way. It also does
    NOT bridge a letter/digit boundary (`gemma4-31b` vs `gemma-4-31b`) — that is an
    operator `service_aliases` decision, not something to guess.

    Note what this no longer does: it does not strip vendor prefixes. Stripping on
    BOTH sides is what used to file `deepseek-v4-pro` under the vendor-free residue
    `v4-pro`, where a peer's `x-ai-v4-pro` matched it — silent cross-vendor routing.
    Vendor handling now lives in `_split_vendor` + the two-layer family index."""
    return re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")


def _squash(name: str) -> str:
    """Every separator REMOVED (`gemma4-31b` and `gemma-4-31b` -> `gemma431b`).
    Strictly an advisory key for the `near_miss` curation hint — never for binding,
    because it also collapses `gpt55` onto `gpt-5.5`, which is the merge
    `_canon_service` exists to prevent."""
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _split_vendor(canon: str) -> tuple[str | None, str]:
    """(vendor, remainder) for a canonical name carrying a leading vendor token,
    else (None, canon) — the peer made no vendor claim we could contradict."""
    for token in _VENDOR_TOKENS:
        if canon.startswith(token + "-"):
            return _VENDOR_PREFIXES[token], canon[len(token) + 1:]
    return None, canon


def _split_variant(name: str) -> tuple[str, str | None]:
    """(base, variant) for a wire name carrying a serving-mode marker, else
    (name, None). Runs on the RAW lowercased name so `:web` is still visible —
    `_canon_service` would have folded the colon into a dash.

    At most ONE marker is taken off, suffix first. A peer stacking two
    (`e2ee-glm-5.1:web`) therefore yields the base `e2ee-glm-5.1`, which is not a
    curated family, so `_bind` finds nothing and the offer stays exposed under its
    raw wire name. Stacked markers are simply NOT supported — safe (no wrong
    route) and, since no peer has been seen selling one, not worth a second pass
    that would have to decide what `@e2ee@web` even means."""
    s = (name or "").strip().lower()
    for suffix, variant in _VARIANT_SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix):
            return s[: -len(suffix)], variant
    for prefix, variant in _VARIANT_PREFIXES:
        if s.startswith(prefix) and len(s) > len(prefix):
            return s[len(prefix):], variant
    return s, None

# Buyer hot-wallet on-chain reads. The marketplace spends from ESCROW
# (depositsAvailable); the raw wallet balance — USDC sitting in the wallet, plus
# ETH for gas — is what tells you whether you can deposit more or pay for a tx at
# all. The buyer CLI/status file expose neither, so we read them straight from
# Base. Native (Circle) USDC on Base mainnet, 6 decimals.
_BASE_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
_USDC_DECIMALS = 6
_DEFAULT_BASE_RPC = "https://mainnet.base.org"


def as_float(value: Any) -> "float | None":
    """The buyer reports its escrow as STRINGS; coerce one to a float, or None
    when it is absent/unparseable. None means UNKNOWN — never zero: a funding
    decision must be able to tell "no money" apart from "no reading"."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _wallet_rpc_url() -> str | None:
    """Base RPC for the wallet balance read. Defaults to a public endpoint;
    set ANTSEED_WALLET_RPC_URL to override, or to ""/off/none to disable the
    on-chain read entirely (then the dashboard shows escrow only)."""
    raw = os.getenv("ANTSEED_WALLET_RPC_URL")
    if raw is None or not raw.strip():
        return _DEFAULT_BASE_RPC  # unset / empty (copied template) -> default on
    raw = raw.strip()
    if raw.lower() in ("off", "none", "disabled"):
        return None
    return raw


async def _fetch_chain_balances(rpc_url: str, address: str) -> dict:
    """Best-effort on-chain read of the wallet's native ETH and USDC balances on
    Base, via a batched JSON-RPC call. Returns {} (never raises) on any failure —
    bad address, network error, RPC error — so a flaky RPC never wedges the poll."""
    if not address or not address.startswith("0x") or len(address) != 42:
        return {}
    addr = address.lower()
    # USDC balanceOf(addr): selector 0x70a08231 + the 32-byte left-padded address.
    call_data = "0x70a08231" + "0" * 24 + addr[2:]
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "eth_getBalance", "params": [addr, "latest"]},
        {"jsonrpc": "2.0", "id": 2, "method": "eth_call",
         "params": [{"to": _BASE_USDC, "data": call_data}, "latest"]},
    ]
    out: dict = {}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=6.0) as c:
            resp = await c.post(rpc_url, json=batch)
            resp.raise_for_status()
            results = resp.json()
        by_id = {r.get("id"): r.get("result") for r in results
                 if isinstance(r, dict)} if isinstance(results, list) else {}
        eth_hex, usdc_hex = by_id.get(1), by_id.get(2)
        if isinstance(eth_hex, str) and eth_hex.startswith("0x"):
            out["wallet_eth"] = int(eth_hex, 16) / 1e18
        if isinstance(usdc_hex, str) and usdc_hex.startswith("0x") and usdc_hex != "0x":
            out["wallet_usdc"] = int(usdc_hex, 16) / (10 ** _USDC_DECIMALS)
    except Exception:  # noqa: BLE001 — on-chain read is best-effort
        return out
    return out


class AntSeedSource:
    name = "antseed"
    poll_interval_s = 300

    def __init__(self, catalog: dict):
        self._models = catalog.get("models") or {}
        # provider_id -> its marketplace config (cap, aliases, endpoint)
        self._providers: dict[str, dict] = {
            pid: p for pid, p in (catalog.get("providers") or {}).items()
            if isinstance(p, dict) and p.get("discovery") == "marketplace"
            and str(p.get("discovery_id", "")).startswith("antseed")
        }
        self.provider_ids = list(self._providers)
        self._stats: dict[str, Any] = {"stale": False, "suppressed_no_funds": 0,
                                       "wallet_health": "unknown"}
        # optional live model-metadata oracle for UNCURATED names (bind_trait_source)
        self._trait_source: Any = None

    def bind_trait_source(self, source: Any) -> None:
        """Hand this source a live model-metadata oracle: any source exposing
        `live_offers()` rows with `model_family` / `wire_model_id` /
        `capabilities.context` / `traits` — in practice sources/openrouter, whose
        live /models snapshot covers most of what AntSeed peers resell.

        AntSeed rows carry NO model metadata of their own, so without this an
        uncurated peer service has no `context` and the core rejects it for any
        `min_context` request (core/llm_policy/filter.lua). Wired at the
        composition root (providers.build_source_registry), never imported across
        sources — `sources/*` stay leaves."""
        self._trait_source = source
        self._trait_index_cache = None

    # ---- market parsing -------------------------------------------------

    def _load_market(self) -> list[dict]:
        """[{peer_id, service, price_in, price_out, price_cached_in,
        max_concurrency, reputation, last_seen}] per peer-service row from the
        host store (written by the antseed sidecar within the sliding window), or
        [] when none are fresh (degraded: no antseed candidates). The fields are
        raw seller announcements: cap mirroring (price_cached_in), per-peer gating
        (max_concurrency), reputation admission and dashboard freshness (last_seen)
        are applied downstream in offers_sync / market_book."""
        rows = host_store.peer_offers(STALE_AFTER_S * 1000)
        self._stats["stale"] = not rows
        return rows

    def _pinned_peer(self, provider_id: str) -> str | None:
        """An optional buyer-side *session* pin (buyer_status' pinned_peer_id).
        Browse mode leaves it null and the host pins per request instead (the
        offer carries peer_id -> x-antseed-pin-peer); when a session pin IS set,
        restrict offers to that peer's services to match what the proxy serves."""
        data = host_store.buyer_status(provider_id)
        return (data or {}).get("pinned_peer_id") or None

    def _family_vendor(self, fam: str, canon: str) -> str | None:
        """The vendor a curated family belongs to, or None when it is UNKNOWN.

        Two sources, strongest first: the operator's explicit `vendor = "…"`
        annotation on the catalog entry, then the vendor token the family's OWN
        name leads with (`claude-opus-4-8` -> anthropic). The annotation exists
        because most families are named WITHOUT their vendor (`glm-5.1`,
        `qwen3-coder`, `gemma-3-27b`), and an unknown vendor now refuses every
        vendor-claiming wire name (see `_match_curated`) — so the annotation is
        what re-opens the legitimate `<vendor>-<family>` spellings peers actually
        advertise (`z-ai-glm-5.1`), and ONLY those.

        The annotation is authoritative when present: it is the operator stating
        the fact, where the name token is an inference from spelling.

        An annotation naming a vendor this module cannot recognise is dead
        config — a peer can only ever claim one of `_VENDOR_PREFIXES`' tokens, so
        an unrecognised value can never match anything and the family stays
        vendor-unknown. That is a typo an operator must see, so it is LOUD (WARN
        + `_stats` `vendor_unknown`) rather than a silently ineffective line."""
        raw = (self._models.get(fam) or {}).get("vendor")
        if raw:
            key = _canon_service(raw)
            vendor = _VENDOR_PREFIXES.get(key) or (key if key in _VENDOR_IDS else None)
            if vendor is None:
                self._unknown_vendors.add(f"{fam}={raw}")
                _log.warning(
                    "antseed: catalog family %r declares vendor %r, which is not a "
                    "vendor any peer can name — the annotation has NO effect and "
                    "the family refuses every vendor-prefixed wire name. Use one "
                    "of %s.", fam, raw, sorted(_VENDOR_IDS))
                return None
            return vendor
        vendor, _ = _split_vendor(canon)
        return vendor

    def _canon_models(self) -> tuple[dict[str, str], dict[str, tuple[str, str]],
                                     dict[str, str]]:
        """Lazy (exact, bare, vendors) family index over self._models, built once.

        `exact` files every curated family under its OWN canonical name
        (`claude-opus-4-8` -> `claude-opus-4-8`). Nothing is stripped here: a
        vendor-stripped INDEX is what created the cross-vendor mis-binds, because
        it exposed `v4-pro` / `fable-5` / `sonnet-4-6` as vendor-free keys any
        prefixed wire name could reach after its own prefix came off.

        `bare` files the vendor-carrying families a SECOND time under their
        vendor-stripped residue, together with that vendor
        (`claude-opus-4-8` -> ("claude-opus-4-8", "anthropic")). It is consulted
        only for a wire name that claims no vendor, or the same one — so a peer's
        bare `opus-4.8` still reaches `claude-opus-4-8`, while `meta-opus-4-8`
        never does. A residue that is a bare version number names no model and is
        skipped (`_VERSIONISH_HEAD`).

        `vendors` files every family whose vendor is KNOWN — annotated or read
        off its own name (`_family_vendor`). A family absent from it has an
        UNKNOWN vendor and, per `_match_curated`, refuses every wire name that
        claims one.

        A canonical form shared by TWO curated families is AMBIGUOUS and dropped
        from that layer — never risk routing to the wrong model; the offer falls
        through to its raw wire name. The drop is LOUD (WARN + `_stats`
        `ambiguous`): catalog growth silently unbinding a family that worked
        yesterday is exactly the failure this index exists to prevent."""
        cached = getattr(self, "_canon_models_cache", None)
        if cached is not None:
            return cached

        # key -> the families claiming it, collected per layer so an ambiguous
        # residue never costs a family its own unambiguous exact key.
        exact_claims: dict[str, set[str]] = {}
        bare_claims: dict[str, set[str]] = {}
        bare_vendor: dict[str, str] = {}
        vendors: dict[str, str] = {}
        self._unknown_vendors: set[str] = set()
        for fam in sorted(self._models):
            canon = _canon_service(fam)
            exact_claims.setdefault(canon, set()).add(fam)
            fam_vendor = self._family_vendor(fam, canon)
            if fam_vendor is not None:
                vendors[fam] = fam_vendor
            vendor, residue = _split_vendor(canon)
            if vendor and residue and not _VERSIONISH_HEAD.match(residue.split("-")[0]):
                bare_claims.setdefault(residue, set()).add(fam)
                bare_vendor[residue] = vendor

        ambiguous: set[str] = set()

        def resolve(layer: str, claims: dict[str, set[str]]) -> dict[str, str]:
            out: dict[str, str] = {}
            for key, families in claims.items():
                if len(families) == 1:
                    out[key] = next(iter(families))
                    continue
                ambiguous.add(key)
                _log.warning(
                    "antseed: catalog families %s share the %s index key %r — "
                    "dropped, so peers advertising that name stay UNBOUND. Rename "
                    "one family or add a service_aliases entry.",
                    sorted(families), layer, key)
            return out

        exact = resolve("exact", exact_claims)
        bare = {key: (fam, bare_vendor[key])
                for key, fam in resolve("bare", bare_claims).items()}
        self._stats["ambiguous"] = sorted(ambiguous)
        self._stats["vendor_unknown"] = sorted(self._unknown_vendors)

        cached = self._canon_models_cache = (exact, bare, vendors)
        return cached

    def _match_curated(self, aliases: dict, name: str) -> str | None:
        """The curated family a wire name denotes, or None. Ordered strongest
        evidence first: the operator's alias, the exact catalog key, the canonical
        form, and only then a vendor-aware match."""
        alias = aliases.get(name)
        if alias is not None and alias in self._models:
            return alias
        if name in self._models:
            return name
        exact, bare, vendors = self._canon_models()
        canon = _canon_service(name)
        family = exact.get(canon)
        if family is not None:
            return family
        vendor, residue = _split_vendor(canon)
        if vendor is None:
            # No vendor claimed -> the vendor-free residue index may answer:
            # `opus-4.8` reaches `claude-opus-4-8`, unambiguously by construction.
            hit = bare.get(canon)
            return hit[0] if hit is not None else None
        # A vendor WAS claimed, so the peer made an assertion that can be
        # CONTRADICTED — and the bind is allowed only when the catalog can
        # positively confirm it. The family's vendor must be KNOWN (its own name
        # carries the token, or the operator annotated it) and must AGREE.
        #
        # Unknown vendor + an explicit peer claim -> refuse, stay raw. The earlier
        # rule inverted the burden: it read a family whose own name carries no
        # vendor token as "makes no claim to contradict" and accepted the bind
        # unconditionally, which is how `x-ai-glm-5.1` reached z-ai's `glm-5.1`
        # and `deepseek-llama-3.3-70b` — the naming shape of a real distill with
        # DIFFERENT weights — reached meta's `llama-3.3-70b`. The catalog having
        # nothing to check against is not evidence the claim is true.
        #
        # The remainder may spell the family in full (`anthropic-` +
        # `claude-opus-4-8`, the double-prefix form), or name the residue of a
        # family belonging to the same vendor (`anthropic-sonnet-4-6`).
        family = exact.get(residue)
        if family is not None:
            return family if vendors.get(family) == vendor else None
        hit = bare.get(residue)
        if hit is not None and hit[1] == vendor:
            return hit[0]
        return None

    def _bind(self, provider_cfg: dict, service: str) -> dict | None:
        """Resolve a peer's wire model name to a routable family:
        {family, base_family, variant}, or None when nothing curated matches (the
        caller then exposes the service under its raw wire name).

        A serving-mode variant of a curated family gets its own `<base>@<variant>`
        family rather than being merged into the base. `family_eq` is an exact
        string compare (core/llm_policy/filter.lua), so this leaves every existing
        policy semantically untouched while unifying all spellings of a variant
        (`glm-5.1:web`, `glm-5.1@web`) into ONE reachable family, and it needs no
        core change on the SELECTION path — a family name is opaque there.

        `@` is NOT free text everywhere in the engine, though: the metrics seed
        keys its rows `"<family>@<provider>"` and splits on the FIRST `@`
        (core/llm_policy.lua seed_runtime_from_metrics), so a seed row for
        `glm-5.1@web` would be read as family `glm-5.1`, provider `web@…` and
        mis-seed the base family. Latent only because metrics.live.lua seeds no
        variant family — variants are discovered, never seeded. Do not seed one
        without changing that split (or the separator here)."""
        aliases = provider_cfg.get("service_aliases") or {}
        family = self._match_curated(aliases, service)
        if family is not None:
            return {"family": family, "base_family": family, "variant": None}
        base, variant = _split_variant(service)
        if variant is None:
            return None
        # Only a CURATED base earns an `@variant` family: that keeps every variant
        # anchored to a real family whose capabilities it can inherit, and leaves
        # uncurated names exposed under their raw wire name as before.
        family = self._match_curated(aliases, base)
        if family is None:
            return None
        return {"family": f"{family}@{variant}", "base_family": family,
                "variant": variant}

    def _family_for(self, provider_cfg: dict, service: str) -> str | None:
        """The policy-facing family for a wire name (see `_bind`), or None."""
        bound = self._bind(provider_cfg, service)
        return bound["family"] if bound is not None else None

    # ---- model metadata ---------------------------------------------------

    def _trait_index(self) -> dict[str, dict]:
        """{canonical model name -> {"context", "traits"}} over the bound trait
        oracle's live catalog, keyed by BOTH the oracle's policy family and its
        wire id so either spelling a peer uses lands. Rebuilt only when the oracle
        publishes a new snapshot list (it replaces the list per refresh), so the
        join stays off the per-rank path. First entry wins: a later duplicate is
        the same model under another vendor scope."""
        if self._trait_source is None:
            return {}
        offers = self._trait_source.live_offers()
        cached = getattr(self, "_trait_index_cache", None)
        if cached is not None and cached[0] is offers:
            return cached[1]
        index: dict[str, dict] = {}
        for offer in offers:
            ctx = (offer.get("capabilities") or {}).get("context")
            meta = {"context": ctx, "traits": offer.get("traits") or {}}
            for name in (offer.get("model_family"), offer.get("wire_model_id")):
                key = _canon_service(name) if name else ""
                if key and key not in index:
                    index[key] = meta
        self._trait_index_cache = (offers, index)
        return index

    def _model_meta(self, row: dict) -> dict:
        """{capabilities, quality_hint, traits} for one market row.

        A curated family hands over its capabilities and quality hint. An
        `@variant` of one inherits the CAPABILITIES but NOT the quality hint.

        The split follows what each number is. Context and tool/json support are
        architecture — no serving-mode switch shrinks a context window — and
        withholding them would drop the variant from every `min_context` request
        (core/llm_policy/filter.lua) for no gain. The quality hint is a claim
        about specific WEIGHTS, and "the marker means the same weights" is a
        premise this module asserts, not one it can check: `-fast` in marketplace
        practice routinely means quantized or distilled, and the marker set is
        applied to every family alike, so a peer can mint `llama-3.3-70b-thinking`
        for a family that has no thinking mode. Inheriting the hint lets any peer
        conjure a family that clears `min_quality` on a number it did not earn,
        by spelling a suffix.

        The asymmetry decides it: an un-inherited hint makes a variant invisible
        to quality-gated policies, which an operator fixes by curating
        `glm-5.1@web` as its own catalog entry with its own measured hint (a
        one-entry job, and the offer still serves every ungated request meanwhile);
        an inherited hint sends a quality-gated prompt to unverified weights
        silently, at request time, with nothing to notice it by.

        KNOWN GAP, not fixed here: learned per-route verdicts are keyed by family
        (route_reliability.route_key = provider|family|peer), so a peer whose
        reliability is burned under `glm-5.1` starts clean under `glm-5.1@fast`.
        Closing it means keying observations by base family — a change to the
        route-identity contract shared with host_store, out of scope for the
        naming layer.

        A genuinely uncurated name gets what the live trait oracle can prove and
        nothing more. An absent context stays ABSENT: inventing one to clear a
        `min_context` gate would route a prompt to a model that silently truncates
        it."""
        base = row.get("base_family")
        model = self._models.get(base) if base else None
        if model:
            return {"capabilities": dict(model.get("capabilities") or {}),
                    "quality_hint": (None if row.get("variant")
                                     else model.get("static_quality_hint")),
                    "traits": None}
        meta = self._trait_index().get(_canon_service(row["service"]))
        if meta is None:
            return {"capabilities": {}, "quality_hint": None, "traits": None}
        caps = {"context": int(meta["context"])} if meta["context"] else {}
        return {"capabilities": caps, "quality_hint": None,
                "traits": meta["traits"] or None}

    def _unbound_top(self, sellers: dict[str, set[str]]) -> list[dict]:
        """The unbound wire names ranked by DISTINCT sellers — "161 unbound" turned
        into a curation queue an operator works top-down. `near_miss` names the
        curated family the wire name would reach if someone added one alias: the
        family it matches once every separator is removed (`gemma4-31b` vs
        `gemma-4-31b`). Deliberately looser than the binding rule, which never
        removes separators — so it also flags `gpt55` against `gpt-5.5`. That is
        the point: a near miss is a question for an operator, never a route."""
        squashed = getattr(self, "_squashed_models_cache", None)
        if squashed is None:
            squashed = self._squashed_models_cache = {}
            for fam in sorted(self._models):
                squashed.setdefault(_squash(fam), fam)
        ranked = sorted(sellers.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        out = []
        for service, peers in ranked[:UNBOUND_TOP_N]:
            base, _variant = _split_variant(service)
            out.append({"service": service, "sellers": len(peers),
                        "near_miss": squashed.get(_squash(base))})
        return out

    def _reset_market_stats(self) -> None:
        """Zero the per-tick market counters, for a tick that returns before
        reading the market. `stale` becomes None — the flag says whether the last
        market READ found fresh rows, and a suppressed tick performs none, so
        neither True nor False is true of it. None is UNKNOWN here for the same
        reason it is in `as_float`: a reading that was not taken must not be
        reported as a reading of zero.

        `suppressed_no_funds`, `deposits_available`, `wallet_health` and
        `ambiguous` are deliberately NOT reset — they are the state of the buyer
        and the catalog, not of this tick's market read."""
        self._stats["stale"] = None
        self._stats["uncurated"] = 0
        self._stats["rejected_by_buyer"] = 0
        self._stats["rejected_by_reputation"] = 0
        self._stats["denied"] = 0
        self._stats["offers"] = 0
        self._stats["unbound_top"] = []

    def offers_sync(self, provider_id: str) -> list[dict]:
        """One offer per advertised service for this buyer proxy — the WHOLE
        market, not just curated families. A service that maps to a curated family
        (directly, or as `<family>@<variant>`) carries that family's
        benchmark/capabilities; every other service is exposed under its raw wire
        name, carrying whatever the live trait oracle can prove about it and never
        dropped. Sync: called from the core's discover hook inside rank."""
        cfg = self._providers.get(provider_id)
        if cfg is None:
            return []
        cap = cfg.get("market_price_cap") or {}
        cap_in = float(cap.get("input", float("inf")))
        cap_out = float(cap.get("output", float("inf")))
        # ONE buyer_status read serves both the session pin and the funds gate.
        status = host_store.buyer_status(provider_id) or {}
        pinned = status.get("pinned_peer_id") or None
        available = as_float(status.get("deposits_available"))
        self._stats["deposits_available"] = available
        # FUNDS TOURNIQUET. Opening a payment channel RESERVES ~1 USDC of escrow,
        # so once deposits_available drops below one reserve EVERY routed call
        # 402s `insufficient_deposits` — prod burned 5,432 attempts in 24h for
        # zero successes that way. Offering routes that provably cannot pay is
        # worse than offering none: the caller's request is spent on a certain
        # failure and the peer's reliability score is poisoned. Suppress instead.
        #
        # FAILS OPEN, deliberately: `available is None` means the status row is
        # absent or unreadable (cold start, DB blip, sidecar lag) — a READ error,
        # not evidence of an empty escrow — so the provider stays offered. The
        # host envelope's `credits` clause is the belt to this braces and fails
        # CLOSED on the same signal; the asymmetry is the point (see
        # config.live.lua policy_envelope + sources.seed_credits).
        if available is not None and available < float(
                settings.get("antseed.min_available_usdc")):
            self._stats["suppressed_no_funds"] = \
                int(self._stats.get("suppressed_no_funds") or 0) + 1
            # This tick read no market, so every per-tick market number must go
            # with it. Leaving last tick's `uncurated`/`unbound_top` standing
            # beside `offers = 0` made the curation queue underreport silently —
            # an operator reads a stale queue as "nothing left to curate", and a
            # suppressed provider is exactly when the numbers stop being refreshed.
            self._reset_market_stats()
            return []
        rep_min = float(settings.get("antseed.reputation_min"))
        allowlist = set(settings.get("antseed.peer_allowlist") or [])
        denylist = set(settings.get("antseed.peer_denylist") or [])
        uncurated = 0
        rejected_by_buyer = 0
        rejected_by_reputation = 0
        denied = 0
        # unbound wire name -> the distinct peers selling it (the curation queue)
        unbound: dict[str, set[str]] = {}
        # family -> rows, one per advertising peer
        by_family: dict[str, list[dict]] = {}
        for row in self._load_market():
            if pinned and row["peer_id"] != pinned:
                continue
            # Operator allow/deny by peer id. Deny wins; a non-empty allowlist
            # restricts to its members. Empty/empty (default) = no change.
            if row["peer_id"] in denylist or (allowlist and row["peer_id"] not in allowlist):
                denied += 1
                continue
            if rep_min > 0 and row.get("reputation") is not None \
                    and row["reputation"] < rep_min:
                # Operator-set floor on the peer's on-chain reputation. A peer
                # that reports NO reputation is kept (cold-start safe); only a
                # known-and-below-floor score is dropped. reputation_min = 0
                # (default) is off → no behaviour change.
                rejected_by_reputation += 1
                continue
            if row["price_in"] < 0 or row["price_out"] < 0:
                # A negative advertised price is bogus (a buggy/hostile peer or a
                # sentinel) — it would win every cost-led policy and bill negative.
                # Free ($0) services stay routable.
                continue
            if row["price_in"] > cap_in or row["price_out"] > cap_out:
                continue
            ci = row.get("price_cached_in")
            if ci is not None and ci > row["price_in"]:
                # The buyer's @antseed/router-local treats an offer whose
                # cached-input price exceeds its input price as malformed
                # (_isValidOffer requires cachedInput <= input) and refuses to
                # route to it — the proxy then answers 502 "…is outside your
                # buyer routing policy". Advertising it anyway pins a candidate
                # the buyer rejects, wasting a route (and, for a single-seller
                # family, killing it). Drop it to mirror the buyer's admission.
                rejected_by_buyer += 1
                continue
            bound = self._bind(cfg, row["service"])
            if bound is None:
                # expose every advertised service, not only curated ones.
                bound = {"family": row["service"], "base_family": None,
                         "variant": None}
                uncurated += 1
                unbound.setdefault(row["service"], set()).add(row["peer_id"])
            by_family.setdefault(bound["family"], []).append({**row, **bound})
        # Surface the OFFERS_TOP_N cheapest *distinct peers* per family as separate
        # routable offers (not just the single cheapest), so the router can rotate
        # to another seller via next_candidate when the cheapest is broken.
        top_n = settings.get("antseed.offers_top_n")
        kept_rows: list[dict] = []
        for rows in by_family.values():
            rows.sort(key=lambda r: (r["price_in"], r["price_out"]))
            seen_peers: set[str] = set()
            for r in rows:
                if r["peer_id"] in seen_peers:
                    continue
                seen_peers.add(r["peer_id"])
                kept_rows.append(r)
                if len(seen_peers) >= top_n:
                    break
        self._stats["uncurated"] = uncurated
        self._stats["rejected_by_buyer"] = rejected_by_buyer
        self._stats["rejected_by_reputation"] = rejected_by_reputation
        self._stats["denied"] = denied
        self._stats["offers"] = len(kept_rows)
        # WHICH names are unbound, ranked — a count alone can't be curated against.
        self._stats["unbound_top"] = self._unbound_top(unbound)
        # #4a/#4c: reliability + latency + learned tool-incapability are derived on
        # the fly from route_observations (one query each per offers_sync, not per
        # candidate), keyed by route identity.
        stats = host_store.route_stats()
        incapable = host_store.tool_incapable_routes()
        offers = []
        for row in kept_rows:
            family = row["family"]
            # Curated family, curated BASE of an `@variant`, or — for a genuinely
            # uncurated name — whatever the live trait oracle can prove. Never a
            # guess: an unknown context stays unset (see _model_meta).
            meta = self._model_meta(row)
            rkey = _route_reliability.route_key(provider_id, family, row["peer_id"])
            rstat = stats.get(rkey) or {}
            # AntSeed rows carry no capability data, so supports_tools defaults to
            # true (else meets_req filters the whole peer market out of any tools
            # request). The default-true HOLE — a peer that accepts `tools` but
            # never function-calls returns a SILENT tools-less answer (no error,
            # no retry) — is closed by the LEARNED per-route signal: a route
            # observed to ignore tools (host_store.tool_incapable_routes) is dropped
            # from supports_tools, so meets_req filters it for tool requests while it
            # still serves non-tool requests. The learned-incapable verdict overrides
            # even a curated claim (the peer is the ground truth); everything else
            # (json_mode, curated caps) is unchanged.
            caps = {"supports_json_mode": True, **meta["capabilities"]}
            if rkey not in incapable:
                caps.setdefault("supports_tools", True)
            else:
                caps.pop("supports_tools", None)
            offers.append({
                "model_family": family,
                # the curated family this offer resolved to — `model_family`
                # itself for a plain match, the base for a `<base>@<variant>`, and
                # None when nothing curated matched. Stamped so a policy can opt
                # into variants later without re-deriving the mapping.
                "base_family": row.get("base_family"),
                "quality_hint": meta["quality_hint"],
                # live model traits for an uncurated family, read pointwise by the
                # algebra's mfield fallback (config.live.lua). None when nothing
                # authoritative was found — never a placeholder.
                "traits": meta["traits"],
                "wire_model_id": row["service"],
                "seller_endpoint": cfg.get("base_url"),
                "price_in_usd_per_mtok": row["price_in"],
                "price_out_usd_per_mtok": row["price_out"],
                "est_tok_s": None,
                "capabilities": caps,
                # the browse-mode buyer disables auto-selection; the host pins
                # this exact peer per request (x-antseed-pin-peer) at call time.
                "peer_id": row["peer_id"],
                # seller in-flight cap, gated host-side per peer to avoid 429s.
                "max_concurrency": row.get("max_concurrency"),
                # peer's on-chain reputation (0-100), stamped on the offer and
                # read pointwise by the algebra as `field reputation_score`
                # (config.live.lua). None when unreported -> field default.
                "reputation_score": row.get("reputation"),
                # host-measured reliability for THIS route, stamped like price and
                # read pointwise by the algebra (offer.success_rate, llm-router
                # #14). None until observed -> algebra default/engine fallback.
                "success_rate": rstat.get("success_rate"),
                # host-measured latency for THIS route, stamped like success_rate
                # and read pointwise by the algebra (offer.latency_ms). None until
                # observed -> field default (optimistically routable, learns down
                # on its first slow call). Lets a policy route by speed.
                "latency_ms": rstat.get("latency_ms"),
            })
        return offers

    def snapshot_stats(self) -> dict:
        # Build the family index if no poll has yet, so the ambiguity audit (a
        # family silently unbound by a catalog name collision) is always reported,
        # not only once the market has been read.
        self._canon_models()
        return dict(self._stats)

    def _refresh_wallet_health(self) -> str:
        """Classify the buyer's serving health from the last few per-attempt
        observations, and warn once per transition when the provider is WEDGED:
        offers were ranked (the router did hand out antseed routes) yet the last
        WEDGE_CONSECUTIVE_FAILURES attempts all failed. That is the signature of
        the 402 loop the funds tourniquet suppresses — if it shows up anyway the
        cause is something the balance gate cannot see (a dead proxy, a wedged
        peer, a stale pin), so it deserves an operator-visible flag rather than a
        silent zero-success day.

        Runs off the request path (from `balances()`, once per refresh tick), not
        inside offers_sync — the rank hot path takes no extra query for it."""
        if not self._stats.get("offers"):
            # Nothing was offered (suppressed, stale market, or genuinely idle):
            # failures cannot be blamed on us handing out unpayable routes.
            health = "suppressed" if self._stats.get("suppressed_no_funds") else "idle"
        else:
            recent = host_store.provider_recent_ok(
                self.name, limit=WEDGE_CONSECUTIVE_FAILURES)
            if len(recent) >= WEDGE_CONSECUTIVE_FAILURES and not any(recent):
                health = "wedged"
            elif not recent:
                health = "unknown"
            else:
                health = "ok"
        if health == "wedged" and self._stats.get("wallet_health") != "wedged":
            _log.warning(
                "antseed wedged: last %d routed attempts all failed while offers "
                "were ranked (deposits_available=%s) — check escrow, the buyer "
                "proxy and the session pin",
                WEDGE_CONSECUTIVE_FAILURES, self._stats.get("deposits_available"))
        self._stats["wallet_health"] = health
        return health

    # ---- full-market book (dashboard only) --------------------------------

    BOOK_TOP_N = 3

    def market_book(self) -> dict:
        """Read-only full-market view for the dashboard: per curated family,
        the BOOK_TOP_N cheapest peer rows plus every pinned-peer row (the
        pinned peer is what the router can actually call, so it's always
        shown even when it isn't among the cheapest). Never feeds ranking."""
        pinned: dict[str, list[str]] = {}
        for pid in self.provider_ids:
            peer = self._pinned_peer(pid)
            if peer:
                pinned.setdefault(peer, []).append(pid)

        by_family: dict[str, list[dict]] = {}
        for row in self._load_market():
            family = None
            for cfg in self._providers.values():
                family = self._family_for(cfg, row["service"])
                if family:
                    break
            # uncurated services are shown under their raw wire name, not hidden
            if family is None:
                family = row["service"]
            by_family.setdefault(family, []).append(row)

        rows_out: list[dict] = []
        families: dict[str, dict] = {}
        for family, rows in by_family.items():
            rows.sort(key=lambda r: (r["price_in"], r["price_out"]))
            keep, seen = [], set()
            for r in rows:
                key = (r["peer_id"], r["service"])
                if key in seen:
                    continue
                if len(keep) >= self.BOOK_TOP_N and r["peer_id"] not in pinned:
                    continue
                seen.add(key)
                keep.append(r)
            families[family] = {"sellers_total": len({
                (r["peer_id"], r["service"]) for r in rows})}
            for r in keep:
                tradable_via = []
                for pid in pinned.get(r["peer_id"], []):
                    cap = self._providers[pid].get("market_price_cap") or {}
                    if (r["price_in"] <= float(cap.get("input", float("inf")))
                            and r["price_out"] <= float(cap.get("output", float("inf")))):
                        tradable_via.append(pid)
                rows_out.append({
                    "model_family": family,
                    "seller": r["peer_id"],
                    "wire_model_id": r["service"],
                    "price_in": r["price_in"],
                    "price_out": r["price_out"],
                    "last_seen": r.get("last_seen"),
                    "pinned_by": pinned.get(r["peer_id"], []),
                    "tradable_via": tradable_via,
                })
        return {"rows": rows_out, "families": families,
                "fetched_at": int(time.time())}

    # ---- ProviderSource capabilities -------------------------------------

    async def pricing(self) -> list[Price]:
        prices: list[Price] = []
        for pid in self.provider_ids:
            for o in self.offers_sync(pid):
                prices.append({
                    "provider_id": pid,
                    "served_model_id": o["wire_model_id"],
                    "model_family": o["model_family"],
                    "price_in_usd_per_mtok": o["price_in_usd_per_mtok"],
                    "price_out_usd_per_mtok": o["price_out_usd_per_mtok"],
                })
        return prices

    def credits_seed(self) -> dict[str, float]:
        """{provider_id -> last-known spendable escrow}, straight from the durable
        buyer_status row — no network, no refresh tick.

        Published into the engine's `__credits|<pid>` slot at STARTUP, before the
        app serves traffic. The host envelope gates antseed on `credits >= 1.0`
        and the engine's `credits` field defaults to 0, so that clause fails
        CLOSED: a fresh pod that has not yet completed a balances refresh would
        reject every antseed candidate until the first tick (up to 300s). Seeding
        from the last known status closes that hole without weakening the gate."""
        out: dict[str, float] = {}
        for pid in self.provider_ids:
            available = as_float((host_store.buyer_status(pid) or {})
                                 .get("deposits_available"))
            if available is not None:
                out[pid] = available
        return out

    async def balances(self) -> dict[str, Balance]:
        out: dict[str, Balance] = {}
        # Off the request path and after this tick's pricing()/offers_sync, so the
        # offer counts it reads are current.
        self._refresh_wallet_health()
        for pid in self.provider_ids:
            data = host_store.buyer_status(pid)
            if not data:
                continue
            try:
                available = float(data.get("deposits_available"))
            except (TypeError, ValueError):
                continue
            detail = {"reserved": data.get("deposits_reserved"),
                      "wallet": data.get("wallet_address"),
                      "connection": data.get("connection_state")}
            rpc = _wallet_rpc_url()
            addr = data.get("wallet_address")
            if rpc and addr:
                detail.update(await _fetch_chain_balances(rpc, addr))
            out[pid] = {
                "kind": "deposits_usdc",
                "value": available,
                "detail": detail,
                "fetched_at": int(time.time()),
            }
        return out
