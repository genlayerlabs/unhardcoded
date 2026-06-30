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

import os
import time
from typing import Any

import host_store
import route_reliability as _route_reliability
import settings
from sources import Balance, Price

STALE_AFTER_S = 900

# Vendor prefixes stripped (as a whole leading token) when canonicalizing a
# marketplace service name to match curated families. `claude-` is included so a
# curated `claude-opus-4-8` and a peer's bare `opus-4.8` reduce to the same form.
_VENDOR_PREFIXES = ("anthropic-", "claude-", "openai-", "google-", "gemini-",
                    "meta-", "qwen-", "x-ai-", "deepseek-")


def _canon_service(name: str) -> str:
    """Canonical form for matching a peer's wire model name to a curated family:
    lowercase, dots->dashes, and strip ONE leading known vendor prefix. Kept
    deliberately conservative — it bridges `opus-4.8` / `anthropic-claude-opus-4.8`
    to `claude-opus-4-8`, but does NOT merge digit-run differences (`gpt-55` vs
    `gpt-5.5`) or model variants (`...-fast`), so it cannot collapse two distinct
    models into one route."""
    s = (name or "").strip().lower().replace(".", "-")
    for vp in _VENDOR_PREFIXES:
        if s.startswith(vp):
            return s[len(vp):]
    return s

# Buyer hot-wallet on-chain reads. The marketplace spends from ESCROW
# (depositsAvailable); the raw wallet balance — USDC sitting in the wallet, plus
# ETH for gas — is what tells you whether you can deposit more or pay for a tx at
# all. The buyer CLI/status file expose neither, so we read them straight from
# Base. Native (Circle) USDC on Base mainnet, 6 decimals.
_BASE_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
_USDC_DECIMALS = 6
_DEFAULT_BASE_RPC = "https://mainnet.base.org"


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
        self._stats: dict[str, Any] = {"stale": False, "dropped_unmapped": 0}

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

    def _canon_models(self) -> dict[str, str]:
        """Lazy {canonical_name -> curated family} index over self._models, so a
        peer that names a curated model with a different vendor prefix / separator
        (e.g. `opus-4.8`, `anthropic-claude-sonnet-4.6`) still folds into the
        curated family instead of being exposed under its raw wire name. A
        canonical form shared by TWO curated families is AMBIGUOUS and dropped —
        never risk routing to the wrong model; the offer falls through to raw."""
        cached = getattr(self, "_canon_models_cache", None)
        if cached is None:
            index: dict[str, str] = {}
            ambiguous: set[str] = set()
            for fam in self._models:
                c = _canon_service(fam)
                if c in index and index[c] != fam:
                    ambiguous.add(c)
                index[c] = fam
            for c in ambiguous:
                index.pop(c, None)
            cached = self._canon_models_cache = index
        return cached

    def _family_for(self, provider_cfg: dict, service: str) -> str | None:
        # Exact wire name and the operator's static aliases are authoritative.
        aliases = provider_cfg.get("service_aliases") or {}
        family = aliases.get(service, service)
        if family in self._models:
            return family
        # Conservative canonical match (vendor-prefix + separator/case only): a
        # peer's `opus-4.8` reaches the curated `claude-opus-4-8`. None on no/
        # ambiguous match, so the caller keeps exposing the service raw.
        return self._canon_models().get(_canon_service(service))

    def offers_sync(self, provider_id: str) -> list[dict]:
        """One offer per advertised service for this buyer proxy — the WHOLE
        market, not just curated families. A service that maps to a curated
        family carries that family's benchmark/capabilities; every other
        service is exposed under its raw wire name (no benchmark → it scores on
        price + learned latency, never dropped). Sync: called from the core's
        discover hook inside rank."""
        cfg = self._providers.get(provider_id)
        if cfg is None:
            return []
        cap = cfg.get("market_price_cap") or {}
        cap_in = float(cap.get("input", float("inf")))
        cap_out = float(cap.get("output", float("inf")))
        pinned = self._pinned_peer(provider_id)
        rep_min = float(settings.get("antseed.reputation_min"))
        allowlist = set(settings.get("antseed.peer_allowlist") or [])
        denylist = set(settings.get("antseed.peer_denylist") or [])
        uncurated = 0
        rejected_by_buyer = 0
        rejected_by_reputation = 0
        denied = 0
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
            family = self._family_for(cfg, row["service"])
            if family is None:
                # expose every advertised service, not only curated ones.
                family = row["service"]
                uncurated += 1
            by_family.setdefault(family, []).append({**row, "family": family})
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
        self._stats["dropped_unmapped"] = 0
        self._stats["uncurated"] = uncurated
        self._stats["rejected_by_buyer"] = rejected_by_buyer
        self._stats["rejected_by_reputation"] = rejected_by_reputation
        self._stats["denied"] = denied
        self._stats["offers"] = len(kept_rows)
        # #4a/#4c: reliability + latency + learned tool-incapability are derived on
        # the fly from route_observations (one query each per offers_sync, not per
        # candidate), keyed by route identity.
        stats = host_store.route_stats()
        incapable = host_store.tool_incapable_routes()
        offers = []
        for row in kept_rows:
            family = row["family"]
            model = self._models.get(family) or {}
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
            caps = {"supports_json_mode": True, **(model.get("capabilities") or {})}
            if rkey not in incapable:
                caps.setdefault("supports_tools", True)
            else:
                caps.pop("supports_tools", None)
            offers.append({
                "model_family": family,
                "quality_hint": model.get("static_quality_hint"),
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
        return dict(self._stats)

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

    async def balances(self) -> dict[str, Balance]:
        out: dict[str, Balance] = {}
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
