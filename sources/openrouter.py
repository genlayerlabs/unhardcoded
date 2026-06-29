"""
OpenRouter source: live model pricing (GET /models — public, per-token USD
strings) and account credits (GET /credits — needs OPENROUTER_API_KEY).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import host_store
import route_reliability as _route_reliability
from sources import Balance, Price

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterSource:
    name = "openrouter"
    # "openrouter" = the static, curated provider (benchmark-ranked families);
    # "openrouter_market" = live discovery of the WHOLE OpenRouter catalog (the
    # long tail), wired through the core discover hook. One source feeds both.
    provider_ids = ["openrouter", "openrouter_market"]
    poll_interval_s = 3600

    def __init__(self, catalog: dict, env_get=os.environ.get,
                 client: Any = None, base_url: str = BASE_URL):
        self._env_get = env_get
        self._base_url = base_url.rstrip("/")
        self._client = client  # injected in tests; lazy httpx otherwise
        # /models snapshot cached by the async pricing() refresh so the SYNC
        # discover hook (offers_sync, called inside rank) never blocks on HTTP.
        self._models_snapshot: list[dict] = []
        # model id -> endpoint availability. A missing entry means "unknown" and
        # stays routable; a present false means OpenRouter explicitly reported no
        # usable endpoint for that model.
        self._endpoint_availability: dict[str, dict] = {}
        self._live_traits: dict[str, dict] = {}  # raw model id -> full traits (+ranks)
        # Discovery offers built ONCE per refresh (in pricing()) and served by the
        # sync hook — offers_sync runs inside rank, so it must not rebuild the
        # whole catalog per call. Empty until the first refresh populates it.
        self._offers: list[dict] = []
        providers = catalog.get("providers") or {}
        market_cfg = providers.get("openrouter_market") or {}
        self._market_aliases: dict[str, str] = {
            str(raw): str(family)
            for aliases in (
                market_cfg.get("service_aliases") or {},
                market_cfg.get("model_aliases") or {},
            )
            for raw, family in aliases.items()
            if raw and family
        }
        # provider_model_id -> curated family, from the catalog's served_by.
        # Static curated routes keep the provider id `openrouter`; dynamic
        # marketplace routes use `openrouter_market` and may alias raw model ids
        # to policy-facing families below.
        self._family_by_id: dict[str, str] = {}
        for family, model in (catalog.get("models") or {}).items():
            for served in model.get("served_by") or []:
                if served.get("provider") == "openrouter" and served.get("provider_model_id"):
                    self._family_by_id[served["provider_model_id"]] = family

    def _url_for(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        # OpenRouter embeds docs/API paths such as
        # /api/v1/models/<slug>/endpoints in links.details. The source base URL
        # already ends with /api/v1, so normalize those links before appending.
        if path.startswith("/api/v1/") and self._base_url.endswith("/api/v1"):
            path = path[len("/api/v1"):]
        return self._base_url + path

    async def _get(self, path: str, headers: dict | None = None):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=15.0)
        resp = await self._client.get(self._url_for(path), headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"openrouter GET {path} -> {resp.status_code}")
        return resp.json()

    @staticmethod
    def _endpoint_usable(endpoint: dict) -> bool:
        status = endpoint.get("status")
        # Observed OpenRouter shape: status=0 is live. If the field is absent,
        # keep the endpoint because old/partial responses should not falsely
        # remove a model. Non-zero statuses are treated as unavailable.
        if status is None:
            return True
        if isinstance(status, str):
            return status.strip().lower() in {"0", "live", "ok"}
        return status == 0

    async def _endpoint_availability_for(self, model: dict) -> tuple[str, dict | None]:
        mid = model.get("id")
        if not mid:
            return "", None
        details = ((model.get("links") or {}).get("details") or "").strip()
        if not details:
            return mid, None
        try:
            body = await self._get(details)
        except Exception:  # noqa: BLE001 — endpoint detail is advisory
            return mid, None
        data = body.get("data") if isinstance(body, dict) else None
        endpoints = (data or {}).get("endpoints") or []
        usable = [e for e in endpoints if isinstance(e, dict) and self._endpoint_usable(e)]
        return mid, {
            "available": bool(usable),
            "endpoints_total": len(endpoints),
            "endpoints_usable": len(usable),
            "provider_tags": [e.get("tag") for e in usable if e.get("tag")],
        }

    async def _refresh_endpoint_availability(self, models: list[dict]) -> None:
        limit_raw = self._env_get("OPENROUTER_ENDPOINTS_CONCURRENCY") or "8"
        try:
            limit = max(1, min(32, int(limit_raw)))
        except ValueError:
            limit = 8
        sem = asyncio.Semaphore(limit)

        async def one(model: dict):
            async with sem:
                return await self._endpoint_availability_for(model)

        availability: dict[str, dict] = {}
        for mid, info in await asyncio.gather(*(one(m) for m in models)):
            if mid and info is not None:
                availability[mid] = info
        self._endpoint_availability = availability

    def _model_available(self, model_id: str) -> bool:
        info = self._endpoint_availability.get(model_id)
        return True if info is None else bool(info.get("available"))

    async def pricing(self) -> list[Price]:
        body = await self._get("/models")
        # cache for the sync offers_sync()/market_book() (whole-catalog discovery)
        self._models_snapshot = body.get("data") or []
        await self._refresh_endpoint_availability(self._models_snapshot)
        # Live, full model-level traits (benchmarks/modalities/caps + ranks) for
        # EVERY model, keyed by raw id, ranked across the whole OpenRouter
        # catalog. Discovered families carry these inline so they rank on real
        # benchmarks (not just price) — see _offer_for + config.live.lua mfield.
        live = {m["id"]: self._traits_for(m) for m in self._models_snapshot if m.get("id")}
        self._add_ranks(live)
        self._live_traits = live
        # Build the discovery offers once, off the fresh snapshot/traits, so the
        # sync hook (offers_sync) and the dashboard book (market_book) just read
        # this list instead of rebuilding it per call.
        self._offers = [o for o in (self._offer_for(m) for m in self._models_snapshot)
                        if o is not None]
        prices: list[Price] = []
        for m in self._models_snapshot:
            mid = m.get("id")
            if mid and not self._model_available(mid):
                continue
            pricing = m.get("pricing") or {}
            try:
                price_in = float(pricing.get("prompt")) * 1e6
                price_out = float(pricing.get("completion")) * 1e6
            except (TypeError, ValueError):
                continue  # non-numeric pricing (some modalities) — skip
            if price_in < 0 or price_out < 0:
                continue  # negative = unpriced/variable sentinel, not a real price
            prices.append({
                "provider_id": "openrouter",
                "served_model_id": mid,
                "model_family": self._family_by_id.get(mid),
                "price_in_usd_per_mtok": price_in,
                "price_out_usd_per_mtok": price_out,
            })
        return prices

    def _offer_for(self, m: dict) -> dict | None:
        """An offer for one live OpenRouter model, or None when it's a curated
        family (already served by the static `openrouter` provider — skip to
        avoid a duplicate candidate) or has non-numeric pricing."""
        mid = m.get("id")
        if not mid or mid in self._family_by_id:
            return None
        if not self._model_available(mid):
            return None
        pricing = m.get("pricing") or {}
        try:
            price_in = float(pricing.get("prompt")) * 1e6
            price_out = float(pricing.get("completion")) * 1e6
        except (TypeError, ValueError):
            return None  # non-numeric pricing (some modalities / BYO) — not routable here
        if price_in < 0 or price_out < 0:
            # A negative per-token price is not a real price — it is OpenRouter's
            # "unpriced / variable" sentinel (e.g. "-1"). Admitting it would make
            # the candidate WIN every cost-led policy (most-negative = "cheapest")
            # and bill a negative cost. Free models ($0) stay routable.
            return None
        ctx = m.get("context_length") or (m.get("top_provider") or {}).get("context_length")
        caps = {"context": int(ctx)} if isinstance(ctx, (int, float)) and ctx else {}
        # Project the model's capability TRAITS onto the flags the core's
        # meets_req filters on (capabilities.supports_*). _traits_for is the SINGLE
        # place that reads supported_parameters/modalities; deriving the supports_*
        # flags from those traits (not re-parsing the model) keeps one source of
        # truth. Without these under `capabilities`, a request carrying tools (or
        # images / json_object) auto-derives a need the discovered candidate can't
        # satisfy → filtered out, making the whole live long tail unroutable.
        traits = self._live_traits.get(mid) or {}
        if traits.get("cap_tools"):
            caps["supports_tools"] = True
        if traits.get("cap_response_format") or traits.get("cap_structured_outputs"):
            caps["supports_json_mode"] = True
        if traits.get("cap_seed"):
            caps["supports_seed"] = True
        if traits.get("in_image"):
            caps["supports_vision"] = True
        family = self._market_family_for(mid)
        return {
            # The policy-facing family is provider-agnostic by default:
            # `openai/gpt-5-mini` -> `gpt-5-mini`. Exact provider-local aliases
            # handle cases where the OpenRouter tail is not the canonical family
            # we want, while wire_model_id preserves the actual provider slug.
            # Either way this is not a second-class "uncurated" row: it carries
            # full live model_meta inline as `traits`, so it ranks on real
            # benchmark just like a curated family.
            "model_family": family,
            "wire_model_id": mid,
            "seller_endpoint": self._base_url,
            "price_in_usd_per_mtok": price_in,
            "price_out_usd_per_mtok": price_out,
            "capabilities": caps,
            "traits": traits,
            "est_tok_s": None,
            "quality_hint": None,
        }

    def _market_family_for(self, model_id: str) -> str:
        """Policy-facing family for an OpenRouter marketplace model.

        OpenRouter ids are provider-scoped (`vendor/model`). Router policies are
        provider-agnostic, so the default family is the model part. Exact aliases
        override this only for canonicalization exceptions such as dated or
        provider-specific suffixes.
        """
        alias = self._market_aliases.get(model_id)
        if alias:
            return alias
        return model_id.rsplit("/", 1)[-1]

    def offers_sync(self, provider_id: str) -> list[dict]:
        """Whole live OpenRouter catalog as marketplace offers (the long tail
        beyond the curated families). Read from the cached /models snapshot the
        async pricing() refresh populates — sync and fast, called from the core
        discover hook inside rank. No peer pinning / concurrency cap: OpenRouter
        is a first-party gateway, so offers omit peer_id/max_concurrency and the
        host calls it directly at seller_endpoint with the OPENROUTER_API_KEY.

        Stamps host-measured latency (offer.latency_ms) per family so a policy can
        route by speed across sources. A gateway route carries no peer_id, so its
        latency is keyed on the provider itself (peer == provider_id) — the same
        key the latency fold uses for peerless routes — making OpenRouter speed
        directly comparable to a marketplace peer's."""
        # #4a: latency derived on the fly from route_observations (one query),
        # keyed on the provider itself for these peerless gateway routes.
        stats = host_store.route_stats()
        out = []
        for o in self._offers:
            lkey = _route_reliability.route_key(provider_id, o["model_family"], provider_id)
            out.append({**o, "latency_ms": (stats.get(lkey) or {}).get("latency_ms")})
        return out

    def market_book(self) -> dict:
        """Read-only full list of the live (uncurated) OpenRouter models for the
        dashboard Catalog. Curated families show via their own static rows; this
        covers everything else. Never feeds ranking. Shape mirrors the antseed
        book the shim's /x/market consumes (source-tagged rows)."""
        rows, families = [], {}
        for o in self._offers:
            fam = o["model_family"]
            rows.append({
                "model_family": fam,
                "source": "openrouter",
                "seller": "openrouter",
                "wire_model_id": o["wire_model_id"],
                "price_in": o["price_in_usd_per_mtok"],
                "price_out": o["price_out_usd_per_mtok"],
                "context": (o.get("capabilities") or {}).get("context"),
                "tradable": True,
                "via": "openrouter",
            })
            # full live traits surfaced to the dashboard Catalog as the family meta
            families[fam] = {"sellers_total": 1, "meta": o.get("traits") or {}}
        return {"rows": rows, "families": families, "fetched_at": int(time.time())}

    # Model-level traits (same whoever serves the family): benchmarks,
    # modalities, capabilities. Written to a registered model_meta file that the
    # host config reads — NOT live state. Provider-level pricing/caching flow
    # live via pricing()/EMA instead. Keyed by curated family.
    _PARAM_CAPS = {
        "cap_tools": "tools", "cap_tool_choice": "tool_choice",
        "cap_parallel_tools": "parallel_tool_calls",
        "cap_structured_outputs": "structured_outputs",
        "cap_response_format": "response_format",
        "cap_seed": "seed", "cap_logprobs": "logprobs",
    }

    def _traits_for(self, m: dict) -> dict:
        """Full model-level traits for one /models entry: benchmarks (0..1),
        input/output modalities, and capability flags. No ranks (those are
        population-relative — added by _add_ranks over a whole set)."""
        traits: dict = {}
        bm = m.get("benchmarks") or {}
        aa = bm.get("artificial_analysis") or {}
        for key, src in (("bench_intelligence", "intelligence_index"),
                         ("bench_coding", "coding_index"),
                         ("bench_agentic", "agentic_index")):
            v = aa.get(src)
            if isinstance(v, (int, float)):
                traits[key] = max(0.0, min(1.0, v / 100.0))
        wins = [a.get("win_rate") for a in (bm.get("design_arena") or [])
                if isinstance(a.get("win_rate"), (int, float))]
        if wins:
            traits["bench_arena"] = max(0.0, min(1.0, max(wins) / 100.0))
        arch = m.get("architecture") or {}
        im = set(arch.get("input_modalities") or [])
        for mod in ("image", "audio", "file", "video"):
            traits["in_" + mod] = mod in im
        traits["out_image"] = "image" in set(arch.get("output_modalities") or [])
        sp = set(m.get("supported_parameters") or [])
        for cap, param in self._PARAM_CAPS.items():
            traits[cap] = param in sp
        traits["cap_reasoning"] = bool(sp & {"reasoning", "include_reasoning"})
        return traits

    @staticmethod
    def _add_ranks(meta: dict[str, dict]) -> None:
        """Per-benchmark catalog ranks (1 = best), in place. "top-k by a
        benchmark" is a FIELD the policy gates on (cmp(<bench>_rank, le, k)).
        Families lacking a benchmark get no rank (field default keeps them out
        of any top-k)."""
        for metric in ("bench_intelligence", "bench_coding",
                       "bench_agentic", "bench_arena"):
            ranked = sorted((f for f in meta if metric in meta[f]),
                            key=lambda f: meta[f][metric], reverse=True)
            for rank, f in enumerate(ranked, start=1):
                meta[f][metric + "_rank"] = rank

    async def model_meta(self) -> dict:
        # The REGISTERED (static, deterministic) meta for curated families, keyed
        # by curated family — written to model_meta.lua for the on-chain path.
        # Discovered families instead carry live traits inline on their offer.
        body = await self._get("/models")
        out: dict[str, dict] = {}
        for m in body.get("data") or []:
            family = self._family_by_id.get(m.get("id"))
            if family:
                out[family] = self._traits_for(m)
        self._add_ranks(out)
        return out

    async def balances(self) -> dict[str, Balance]:
        key = self._env_get("OPENROUTER_API_KEY")
        if not key:
            return {}
        body = await self._get("/credits", headers={"Authorization": f"Bearer {key}"})
        data = body.get("data") or {}
        remaining = float(data.get("total_credits") or 0) - float(data.get("total_usage") or 0)
        return {"openrouter": {
            "kind": "credits_usd",
            "value": remaining,
            "detail": {"total_credits": data.get("total_credits"),
                       "total_usage": data.get("total_usage")},
            "fetched_at": int(time.time()),
        }}
