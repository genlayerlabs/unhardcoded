"""Amazon Bedrock source: native AWS discovery plus public price ingestion."""
from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Callable

from sources import Balance, Price

AWS_PRICE_BASE = "https://pricing.us-east-1.amazonaws.com"
PRICE_OFFERS = (
    "AmazonBedrock",
    "AmazonBedrockService",
    "AmazonBedrockFoundationModels",
)


_FAMILY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("claude-opus-4-8", ("claude-opus-4-8", "claude opus 4.8")),
    ("claude-sonnet-4-6", ("claude-sonnet-4-6", "claude sonnet 4.6")),
    ("qwen3-235b-a22b", ("qwen3-vl-235b-a22b", "qwen3 235b a22b", "qwen3-235b")),
    ("qwen3-coder-next", ("qwen3-coder-next", "qwen3 coder next")),
    ("llama-4-maverick", ("llama4-maverick", "llama 4 maverick")),
    ("gpt-oss-120b", ("gpt-oss-120b", "gpt oss 120b")),
    ("gpt-oss-20b", ("gpt-oss-20b", "gpt oss 20b")),
    ("deepseek-v3.2", ("deepseek-v3.2", "deepseek v3.2")),
    ("deepseek-r1", ("deepseek.r1", "deepseek-r1", "deepseek r1")),
    ("gemma-3-27b", ("gemma-3-27b", "gemma 3 27b")),
    ("nova-2-lite", ("nova-2-lite", "nova 2 lite", "nova 2.0 lite")),
    ("nova-micro", ("nova-micro", "nova micro")),
    ("nova-pro", ("nova-pro", "nova pro")),
]

_PRICE_EXCLUDE = (
    "reserved",
    "batch",
    "cache",
    "cached",
    "embedding",
    "image",
    "video",
    "fine-tun",
    "custom",
    "provisioned",
    "training",
    "prompt cache",
    "flex",
    "priority",
)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def _region(env_get: Callable[[str], str | None]) -> str:
    return (
        env_get("BEDROCK_REGION")
        or env_get("AWS_REGION")
        or env_get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def _family_from_text(text: str) -> str | None:
    hay = _norm(text)
    for family, patterns in _FAMILY_PATTERNS:
        if any(_norm(p) in hay for p in patterns):
            return family
    return None


def _model_id_from_arn(arn: str) -> str | None:
    marker = "foundation-model/"
    if marker not in arn:
        return None
    return arn.split(marker, 1)[1]


def _is_available(row: dict) -> bool:
    agreement = row.get("agreementAvailability") or {}
    agreement_status = agreement.get("status")
    if agreement_status not in (None, "AVAILABLE"):
        return False
    if row.get("authorizationStatus") not in (None, "AUTHORIZED"):
        return False
    if row.get("entitlementAvailability") not in (None, "AVAILABLE"):
        return False
    if row.get("regionAvailability") not in (None, "AVAILABLE"):
        return False
    return True


class BedrockSource:
    name = "bedrock"
    poll_interval_s = 3600

    def __init__(
        self,
        catalog: dict,
        env_get=os.environ.get,
        client: Any = None,
        pricing_base: str = AWS_PRICE_BASE,
        bedrock_client: Any = None,
        bedrock_client_factory: Callable[[str], Any] | None = None,
    ):
        self._catalog = catalog
        self._env_get = env_get
        self._client = client
        self._pricing_base = pricing_base.rstrip("/")
        self._bedrock_client = bedrock_client
        self._bedrock_client_factory = bedrock_client_factory
        self._models = catalog.get("models") or {}
        self._providers = {
            pid: p for pid, p in (catalog.get("providers") or {}).items()
            if isinstance(p, dict)
            and (p.get("source") == "bedrock" or str(pid).startswith("bedrock"))
        }
        self.provider_ids = list(self._providers)
        self._aliases: dict[str, str] = {}
        for p in self._providers.values():
            for raw, family in (p.get("service_aliases") or {}).items():
                if raw and family:
                    self._aliases[str(raw).lower()] = str(family)
            for raw, family in (p.get("model_aliases") or {}).items():
                if raw and family:
                    self._aliases[str(raw).lower()] = str(family)
        self._static_family_by_id: dict[tuple[str, str], str] = {}
        for family, model in self._models.items():
            for served in model.get("served_by") or []:
                pid = served.get("provider")
                mid = served.get("provider_model_id")
                if pid in self._providers and mid:
                    self._static_family_by_id[(pid, str(mid))] = family
        self._offers_by_provider: dict[str, list[dict]] = {}

    async def _get_json(self, url: str, headers: dict | None = None) -> dict:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=20.0)
        resp = await self._client.get(url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"bedrock source GET {url} -> {resp.status_code}")
        return resp.json()

    def _aws_client(self, region: str):
        if self._bedrock_client is not None:
            return self._bedrock_client
        if self._bedrock_client_factory is not None:
            return self._bedrock_client_factory(region)
        import boto3
        return boto3.client("bedrock", region_name=region)

    async def _bedrock_catalog(self, region: str) -> list[dict]:
        client = self._aws_client(region)
        models_resp, profiles_resp = await asyncio.gather(
            asyncio.to_thread(client.list_foundation_models),
            asyncio.to_thread(client.list_inference_profiles),
        )
        profiles_by_model = self._profile_ids_by_model(
            profiles_resp.get("inferenceProfileSummaries") or [],
            region,
        )
        availability = await self._availability_by_model(
            client,
            [str(m.get("modelId")) for m in models_resp.get("modelSummaries") or []
             if isinstance(m, dict) and m.get("modelId")],
        )
        out = []
        for model in models_resp.get("modelSummaries") or []:
            if not isinstance(model, dict) or not model.get("modelId"):
                continue
            raw_id = str(model["modelId"])
            if raw_id in availability and not _is_available(availability[raw_id]):
                continue
            invoke_id = profiles_by_model.get(raw_id) or raw_id
            out.append({
                **model,
                "id": invoke_id,
                "modelId": raw_id,
                "invocationModelId": invoke_id,
                "inferenceProfileId": profiles_by_model.get(raw_id),
            })
        return out

    async def _availability_by_model(self, client: Any, model_ids: list[str]) -> dict[str, dict]:
        getter = getattr(client, "get_foundation_model_availability", None)
        if getter is None:
            return {}

        async def one(model_id: str) -> tuple[str, dict | None]:
            try:
                return model_id, await asyncio.to_thread(getter, modelId=model_id)
            except Exception:
                # Discovery should not disappear because this optional control
                # API is unavailable or throttled. Invocation errors still fold
                # into route reliability if a model is actually selected.
                return model_id, None

        rows = await asyncio.gather(*(one(mid) for mid in model_ids))
        return {mid: row for mid, row in rows if row is not None}

    @staticmethod
    def _profile_ids_by_model(profiles: list[dict], region: str) -> dict[str, str]:
        choices: dict[str, list[str]] = {}
        for profile in profiles:
            pid = profile.get("inferenceProfileId")
            if not pid or profile.get("status") not in (None, "ACTIVE"):
                continue
            for model in profile.get("models") or []:
                mid = _model_id_from_arn(str(model.get("modelArn") or ""))
                if mid:
                    choices.setdefault(mid, []).append(str(pid))

        preferred_prefix = region.split("-", 1)[0] if "-" in region else region
        out = {}
        for mid, ids in choices.items():
            def rank(pid: str) -> tuple[int, str]:
                if pid.startswith(region + "."):
                    return (0, pid)
                if pid.startswith(preferred_prefix + "."):
                    return (1, pid)
                if pid.startswith("global."):
                    return (2, pid)
                return (3, pid)
            out[mid] = sorted(ids, key=rank)[0]
        return out

    async def _price_region_url(self, offer_code: str, region: str) -> str:
        idx = await self._get_json(
            f"{self._pricing_base}/offers/v1.0/aws/{offer_code}/current/region_index.json")
        region_row = (idx.get("regions") or {}).get(region)
        if not region_row or not region_row.get("currentVersionUrl"):
            raise RuntimeError(f"Bedrock price region {region!r} not found for {offer_code}")
        return self._pricing_base + str(region_row["currentVersionUrl"])

    @staticmethod
    def _dimension_kind(attrs: dict, description: str) -> str | None:
        text = " ".join(str(x or "") for x in (
            attrs.get("usagetype"),
            attrs.get("inferenceType"),
            attrs.get("tokenType"),
            attrs.get("feature"),
            attrs.get("service_tier"),
            description,
        )).lower()
        if any(x in text for x in _PRICE_EXCLUDE):
            return None
        if "input" in text and "token" in text:
            return "input"
        if ("output" in text or "response" in text) and "token" in text:
            return "output"
        return None

    @staticmethod
    def _price_per_mtok(pd: dict) -> float | None:
        try:
            price = float((pd.get("pricePerUnit") or {}).get("USD"))
        except (TypeError, ValueError):
            return None
        unit = str(pd.get("unit") or "").lower()
        desc = str(pd.get("description") or "").lower()
        if "1k" in unit or "1k token" in desc or "per 1k" in desc:
            return price * 1000.0
        return price

    @staticmethod
    def _price_family(attrs: dict) -> str | None:
        return _family_from_text(" ".join(str(x or "") for x in (
            attrs.get("model"),
            attrs.get("servicename"),
            attrs.get("usagetype"),
        )))

    async def _prices_by_family(self, region: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for offer_code in PRICE_OFFERS:
            data = await self._get_json(await self._price_region_url(offer_code, region))
            terms = data.get("terms", {}).get("OnDemand", {})
            for sku, product in (data.get("products") or {}).items():
                attrs = product.get("attributes") or {}
                family = self._price_family(attrs)
                if not family:
                    continue
                for term in (terms.get(sku) or {}).values():
                    for pd in (term.get("priceDimensions") or {}).values():
                        kind = self._dimension_kind(attrs, str(pd.get("description") or ""))
                        if not kind:
                            continue
                        price = self._price_per_mtok(pd)
                        if price is None:
                            continue
                        slot = out.setdefault(family, {})
                        slot[kind] = min(slot.get(kind, price), price)
        return {
            fam: {"input": p["input"], "output": p["output"]}
            for fam, p in out.items()
            if p.get("input") is not None and p.get("output") is not None
        }

    def _family_for_model(self, provider_id: str, model_id: str, model: dict | None = None) -> str | None:
        model_id = str(model_id)
        if (provider_id, model_id) in self._static_family_by_id:
            return self._static_family_by_id[(provider_id, model_id)]
        raw_id = str((model or {}).get("modelId") or "")
        if (provider_id, raw_id) in self._static_family_by_id:
            return self._static_family_by_id[(provider_id, raw_id)]
        low = model_id.lower()
        if low in self._aliases:
            return self._aliases[low]
        raw_low = raw_id.lower()
        if raw_low in self._aliases:
            return self._aliases[raw_low]
        return _family_from_text(" ".join(str(x or "") for x in (
            model_id,
            raw_id,
            (model or {}).get("modelName"),
            (model or {}).get("providerName"),
        )))

    def _capabilities_for(self, family: str, model: dict) -> dict:
        caps = dict((self._models.get(family) or {}).get("capabilities") or {})
        ctx = model.get("context_length") or model.get("contextLength")
        if isinstance(ctx, (int, float)) and ctx:
            caps["context"] = int(ctx)
        # Bedrock model summaries do not consistently advertise tool/json
        # support. Curated catalog capabilities remain the source of truth.
        return caps

    async def pricing(self) -> list[Price]:
        region = _region(self._env_get)
        if not self._providers:
            self._offers_by_provider = {}
            return []

        models, prices = await asyncio.gather(
            self._bedrock_catalog(region),
            self._prices_by_family(region),
        )

        rows: list[Price] = []
        offers_by_provider: dict[str, list[dict]] = {}
        for pid, cfg in self._providers.items():
            provider_offers: list[dict] = []
            for model in models:
                mid = str(model.get("id") or model.get("modelId") or "")
                if not mid:
                    continue
                family = self._family_for_model(pid, mid, model)
                price = prices.get(family or "")
                if not family or not price:
                    continue
                row = {
                    "provider_id": pid,
                    "served_model_id": mid,
                    "model_family": family,
                    "price_in_usd_per_mtok": price["input"],
                    "price_out_usd_per_mtok": price["output"],
                }
                rows.append(row)
                if cfg.get("discovery") == "marketplace":
                    provider_offers.append({
                        "model_family": family,
                        "wire_model_id": mid,
                        "seller_endpoint": "bedrock:" + region,
                        "price_in_usd_per_mtok": price["input"],
                        "price_out_usd_per_mtok": price["output"],
                        "capabilities": self._capabilities_for(family, model),
                        "quality_hint": (self._models.get(family) or {}).get("static_quality_hint"),
                    })
            offers_by_provider[pid] = provider_offers
        self._offers_by_provider = offers_by_provider
        return rows

    def offers_sync(self, provider_id: str) -> list[dict]:
        return list(self._offers_by_provider.get(provider_id) or [])

    def market_book(self) -> dict:
        rows = []
        families: dict[str, dict] = {}
        for provider, offers in self._offers_by_provider.items():
            for o in offers:
                fam = o["model_family"]
                rows.append({
                    "model_family": fam,
                    "source": "bedrock",
                    "seller": provider,
                    "wire_model_id": o.get("wire_model_id"),
                    "price_in": o.get("price_in_usd_per_mtok"),
                    "price_out": o.get("price_out_usd_per_mtok"),
                    "context": (o.get("capabilities") or {}).get("context"),
                    "tradable": True,
                    "via": provider,
                })
                families[fam] = {"sellers_total": families.get(fam, {}).get("sellers_total", 0) + 1}
        return {"rows": rows, "families": families, "fetched_at": int(time.time())}

    async def balances(self) -> dict[str, Balance]:
        return {}
