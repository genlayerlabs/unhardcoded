"""Amazon Bedrock source: dynamic OpenAI-compatible model discovery plus
public AWS price-list ingestion.

The Bedrock OpenAI-compatible data plane is already just another
openai_compatible provider to the router. This source supplies the missing
catalog facts: which Bedrock models the configured account/region exposes and
what their on-demand token prices are. Pricing comes from AWS's public offer
files, while model availability is account/region scoped and therefore uses the
same Bedrock bearer token as inference.
"""
from __future__ import annotations

import os
import time
from typing import Any

from sources import Balance, Price

AWS_PRICE_BASE = "https://pricing.us-east-1.amazonaws.com"
PRICE_OFFER = "AmazonBedrockFoundationModels"

_SECRET_PLACEHOLDERS = {"", "CHANGE_ME", "TODO", "TODO_CHANGE_ME", "PLACEHOLDER"}


def _configured(value: str | None) -> bool:
    return value is not None and value.strip().upper() not in _SECRET_PLACEHOLDERS


_FAMILY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("claude-opus-4-8", ("claude-opus-4-8", "claude-4-8-opus", "claude-opus-4.8")),
    ("claude-sonnet-4-6", ("claude-sonnet-4-6", "claude-4-6-sonnet", "claude-sonnet-4.6")),
    ("qwen3-235b-a22b", ("qwen3-235b-a22b", "qwen-3-235b", "qwen3-235b")),
    ("llama-4-maverick", ("llama-4-maverick", "llama4-maverick")),
    ("deepseek-v4-pro", ("deepseek-v4-pro", "deepseek-v4")),
    ("deepseek-v4-flash", ("deepseek-v4-flash",)),
    ("gemma-3-27b", ("gemma-3-27b", "gemma3-27b")),
]

_PRICE_SERVICE_PATTERNS: dict[str, tuple[str, ...]] = {
    "claude-opus-4-8": ("claude opus 4.8",),
    "claude-sonnet-4-6": ("claude sonnet 4.6",),
    "qwen3-235b-a22b": ("qwen3 235b", "qwen 3 235b", "qwen3-235b"),
    "llama-4-maverick": ("llama 4 maverick", "llama-4-maverick"),
    "deepseek-v4-pro": ("deepseek v4 pro", "deepseek-v4-pro"),
    "deepseek-v4-flash": ("deepseek v4 flash", "deepseek-v4-flash"),
    "gemma-3-27b": ("gemma 3 27b", "gemma-3-27b"),
}

_PRICE_EXCLUDE = (
    "reserved", "batch", "cache", "cached", "embedding", "image", "video",
    "fine-tun", "custom", "provisioned", "training",
)


class BedrockSource:
    name = "bedrock"
    poll_interval_s = 3600

    def __init__(self, catalog: dict, env_get=os.environ.get,
                 client: Any = None, pricing_base: str = AWS_PRICE_BASE):
        self._catalog = catalog
        self._env_get = env_get
        self._client = client
        self._pricing_base = pricing_base.rstrip("/")
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

    def _provider_token(self, cfg: dict) -> str | None:
        auth = cfg.get("auth") if isinstance(cfg.get("auth"), dict) else None
        env = cfg.get("auth_env") or (auth.get("env") if auth else None)
        token = self._env_get(str(env)) if env else None
        return token if _configured(token) else None

    @staticmethod
    def _normalize(s: str) -> str:
        return str(s or "").lower().replace(".", "-").replace("_", "-").replace(":", "-")

    def _family_for_model(self, provider_id: str, model_id: str, model: dict | None = None) -> str | None:
        model_id = str(model_id)
        if (provider_id, model_id) in self._static_family_by_id:
            return self._static_family_by_id[(provider_id, model_id)]
        low = model_id.lower()
        if low in self._aliases:
            return self._aliases[low]
        haystack = " ".join(str(x or "") for x in (
            model_id,
            (model or {}).get("id"),
            (model or {}).get("modelId"),
            (model or {}).get("model_name"),
            (model or {}).get("modelName"),
        ))
        norm = self._normalize(haystack)
        for family, patterns in _FAMILY_PATTERNS:
            if any(self._normalize(pat) in norm for pat in patterns):
                return family
        return None

    async def _provider_models(self, cfg: dict) -> list[dict]:
        token = self._provider_token(cfg)
        if not token:
            return []
        base = str(cfg.get("base_url") or "").rstrip("/")
        if not base:
            return []
        body = await self._get_json(base + "/models", headers={"Authorization": f"Bearer {token}"})
        if isinstance(body.get("data"), list):
            return [m for m in body["data"] if isinstance(m, dict) and m.get("id")]
        if isinstance(body.get("modelSummaries"), list):
            out = []
            for m in body["modelSummaries"]:
                if not isinstance(m, dict) or not m.get("modelId"):
                    continue
                out.append({**m, "id": m.get("modelId")})
            return out
        return []

    async def _price_region_url(self, region: str) -> str:
        idx = await self._get_json(
            f"{self._pricing_base}/offers/v1.0/aws/{PRICE_OFFER}/current/region_index.json")
        region_row = (idx.get("regions") or {}).get(region)
        if not region_row or not region_row.get("currentVersionUrl"):
            raise RuntimeError(f"Bedrock price region {region!r} not found")
        return self._pricing_base + str(region_row["currentVersionUrl"])

    @staticmethod
    def _dimension_kind(attrs: dict, description: str) -> str | None:
        text = f"{attrs.get('usagetype', '')} {description}".lower()
        if any(x in text for x in _PRICE_EXCLUDE):
            return None
        if "inputtokencount" in text or "input tokens standard" in text \
                or "price per 1 million input tokens" in text:
            return "input"
        if "outputtokencount" in text or "output tokens standard" in text \
                or "response tokens" in text \
                or "price per 1 million output tokens" in text:
            return "output"
        return None

    @staticmethod
    def _family_for_service_name(name: str) -> str | None:
        low = name.lower()
        for family, patterns in _PRICE_SERVICE_PATTERNS.items():
            if any(pat in low for pat in patterns):
                return family
        return None

    async def _prices_by_family(self, region: str) -> dict[str, dict[str, float]]:
        data = await self._get_json(await self._price_region_url(region))
        terms = data.get("terms", {}).get("OnDemand", {})
        out: dict[str, dict[str, float]] = {}
        for sku, product in (data.get("products") or {}).items():
            attrs = product.get("attributes") or {}
            family = self._family_for_service_name(str(attrs.get("servicename") or ""))
            if not family:
                continue
            for term in (terms.get(sku) or {}).values():
                for pd in (term.get("priceDimensions") or {}).values():
                    kind = self._dimension_kind(attrs, str(pd.get("description") or ""))
                    if not kind:
                        continue
                    try:
                        price = float((pd.get("pricePerUnit") or {}).get("USD"))
                    except (TypeError, ValueError):
                        continue
                    slot = out.setdefault(family, {})
                    slot[kind] = min(slot.get(kind, price), price)
        return {
            fam: {"input": p["input"], "output": p["output"]}
            for fam, p in out.items()
            if p.get("input") is not None and p.get("output") is not None
        }

    def _capabilities_for(self, family: str, model: dict) -> dict:
        caps = dict((self._models.get(family) or {}).get("capabilities") or {})
        ctx = model.get("context_length") or model.get("contextLength")
        if isinstance(ctx, (int, float)) and ctx:
            caps["context"] = int(ctx)
        # Bedrock's OpenAI-compatible /models response may expose the same field
        # as OpenRouter. When absent, curated catalog capabilities are the source
        # of truth for the production families.
        params = set(model.get("supported_parameters") or [])
        if "tools" in params:
            caps["supports_tools"] = True
        if params & {"response_format", "structured_outputs"}:
            caps["supports_json_mode"] = True
        return caps

    async def pricing(self) -> list[Price]:
        region = self._env_get("BEDROCK_REGION") or "us-east-1"
        try:
            prices = await self._prices_by_family(region)
        except Exception:
            prices = {}

        rows: list[Price] = []
        offers_by_provider: dict[str, list[dict]] = {}
        for pid, cfg in self._providers.items():
            models = await self._provider_models(cfg)
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
                        "seller_endpoint": cfg.get("base_url"),
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
