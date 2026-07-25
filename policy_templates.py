"""Intent-level policy templates compiled to existing sigma-pol/v2 terms.

The policy algebra is intentionally low-level.  These helpers keep common
product intents out of callers' hands: callers choose a goal and a few bounded
parameters; this module emits the raw IR that the existing admission pipeline
normalizes, validates, previews, and executes.

Strict provider precedence uses the engine's ``prefer(pred, inner)`` Selector:
each preference is a stable partition over the inner order.  Nesting it yields
lexicographic priority without changing scores, while ``argmax`` still ranks
routes by the template's scorer inside each provider group.

Circuit-breaker-open candidates are placed after every healthy candidate but
retained as a final fallback.  Missing providers are naturally skipped by the
live candidate set.
"""
from __future__ import annotations

import math
from typing import Any


class PolicyTemplateError(ValueError):
    """The requested intent cannot be compiled safely."""


DEFAULT_PROVIDER_ORDER: tuple[tuple[str, ...], ...] = (
    ("openai_codex",),
    ("antseed",),
    ("bedrock", "bedrock_market"),
    ("openrouter", "openrouter_market"),
)
DEFAULT_EXPECTED_INPUT_SHARE = 0.8
DEFAULT_RELIABILITY_FLOOR = 0.8
# Unknown prices observe as +inf in sigma-pol.  Product templates deliberately
# reject them instead of letting an unpriced route collapse normalization or
# create unbounded spend.  These are USD per million-token rails.
DEFAULT_MAX_PRICE_IN = 5.0
DEFAULT_MAX_PRICE_OUT = 25.0
DEFAULT_QUALITY_TOP_N = 5
DEFAULT_COST_WEIGHT = 0.75
DEFAULT_INTELLIGENCE_WEIGHT = 0.25

_BALANCED_FAILURE_ACTIONS: dict[str, dict[str, Any]] = {
    "unknown": {"action": "next_candidate"},
    "rate_limit": {
        "action": "next_candidate",
        "open_breaker_ms": 30_000,
    },
    "timeout": {"action": "next_candidate"},
    "server_error": {
        "action": "retry_same",
        "attempts": 1,
        "backoff_ms": 500,
        "then_action": "next_candidate",
    },
    "auth_error": {"action": "disable_provider"},
    "bad_request": {"action": "next_candidate"},
    "content_filter": {"action": "next_candidate"},
    "bad_response": {"action": "next_candidate"},
    "model_unavailable": {
        "action": "next_provider_same_model",
        "mark_unavailable_ms": 300_000,
    },
    "network_error": {
        "action": "retry_same",
        "attempts": 2,
        "backoff_ms": [200, 600],
        "then_action": "next_candidate",
    },
    "context_overflow": {"action": "next_candidate"},
    "stream_interrupted": {"action": "abort"},
    "payment_required": {
        "action": "next_candidate",
        "open_breaker_ms": 300_000,
    },
}

_TEMPLATE_IDS = ("cheapest-family", "smart-value", "default")
_COMMON_OPTIONS = {
    "expected_input_share",
    "max_price_in",
    "max_price_out",
    "reliability_floor",
}
_ALLOWED_OPTIONS = {
    "cheapest-family": _COMMON_OPTIONS | {
        "family",
        "provider_order",
        "provider_strategy",
    },
    "smart-value": _COMMON_OPTIONS | {"top_n"},
    "default": set(),
}


def template_catalog() -> list[dict[str, Any]]:
    """Return stable, UI-friendly descriptions of the blessed templates."""
    return [
        {
            "id": "cheapest-family",
            "description": (
                "Cheapest reliable route inside one exact model family. "
                "Optionally enforce subscription-first provider precedence."
            ),
            "required": ["family"],
            "defaults": {
                "expected_input_share": DEFAULT_EXPECTED_INPUT_SHARE,
                "reliability_floor": DEFAULT_RELIABILITY_FLOOR,
                "max_price_in": DEFAULT_MAX_PRICE_IN,
                "max_price_out": DEFAULT_MAX_PRICE_OUT,
                "provider_strategy": "cost",
                "provider_order": [list(group) for group in DEFAULT_PROVIDER_ORDER],
            },
        },
        {
            "id": "smart-value",
            "description": (
                "Cheapest reliable candidate among the current top-N models "
                "by measured intelligence."
            ),
            "required": [],
            "defaults": {
                "top_n": DEFAULT_QUALITY_TOP_N,
                "expected_input_share": DEFAULT_EXPECTED_INPUT_SHARE,
                "reliability_floor": DEFAULT_RELIABILITY_FLOOR,
                "max_price_in": DEFAULT_MAX_PRICE_IN,
                "max_price_out": DEFAULT_MAX_PRICE_OUT,
            },
        },
        {
            "id": "default",
            "description": (
                "Safe policy used by OpenAI-compatible callers that send no "
                "policy: Codex, AntSeed, Bedrock, then OpenRouter; prefer a "
                "top-five intelligence model and use a cost-dominant value "
                "score inside each provider."
            ),
            "required": [],
            "defaults": {
                "top_n": DEFAULT_QUALITY_TOP_N,
                "expected_input_share": DEFAULT_EXPECTED_INPUT_SHARE,
                "reliability_floor": DEFAULT_RELIABILITY_FLOOR,
                "max_price_in": DEFAULT_MAX_PRICE_IN,
                "max_price_out": DEFAULT_MAX_PRICE_OUT,
                "cost_weight": DEFAULT_COST_WEIGHT,
                "intelligence_weight": DEFAULT_INTELLIGENCE_WEIGHT,
                "provider_order": [list(group) for group in DEFAULT_PROVIDER_ORDER],
            },
        },
    ]


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PolicyTemplateError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PolicyTemplateError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise PolicyTemplateError(f"{name} must be a finite number")
    return number


def _bounded_number(value: Any, name: str, *, low: float, high: float) -> float:
    number = _finite_number(value, name)
    if number < low or number > high:
        raise PolicyTemplateError(f"{name} must be between {low:g} and {high:g}")
    return number


def _positive_price(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0:
        raise PolicyTemplateError(f"{name} must be greater than or equal to 0")
    return number


def _and(parts: list[list]) -> list:
    if not parts:
        return ["top"]
    if len(parts) == 1:
        return parts[0]
    return ["and", *parts]


def _or(parts: list[list]) -> list:
    if not parts:
        return ["bot"]
    if len(parts) == 1:
        return parts[0]
    return ["or", *parts]


def _cost_score(expected_input_share: float) -> list:
    """Higher is cheaper for the declared expected input/output token mix."""
    output_share = 1.0 - expected_input_share
    expected_cost = [
        "add",
        ["scale", expected_input_share, ["field", "price_in"]],
        ["scale", output_share, ["field", "price_out"]],
    ]
    return ["neg", ["normalize", expected_cost]]


def _healthy_first(selector: list) -> list:
    """Keep open-breaker routes as last resort, below every healthy route."""
    return [
        "prefer",
        ["not", ["is", "breaker_open"]],
        selector,
    ]


def _parse_provider_order(value: Any) -> tuple[tuple[str, ...], ...]:
    if value is None:
        return DEFAULT_PROVIDER_ORDER
    if not isinstance(value, list) or not value:
        raise PolicyTemplateError("provider_order must be a non-empty array")
    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for index, raw_group in enumerate(value):
        entries = raw_group if isinstance(raw_group, list) else [raw_group]
        if not entries:
            raise PolicyTemplateError(
                f"provider_order[{index}] must contain at least one provider")
        group: list[str] = []
        for raw_provider in entries:
            provider = str(raw_provider).strip() if isinstance(raw_provider, str) else ""
            if not provider:
                raise PolicyTemplateError(
                    f"provider_order[{index}] contains an empty provider")
            if provider in seen:
                raise PolicyTemplateError(
                    f"provider_order contains duplicate provider '{provider}'")
            seen.add(provider)
            group.append(provider)
        groups.append(tuple(group))
    return tuple(groups)


def _ordered_provider_selector(
    provider_order: tuple[tuple[str, ...], ...],
    inner: list | None = None,
) -> tuple[list, list]:
    """Return (allowed-provider predicate, strict provider/cost selector)."""
    provider_preds: list[list] = []
    for group in provider_order:
        group_pred = _or([["provider_eq", provider] for provider in group])
        provider_preds.append(group_pred)
    selector: list = inner or ["argmax"]
    for pred in reversed(provider_preds):
        selector = ["prefer", pred, selector]
    return _or(provider_preds), selector


def _base_filter(
    reliability: float,
    max_price_in: float,
    max_price_out: float,
) -> list[list]:
    parts: list[list] = [
        ["meets_req"],
        ["not", ["is", "disabled"]],
        ["cmp", "success_rate", "ge", reliability],
        ["cmp", "price_in", "le", max_price_in],
        ["cmp", "price_out", "le", max_price_out],
    ]
    return parts


def _common_values(options: dict[str, Any]) -> tuple[list[list], list, dict[str, Any]]:
    input_share = _bounded_number(
        options.get("expected_input_share", DEFAULT_EXPECTED_INPUT_SHARE),
        "expected_input_share",
        low=0,
        high=1,
    )
    reliability = _bounded_number(
        options.get("reliability_floor", DEFAULT_RELIABILITY_FLOOR),
        "reliability_floor",
        low=0,
        high=1,
    )
    max_price_in = _positive_price(
        options.get("max_price_in", DEFAULT_MAX_PRICE_IN),
        "max_price_in",
    )
    max_price_out = _positive_price(
        options.get("max_price_out", DEFAULT_MAX_PRICE_OUT),
        "max_price_out",
    )
    parts = _base_filter(reliability, max_price_in, max_price_out)
    intent = {
        "expected_input_share": input_share,
        "reliability_floor": reliability,
        "max_price_in": max_price_in,
        "max_price_out": max_price_out,
    }
    return parts, _cost_score(input_share), intent


def _balanced_fail_plan() -> list:
    plan: list = ["always", dict(_BALANCED_FAILURE_ACTIONS["unknown"])]
    for reason in sorted(_BALANCED_FAILURE_ACTIONS):
        if reason != "unknown":
            plan = [
                "override",
                plan,
                reason,
                dict(_BALANCED_FAILURE_ACTIONS[reason]),
            ]
    return plan


def _policy(pred: list, scorer: list, selector: list | None = None) -> list:
    return [
        "policy",
        pred,
        scorer,
        selector or ["argmax"],
        ["id"],
        _balanced_fail_plan(),
    ]


def _cheapest_family(options: dict[str, Any]) -> tuple[list, dict[str, Any]]:
    family = str(options.get("family") or "").strip()
    if not family:
        raise PolicyTemplateError("cheapest-family requires a non-empty family")
    parts, cost_score, intent = _common_values(options)
    parts.append(["family_eq", family])

    strategy = str(options.get("provider_strategy") or "cost").strip()
    if strategy == "cost":
        selector = _healthy_first(["argmax"])
    elif strategy == "ordered":
        provider_order = _parse_provider_order(options.get("provider_order"))
        allowed, selector = _ordered_provider_selector(provider_order)
        selector = _healthy_first(selector)
        parts.append(allowed)
        intent["provider_order"] = [list(group) for group in provider_order]
    else:
        raise PolicyTemplateError(
            "provider_strategy must be 'cost' or 'ordered'")

    intent.update({"family": family, "provider_strategy": strategy})
    return _policy(_and(parts), cost_score, selector), intent


def _smart_value(options: dict[str, Any]) -> tuple[list, dict[str, Any]]:
    parts, cost_score, intent = _common_values(options)
    top_n_raw = options.get("top_n", DEFAULT_QUALITY_TOP_N)
    if isinstance(top_n_raw, bool):
        raise PolicyTemplateError("top_n must be a positive integer")
    try:
        top_n = int(top_n_raw)
    except (TypeError, ValueError) as exc:
        raise PolicyTemplateError("top_n must be a positive integer") from exc
    if top_n < 1 or top_n != top_n_raw:
        raise PolicyTemplateError("top_n must be a positive integer")
    parts.append(["cmp", "bench_intelligence_rank", "le", top_n])
    intent["top_n"] = top_n
    return _policy(
        _and(parts),
        cost_score,
        _healthy_first(["argmax"]),
    ), intent


def _default() -> tuple[list, dict[str, Any]]:
    parts, cost_score, intent = _common_values({})
    provider_order = DEFAULT_PROVIDER_ORDER
    allowed, selector = _ordered_provider_selector(provider_order)
    selector = [
        "prefer",
        ["cmp", "bench_intelligence_rank", "le", DEFAULT_QUALITY_TOP_N],
        selector,
    ]
    selector = _healthy_first(selector)
    parts.append(allowed)
    intent.update({
        "top_n": DEFAULT_QUALITY_TOP_N,
        "cost_weight": DEFAULT_COST_WEIGHT,
        "intelligence_weight": DEFAULT_INTELLIGENCE_WEIGHT,
        "provider_order": [list(group) for group in provider_order],
    })
    value_score = [
        "add",
        ["scale", DEFAULT_COST_WEIGHT, cost_score],
        [
            "scale",
            DEFAULT_INTELLIGENCE_WEIGHT,
            ["normalize", ["field", "bench_intelligence"]],
        ],
    ]
    return _policy(_and(parts), value_score, selector), intent


def build_policy_template(template_id: str,
                          options: dict[str, Any] | None = None) -> tuple[list, dict[str, Any]]:
    """Compile a blessed template and return ``(policy_ir, normalized_intent)``.

    The returned term is deliberately not normalized or admitted here.  The
    host's existing ``/x/policy/normalize`` and ``/x/rank`` paths remain the
    single authority for sigma-pol identity and live-schema admission.
    """
    template = str(template_id or "").strip()
    if template not in _TEMPLATE_IDS:
        raise PolicyTemplateError(
            f"unknown policy template '{template}'; expected one of "
            + ", ".join(_TEMPLATE_IDS))
    opts = dict(options or {})
    unknown = set(opts) - _ALLOWED_OPTIONS[template]
    if unknown:
        raise PolicyTemplateError(
            f"{template}: unknown option(s): {', '.join(sorted(unknown))}")

    if template == "cheapest-family":
        term, intent = _cheapest_family(opts)
    elif template == "smart-value":
        term, intent = _smart_value(opts)
    else:
        # A pinned, versioned product default.  Keep this in parity with the
        # default profile in config.live.lua (covered by a live-config test).
        term, intent = _default()
    return term, {"template": template, **intent}
