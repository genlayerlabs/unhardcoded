"""Versioned product policies. Emit ordinary Σ_pol; never execute a second router.

The catalog describes workloads, not providers. Published bundles freeze terms
and descriptions; live candidates may change while their constraints stay fixed.
"""
from __future__ import annotations

from copy import deepcopy
import math

CATALOG_VERSION = "auto/v1"

# id, description, example, quality weight, minimum context, required capability
_SPECS = (
    ("general", "General assistance or an ambiguous task; the conservative fallback.", "Help me plan my day", .35, 0, None),
    ("conversation", "Short conversational replies and routine customer support.", "Write a friendly reply to this customer", .15, 0, None),
    ("classification", "Assign labels, detect sentiment or categorize text.", "Classify these tickets by department", .1, 0, None),
    ("extraction", "Extract fields into a JSON object or a structured schema.", "Extract the invoice date and total as JSON", .25, 0, "supports_json_mode"),
    ("translation", "Translate or localize supplied text faithfully.", "Translate this paragraph into Spanish", .3, 0, None),
    ("summarization", "Condense supplied material without researching new facts.", "Summarize this meeting transcript", .3, 0, None),
    ("long-document", "Compare or synthesize extensive documents with long context.", "Compare these three lengthy reports", .55, 128000, None),
    ("coding", "Write, explain or review code without running tools.", "Explain why this SQL query is slow", .65, 0, None),
    ("coding-agent", "Repository work, debugging or code changes using tools.", "Inspect the repository, fix the bug and run tests", .75, 32000, "supports_tools"),
    ("reasoning", "Difficult multi-step analysis, mathematical proofs or tradeoffs.", "Prove this claim and examine counterexamples", .85, 0, None),
    ("creative-writing", "Original narrative, persuasive copy or creative ideation.", "Write the opening scene of a mystery", .5, 0, None),
    ("vision", "Interpret an image supplied with the request.", "Describe the chart in this image", .5, 0, "supports_vision"),
)


def catalog():
    return [{"id": row[0], "version": CATALOG_VERSION, "description": row[1],
             "examples": [row[2]]} for row in _SPECS]


def validate_constraints(value):
    if not isinstance(value, dict):
        raise ValueError("Routing limits must be an object")
    defaults = {"max_price_in": 5.0, "max_price_out": 25.0,
                "reliability_floor": .8, "timeout_seconds": 8,
                "max_output_tokens": 2048, "priority": "cost",
                "required_capabilities": [], "allowed_models": []}
    if set(value) - set(defaults):
        raise ValueError("Unknown routing limit")
    out = {**defaults, **deepcopy(value)}
    for name, low, high in (("max_price_in", 0, 1000), ("max_price_out", 0, 1000),
                            ("reliability_floor", 0, 1)):
        n = out[name]
        if type(n) not in (int, float) or not math.isfinite(n) or not low <= n <= high:
            raise ValueError(f"Invalid {name}")
    for name, high in (("timeout_seconds", 20), ("max_output_tokens", 32768)):
        if type(out[name]) is not int or not 1 <= out[name] <= high:
            raise ValueError(f"Invalid {name}")
    if out["priority"] not in ("cost", "speed", "quality"):
        raise ValueError("Unknown priority")
    caps = out["required_capabilities"]
    if not isinstance(caps, list) or len(caps) > 3 or any(c not in (
            "supports_tools", "supports_json_mode", "supports_vision") for c in caps):
        raise ValueError("Unsupported capability")
    models = out["allowed_models"]
    if (not isinstance(models, list) or len(models) > 100 or
            any(not isinstance(m, str) or len(m) > 240 or m.count("|") != 1
                or not all(m.split("|")) for m in models)):
        raise ValueError("Invalid model restrictions")
    return out


def compile_policy(policy_id, constraints=None):
    spec = next((s for s in _SPECS if s[0] == policy_id), None)
    if spec is None:
        raise ValueError("Unknown automatic policy")
    limits = validate_constraints(constraints or {})
    _, _, _, quality, context, capability = spec
    gates = ["and", ["meets_req"], ["not", ["is", "disabled"]],
             ["cmp", "success_rate", "ge", limits["reliability_floor"]],
             ["cmp", "price_in", "le", limits["max_price_in"]],
             ["cmp", "price_out", "le", limits["max_price_out"]]]
    if context:
        gates.append(["cmp", "context", "ge", context])
    for cap in sorted(set(limits["required_capabilities"] + ([capability] if capability else []))):
        gates.append(["has_cap", cap])
    if limits["allowed_models"]:
        gates.append(["or", *[["and", ["provider_eq", m.split("|")[0]],
                               ["family_eq", m.split("|")[1]]] for m in limits["allowed_models"]]])
    cost = ["neg", ["normalize", ["add", ["scale", .8, ["field", "price_in"]],
                                 ["scale", .2, ["field", "price_out"]]]]]
    if limits["priority"] == "quality":
        quality = max(quality, .75)
    score = ["add", ["scale", quality, ["normalize", ["field", "bench_intelligence"]]],
             ["scale", 1 - quality, cost]]
    if limits["priority"] == "speed":
        score = ["add", ["scale", .5, score],
                 ["scale", .5, ["neg", ["normalize", ["field", "latency_ms"]]]]]
    failure = ["always", {"action": "abort"}]
    for reason in ("timeout", "server_error", "network_error", "rate_limit", "auth_error",
                   "payment_required", "model_unavailable", "context_overflow"):
        failure = ["override", failure, reason, {"action": "next_candidate"}]
    return ["policy", gates, score, ["prefer", ["not", ["is", "breaker_open"]], ["argmax"]],
            ["set_param", "timeout_ms", limits["timeout_seconds"] * 1000], failure]


def build_bundle(host, policy_ids, constraints=None):
    if (not isinstance(policy_ids, list) or not policy_ids or len(policy_ids) > len(_SPECS)
            or any(not isinstance(p, str) for p in policy_ids) or len(set(policy_ids)) != len(policy_ids)
            or "general" not in policy_ids):
        raise ValueError("Choose distinct policies including the general fallback")
    descriptions = {p["id"]: p for p in catalog()}
    bundle = {}
    for policy_id in policy_ids:
        term = host.normalize_policy(compile_policy(policy_id, constraints), admit=True)
        bundle[policy_id] = {**descriptions[policy_id], "policy_ir": term["policy_ir"],
                             "policy_id": term["policy_id"]}
    return bundle
