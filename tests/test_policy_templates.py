"""Blessed intent templates compile to real policies with enforceable behavior."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_router_host import LLMRouterHost
from policy_templates import (
    PolicyTemplateError,
    build_policy_template,
    template_catalog,
)
from shim import create_app


ROOT = Path(__file__).resolve().parents[1]
FAMILY = "family-a"


@pytest.fixture
def template_host(tmp_path):
    config = tmp_path / "templates.lua"
    config.write_text(
        """
local intelligence_rank = {
    ["family-a"] = 20,
    ["frontier-expensive"] = 1,
    ["value"] = 2,
    ["unpriced-top"] = 3,
    ["cheap-low"] = 20,
}
local intelligence = {
    ["family-a"] = 0.1,
    ["frontier-expensive"] = 0.9,
    ["value"] = 0.8,
    ["unpriced-top"] = 0.7,
    ["cheap-low"] = 0.1,
}

return {
    providers = {
        openai_codex = { discovery = "static", base_url = "http://codex",
                         api_kind = "openai_compatible", tier = "partner" },
        antseed = { discovery = "static", base_url = "http://antseed",
                    api_kind = "openai_compatible", tier = "marketplace" },
        bedrock = { discovery = "static", base_url = "http://bedrock",
                    api_kind = "openai_compatible", tier = "partner" },
        bedrock_market = { discovery = "static", base_url = "http://bedrock-market",
                           api_kind = "openai_compatible", tier = "partner" },
        openrouter = { discovery = "static", base_url = "http://openrouter",
                       api_kind = "openai_compatible", tier = "fallback" },
        openrouter_market = { discovery = "static", base_url = "http://openrouter-market",
                              api_kind = "openai_compatible", tier = "marketplace" },
        other = { discovery = "static", base_url = "http://other",
                  api_kind = "openai_compatible", tier = "partner" },
        unpriced = { discovery = "static", base_url = "http://unpriced",
                     api_kind = "openai_compatible", tier = "partner" },
    },
    models = {
        ["family-a"] = {
            served_by = {
                { provider = "openai_codex" },
                { provider = "antseed" },
                { provider = "bedrock" },
                { provider = "bedrock_market" },
                { provider = "openrouter" },
                { provider = "openrouter_market" },
                { provider = "other" },
            },
            capabilities = { context = 128000, supports_tools = true },
        },
        ["frontier-expensive"] = {
            served_by = { { provider = "openrouter" } },
            capabilities = { context = 128000, supports_tools = true },
        },
        ["value"] = {
            served_by = { { provider = "antseed" } },
            capabilities = { context = 128000, supports_tools = true },
        },
        ["unpriced-top"] = {
            served_by = { { provider = "unpriced" } },
            capabilities = { context = 128000, supports_tools = true },
        },
        ["cheap-low"] = {
            served_by = { { provider = "openai_codex" } },
            capabilities = { context = 128000, supports_tools = true },
        },
    },
    profiles = { default = { scorer = { "zero" } } },
    fields = {
        bench_intelligence = {
            sort = "Num", default = 0, group = "model",
            get = function(c) return intelligence[c.model_family] end,
        },
        bench_intelligence_rank = {
            sort = "Num", default = 1e9, group = "model",
            get = function(c) return intelligence_rank[c.model_family] end,
        },
    },
    policy_envelope = {
        "and", { "meets_req" }, { "not", { "is", "disabled" } },
    },
}
"""
    )
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=config,
        now_ms=lambda: 1_000,
    )
    host.init()
    prices = {
        ("openai_codex", FAMILY): 4.9,
        ("antseed", FAMILY): 1,
        ("bedrock", FAMILY): 2,
        ("bedrock_market", FAMILY): 2.5,
        ("openrouter", FAMILY): 3,
        ("openrouter_market", FAMILY): 3.5,
        ("other", FAMILY): 0.1,
        ("openrouter", "frontier-expensive"): 0.5,
        ("antseed", "value"): 2,
        ("openai_codex", "cheap-low"): 0,
    }
    for (provider, family), price in prices.items():
        host.update_metrics(provider, family, {
            "price_in": price,
            "price_out": price,
        })
    return host


def _providers(host, term):
    ranked, _ = host.rank({
        "policy_ir": term,
        "requirements": {"context": 8_000},
    })
    return [row["candidate"]["provider_id"] for row in ranked]


def test_template_catalog_exposes_four_product_intents():
    assert [item["id"] for item in template_catalog()] == [
        "cheapest-family",
        "smart-value",
        "agent",
        "default",
    ]


def test_template_endpoint_returns_normalized_identified_policy(template_host):
    client = TestClient(create_app(template_host, default_profile="default"))
    response = client.post(
        "/x/policy/templates/cheapest-family",
        json={"family": FAMILY, "provider_strategy": "ordered"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["policy_ir"][0] == "policy"
    assert body["fingerprint"]
    assert body["version"] == "sigma-pol/v2"
    assert body["intent"]["provider_order"][0] == ["openai_codex"]
    # The result is admitted by the real live-schema path, not just generated.
    assert client.post(
        "/x/rank",
        json={"policy_ir": body["policy_ir"]},
    ).status_code == 200


def test_parameterless_default_template_needs_no_request_body(template_host):
    client = TestClient(create_app(template_host, default_profile="default"))
    response = client.post("/x/policy/templates/default")
    assert response.status_code == 200, response.text
    assert response.json()["intent"]["template"] == "default"


def test_parameterless_agent_template_needs_no_request_body(template_host):
    client = TestClient(create_app(template_host, default_profile="default"))
    response = client.post("/x/policy/templates/agent")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["intent"]["template"] == "agent"
    assert body["intent"]["first_token_timeout_ms"] == 10_000


def test_agent_template_owns_quality_privacy_and_liveness_rails():
    term, intent = build_policy_template("agent")
    _, pred, scorer, selector, xform, fail_plan = term
    predicates = pred[1:]

    assert ["is", "cap_tools"] in predicates
    assert ["cmp", "context", "ge", 128_000] in predicates
    assert ["cmp", "success_rate", "ge", 0.8] in predicates
    assert ["cmp", "bench_intelligence_rank", "le", 10] in predicates
    assert ["cmp", "price_in", "le", 15.0] in predicates
    assert ["cmp", "price_out", "le", 30.0] in predicates

    antseed_gate = next(
        item for item in predicates
        if item[0] == "or" and ["not", ["provider_eq", "antseed"]] in item
    )
    assert ["cmp", "reputation_score", "gt", 95] in antseed_gate
    assert len([item for item in antseed_gate if item[0] == "served_by_eq"]) == 5

    assert scorer[0] == "add"
    assert selector[:2] == ["top_k", 8]
    assert selector[2][0:2] == [
        "prefer",
        ["not", ["is", "breaker_open"]],
    ]
    assert xform == [
        "seq",
        ["set_param", "first_token_timeout_ms", 10_000],
        ["set_param", "timeout_ms", 22_000],
    ]
    assert intent["provider_order"][0] == ["openai_codex"]
    assert intent["provider_order"][-1] == ["antseed"]

    actions = {}
    cursor = fail_plan
    while cursor[0] == "override":
        actions[cursor[2]] = cursor[3]
        cursor = cursor[1]
    actions["unknown"] = cursor[1]
    assert actions["timeout"] == {"action": "next_candidate"}
    assert actions["server_error"] == {"action": "next_candidate"}
    assert actions["network_error"] == {"action": "next_candidate"}


def test_cheapest_family_is_family_strict_and_really_cost_first(template_host):
    term, _ = build_policy_template("cheapest-family", {"family": FAMILY})
    providers = _providers(template_host, term)
    assert providers[0] == "other", "the cheapest route wins without provider precedence"
    ranked, _ = template_host.rank({
        "policy_ir": term,
        "requirements": {"context": 8_000},
    })
    assert {row["candidate"]["model_family"] for row in ranked} == {FAMILY}


def test_ordered_family_is_strict_and_keeps_cost_order_inside_groups(template_host):
    term, _ = build_policy_template("cheapest-family", {
        "family": FAMILY,
        "provider_strategy": "ordered",
    })
    assert _providers(template_host, term) == [
        "openai_codex",
        "antseed",
        "bedrock",
        "bedrock_market",
        "openrouter",
        "openrouter_market",
    ]


def test_ordered_family_skips_open_breaker_before_provider_priority(template_host):
    state = template_host.dump_state()
    state["circuit_breakers"] = {
        "openai_codex": {
            "open": True,
            "opened_at_ms": 1_000,
            "consecutive_failures": 4,
        },
    }
    template_host.restore_state(state)
    term, _ = build_policy_template("cheapest-family", {
        "family": FAMILY,
        "provider_strategy": "ordered",
    })
    providers = _providers(template_host, term)
    assert providers[:3] == ["antseed", "bedrock", "bedrock_market"]
    assert providers[-1] == "openai_codex", \
        "an open route remains available, but only after every healthy route"


def test_ordered_family_executes_the_declared_fallback_chain(template_host):
    template_host.set_mock_response("openai_codex", FAMILY, {
        "ok": False,
        "error_kind": "server_error",
        "latency_ms": 1,
    })
    template_host.set_mock_response("antseed", FAMILY, {
        "ok": True,
        "latency_ms": 1,
        "response": {
            "text": "from-antseed",
            "finish_reason": "stop",
            "tokens_in": 1,
            "tokens_out": 1,
            "tokens_total": 2,
        },
    })
    term, _ = build_policy_template("cheapest-family", {
        "family": FAMILY,
        "provider_strategy": "ordered",
    })
    result = template_host.execute({
        "policy_ir": term,
        "requirements": {"context": 8_000},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert result["ok"]
    assert result["chosen"]["provider_id"] == "antseed"
    attempted = [
        event["provider_id"]
        for event in result["trace"]["decision_path"]
        if event["event"] == "attempted"
    ]
    assert attempted == ["openai_codex", "openai_codex", "antseed"]


def test_smart_value_chooses_cheapest_inside_quality_shortlist(template_host):
    term, _ = build_policy_template("smart-value")
    ranked, _ = template_host.rank({
        "policy_ir": term,
        "requirements": {"context": 8_000},
    })
    pairs = [
        (row["candidate"]["provider_id"], row["candidate"]["model_family"])
        for row in ranked
    ]
    assert pairs[0] == ("openrouter", "frontier-expensive")
    assert ("unpriced", "unpriced-top") not in pairs, \
        "unknown prices must fail closed instead of collapsing cost ordering"
    assert ("openai_codex", "cheap-low") not in pairs, \
        "a free but out-of-shortlist model must not silently lower quality"


def test_default_prefers_quality_then_subscription_provider_order(template_host):
    term, intent = build_policy_template("default")
    ranked, _ = template_host.rank({
        "policy_ir": term,
        "requirements": {"context": 8_000},
    })
    pairs = [
        (row["candidate"]["provider_id"], row["candidate"]["model_family"])
        for row in ranked
    ]
    assert pairs[0] == ("antseed", "value"), \
        "provider precedence wins among quality-shortlisted candidates"
    assert ("unpriced", "unpriced-top") not in pairs
    assert intent["provider_order"][0] == ["openai_codex"]


def test_default_keeps_requested_family_outside_top_five(template_host):
    term, _ = build_policy_template("default")
    ranked, _ = template_host.rank({
        "policy_ir": term,
        "requirements": {"context": 8_000, "model_family": FAMILY},
    })
    assert [row["candidate"]["provider_id"] for row in ranked] == [
        "openai_codex",
        "antseed",
        "bedrock",
        "bedrock_market",
        "openrouter",
        "openrouter_market",
    ]


def test_live_default_profile_is_exactly_the_published_default_template():
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env={
            "OPENAI_API_KEY": "test",
            "OPENROUTER_API_KEY": "test",
        },
        now_ms=lambda: 1_000,
        enforce_provider_auth=False,
    )
    host.init()
    generated, _ = build_policy_template("default")
    configured = host.catalog()["profiles"]["default"]["policy_ir"]
    assert (
        host.normalize_policy(configured)["fingerprint"]
        == host.normalize_policy(generated)["fingerprint"]
    )

    configured_rank, _ = host.rank({
        "profile": "default",
        "requirements": {"context": 8_000},
    })
    generated_rank, _ = host.rank({
        "policy_ir": generated,
        "requirements": {"context": 8_000},
    })
    configured_pairs = [
        (row["candidate"]["provider_id"], row["candidate"]["model_family"])
        for row in configured_rank
    ]
    generated_pairs = [
        (row["candidate"]["provider_id"], row["candidate"]["model_family"])
        for row in generated_rank
    ]
    assert configured_pairs == generated_pairs
    assert configured_pairs[0] == ("openai_codex", "gpt-5.5")


def test_live_agent_profile_is_exactly_the_published_agent_template():
    import asyncio

    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env={
            "OPENAI_API_KEY": "test",
            "OPENROUTER_API_KEY": "test",
        },
        now_ms=lambda: 1_000,
        enforce_provider_auth=False,
    )
    host.init()
    generated, _ = build_policy_template("agent")
    configured = host.catalog()["profiles"]["agent"]["policy_ir"]
    assert (
        host.normalize_policy(configured)["fingerprint"]
        == host.normalize_policy(generated)["fingerprint"]
    )

    configured_rank, _ = host.rank({
        "profile": "agent",
        "requirements": {"context": 128_000},
    })
    generated_rank, _ = host.rank({
        "policy_ir": generated,
        "requirements": {"context": 128_000},
    })
    configured_pairs = [
        (row["candidate"]["provider_id"], row["candidate"]["model_family"])
        for row in configured_rank
    ]
    generated_pairs = [
        (row["candidate"]["provider_id"], row["candidate"]["model_family"])
        for row in generated_rank
    ]
    assert configured_pairs == generated_pairs
    assert 1 <= len(configured_pairs) <= 8
    assert configured_pairs[0][0] == "openai_codex"

    seen = []

    async def override(request):
        seen.append(request)
        if len(seen) == 1:
            return {"ok": False, "error_kind": "timeout", "latency_ms": 1}
        return {
            "ok": True,
            "latency_ms": 1,
            "response": {"text": "fallback", "finish_reason": "stop"},
        }

    result = asyncio.run(host.execute_async({
        "prompt": "hi",
        "profile": "agent",
    }, call_override=override))
    assert result["ok"] and result["response"]["text"] == "fallback"
    assert len(seen) == 2
    assert all(req["first_token_timeout_ms"] == 10_000 for req in seen)
    assert all(req["timeout_ms"] == 22_000 for req in seen)


@pytest.mark.parametrize(
    ("template", "options", "message"),
    [
        ("cheapest-family", {}, "requires a non-empty family"),
        ("cheapest-family", {"family": FAMILY, "provider_strategy": "magic"},
         "must be 'cost' or 'ordered'"),
        ("smart-value", {"top_n": 0}, "positive integer"),
        ("smart-value", {"surprise": True}, "unknown option"),
        ("missing", {}, "unknown policy template"),
    ],
)
def test_invalid_template_intents_fail_closed(template, options, message):
    with pytest.raises(PolicyTemplateError, match=message):
        build_policy_template(template, options)
