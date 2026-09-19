"""Real-engine checks of product routing, not snapshots of the compiler."""
import pytest

from auto_policies import build_bundle, catalog, compile_policy
from test_policy_templates import template_host


def test_catalog_compiles_and_identity_is_stable(template_host):
    ids = [p["id"] for p in catalog()]
    assert len(ids) == len(set(ids)) == 12
    a = build_bundle(template_host, ids)
    assert a == build_bundle(template_host, ids)
    assert all(len(p["policy_id"]) == 64 for p in a.values())


def test_all_policies_preserve_price_and_provider_restrictions(template_host):
    for policy in catalog():
        term = compile_policy(policy["id"], {"max_price_in": 2, "max_price_out": 2,
                                          "allowed_models": ["antseed|family-a"]})
        ranked, _ = template_host.rank({"policy_ir": term})
        assert all(r["candidate"]["provider_id"] == "antseed" and
                   r["candidate"]["model_family"] == "family-a" for r in ranked)
    ranked, _ = template_host.rank({"policy_ir": compile_policy("general", {
        "max_price_in": 0, "max_price_out": 0, "allowed_models": ["antseed|family-a"]})})
    assert ranked == []


def test_unknown_prices_and_missing_capabilities_do_not_qualify(template_host):
    for name in ("general", "extraction", "vision"):
        ranked, _ = template_host.rank({"policy_ir": compile_policy(name)})
        assert all(r["candidate"]["provider_id"] != "unpriced" for r in ranked)
        if name != "general":
            assert ranked == []


def test_new_candidates_can_qualify_without_changing_policy(template_host):
    term = compile_policy("general")
    candidate = {"provider_id": "new-provider", "model_family": "new-model", "price_in": .01,
                 "price_out": .01, "capabilities": {"context": 128000}}
    before = template_host.normalize_policy(term, admit=True)
    ranked, _ = template_host.rank({"policy_ir": term, "extra_candidates": [candidate]})
    assert any(r["candidate"]["provider_id"] == "new-provider" for r in ranked)
    assert template_host.normalize_policy(term, admit=True) == before


@pytest.mark.parametrize("limits", [{"max_price_in": float("nan")}, {"max_price_out": -1},
    {"timeout_seconds": True}, {"priority": "ignore"}, {"required_capabilities": ["shell"]},
    {"allowed_models": ["invalid"]}, {"unknown": 1}])
def test_invalid_limits_rejected(limits):
    with pytest.raises(ValueError):
        compile_policy("general", limits)
