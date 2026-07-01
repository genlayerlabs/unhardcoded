"""The modular provider registry: every provider composes its aspects (source /
adapter / knobs / enabled) in ONE place, and build_source_registry, the api_kind
dispatcher handlers, and settings.SCHEMA all derive from it."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import providers  # noqa: E402
import settings  # noqa: E402


def test_every_provider_declares_at_least_one_aspect():
    # composition, not inheritance: a provider supplies only the aspects it has,
    # but it must contribute *something* (a source, a wire adapter, or knobs).
    for p in providers.PROVIDERS:
        assert p.source or p.adapter or p.knobs or p.special, p.id


def test_knob_schema_is_namespaced_and_grouped():
    sch = providers.provider_knob_schema()
    for key, spec in sch.items():
        pid, dot, knob = key.partition(".")
        assert dot and knob, key
        assert spec["provider"] == pid          # grouped by the owning provider
    # a sample of the providers that own knobs
    assert "antseed.reputation_min" in sch
    assert "codex.imputed_price_in" in sch
    assert "openrouter.runway_credits_low_usd" in sch


def test_settings_schema_merges_provider_knobs_plus_compaction():
    # the host-level compaction knob stays; provider knobs are merged in, so the
    # flat <provider>.<knob> schema settings.get()/the Config tab read is whole.
    assert "compaction.at_tokens" in settings.SCHEMA
    for key in providers.provider_knob_schema():
        assert key in settings.SCHEMA


def test_priced_providers_get_an_effective_multiplier_knob():
    sch = providers.provider_knob_schema()
    defaults = {"bedrock": 0.8, "openrouter": 1.05}
    for p in providers.PROVIDERS:
        key = f"{p.id}.price_multiplier"
        if p.source is not None:                 # only providers that push a price
            assert key in sch and sch[key]["default"] == defaults.get(p.id, 1.0) \
                and sch[key]["type"] == "float"
        else:
            assert key not in sch                # no dead knob where it can't apply


def test_enabled_predicates_gate_on_the_catalog():
    antseed = next(p for p in providers.PROVIDERS if p.id == "antseed")
    assert antseed.enabled(
        {"providers": {"antseed": {"discovery": "marketplace", "discovery_id": "antseed"}}})
    assert not antseed.enabled({"providers": {"openrouter": {"discovery": "static"}}})
    bedrock = next(p for p in providers.PROVIDERS if p.id == "bedrock")
    assert bedrock.enabled({"providers": {"bedrock_market": {"source": "bedrock"}}})


def test_native_api_kinds_declared_and_codex_is_the_one_exception():
    native = {p.api_kind for p in providers.PROVIDERS if p.api_kind and not p.special}
    assert {"anthropic", "bedrock", "google"} <= native
    codex = next(p for p in providers.PROVIDERS if p.id == "codex")
    # codex is special only in its WIRE: its adapter + the observe/bind coupling
    # live in serve.py, so it declares no api_kind/adapter here and never lands in
    # the native handler table. Its SOURCE, though, is built generically.
    assert codex.special and codex.api_kind is None and codex.adapter is None
    cat = {"providers": {"oai": {"api_kind": "openai_codex"}}, "models": {}}
    assert [s.name for s in providers.build_source_registry(cat)] == ["codex"]
