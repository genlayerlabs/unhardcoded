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
        assert p.source or p.adapter or p.stream_adapter or p.knobs or p.special, p.id


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
    for p in providers.PROVIDERS:
        key = f"{p.id}.price_multiplier"
        if p.source is not None:                 # only providers that push a price
            # default is a neutral 1.0 (no nudge); a routing preference is set
            # from the UI and persisted, never hardcoded here.
            assert key in sch and sch[key]["default"] == 1.0 \
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
    streaming = providers.native_streaming_adapter_handlers(timeout_s=1)
    assert {"anthropic", "bedrock", "google"} <= set(streaming)
    codex = next(p for p in providers.PROVIDERS if p.id == "codex")
    # codex is special only in its WIRE: its adapter + the observe/bind coupling
    # live in serve.py, so it declares no api_kind/adapter here and never lands in
    # the native handler table. Its SOURCE, though, is built generically.
    assert codex.special and codex.api_kind is None and codex.adapter is None
    cat = {"providers": {"oai": {"api_kind": "openai_codex"}}, "models": {}}
    assert [s.name for s in providers.build_source_registry(cat)] == ["codex"]


def test_native_streaming_handlers_bind_the_real_backends():
    # serve.py wires the streaming dispatcher from these handlers: a native
    # api_kind must reach its REAL stream backend, not the old
    # stream_unsupported_api_kind fallback (#63).
    from provider_adapters import anthropic, bedrock, google
    handlers = providers.native_streaming_adapter_handlers(timeout_s=1)
    assert handlers["anthropic"].func is anthropic.stream_anthropic
    assert handlers["google"].func is google.stream_google
    assert handlers["bedrock"].func is bedrock.stream_bedrock


def test_streaming_dispatcher_routes_by_api_kind():
    # the dispatcher selects the per-api_kind streaming backend and falls back to
    # the default for openai_compatible — the wiring serve.py relies on.
    import asyncio
    from streaming import make_streaming_dispatcher

    seen = {}

    async def _default(req, emit):
        seen["h"] = "default"
        return {"ok": True}

    async def _anthropic(req, emit):
        seen["h"] = "anthropic"
        return {"ok": True}

    async def _emit(_d):
        return None

    dispatch = make_streaming_dispatcher(default=_default,
                                         handlers={"anthropic": _anthropic})
    asyncio.run(dispatch({"api_kind": "anthropic"}, _emit))
    assert seen["h"] == "anthropic"
    asyncio.run(dispatch({"api_kind": "openai_compatible"}, _emit))
    assert seen["h"] == "default"
