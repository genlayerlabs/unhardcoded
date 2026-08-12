"""
Entrypoint for the unhardcoded HTTP host (OpenAI-compatible shim).

    python serve.py \
        --config config.live.lua \
        --metrics metrics.live.lua \
        --default-profile edge \
        --host 127.0.0.1 --port 8080

The unhardcoded-engine core is vendored as a git submodule under `core/`. Provider auth
lives in the process environment (the core resolves `auth_env` per provider via
`host.env`). Clients hitting the shim do NOT need provider API keys — they only
need to reach the shim's URL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"                 # the unhardcoded-engine core (git submodule)
sys.path.insert(0, str(ROOT))

from llm_router_host import LLMRouterHost  # noqa: E402
# The native api_kind adapters (anthropic/bedrock/google) are now owned by the
# modular provider registry (`providers.py`); serve only needs the dispatcher,
# the default openai_compatible backend, and the codex backend (special).
from provider_adapters.dispatcher import make_api_kind_dispatcher  # noqa: E402
from provider_adapters.openai_compatible import make_async_call_provider  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(prog="serve.py")
    p.add_argument("--router", type=Path, default=CORE / "router.lua",
                   help="path to the core entry (default: core/router.lua)")
    p.add_argument("--config", type=Path, default=ROOT / "config.live.lua",
                   help="path to the catalog (default: config.live.lua)")
    p.add_argument("--metrics", type=Path, default=None,
                   help="optional path to metrics.lua")
    p.add_argument("--default-profile", default="default",
                   help="fallback policy used when a caller sends no policy_ir "
                        "and no `profile:`/`family:` prefix")
    p.add_argument("--default-max-tokens", type=int, default=4096,
                   help="max_tokens supplied when a request omits it (some "
                        "upstreams reject requests without it). Set to 0 to "
                        "forward nothing (strict OpenAI-spec behaviour).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--timeout-s", type=float, default=30.0,
                   help="upstream provider call timeout in seconds")
    p.add_argument("--codex-auth", type=Path, default=None,
                   help="path to Codex auth.json for api_kind=openai_codex "
                        "(default: ~/.codex/auth.json). Enables the ChatGPT "
                        "subscription provider — unofficial, ToS-risky.")
    args = p.parse_args()

    # Operator settings (overrides) live in the host store (Postgres). Refresh
    # them here so this process reflects any writes made before it started — its
    # import-time load may have run before the pool was ready.
    import settings
    settings.reload()

    # Operator-managed provider keys persisted by the dashboard live on the PVC
    # (.env.secrets) and are the source of truth. Load them over the container
    # env BEFORE LLMRouterHost snapshots os.environ, so dashboard edits to
    # heurist/ionet/openrouter keys take effect and survive pod restarts.
    from env_secrets import load_env_secrets  # noqa: E402
    loaded = load_env_secrets()
    if loaded:
        print(f"env secrets loaded from PVC: {len(loaded)} keys")

    # The job: refresh the registered model_meta.lua (OpenRouter benchmarks/
    # modalities/capabilities) BEFORE the config is loaded, so config.live.lua
    # picks up fresh per-family traits at init. Best-effort: a network blip or
    # MODEL_META_REFRESH=0 just keeps the last committed file.
    import os
    if os.getenv("MODEL_META_REFRESH", "1") != "0":
        try:
            import asyncio

            from scripts.refresh_model_meta import generate
            n = asyncio.run(generate(args.config, ROOT / "model_meta.lua"))
            print(f"model_meta refreshed: {n} families")
        except Exception as exc:  # noqa: BLE001
            print(f"model_meta refresh skipped: {type(exc).__name__}: {exc}")

    host = LLMRouterHost(
        router_path=args.router,
        config_path=args.config,
        metrics_path=args.metrics,
    )
    # operator-added providers (dashboard "Add provider" flow) merge into the
    # catalog before init; their keys arrive via env (.env.secrets)
    from provider_overlay import apply_to_host, load_overlay
    overlay_applied = apply_to_host(host, load_overlay())
    host.init()
    if overlay_applied:
        print(f"provider overlay applied: {', '.join(overlay_applied)}")

    # Per-provider declarative rules (error_map) come from the loaded catalog,
    # so the dispatcher is built after the host. The source registry is also
    # built here: the codex source must observe the codex backend's traffic.
    import providers
    catalog = host.catalog()
    registry = providers.build_source_registry(catalog)
    codex_src = next((s for s in registry if s.name == "codex"), None)
    observe = None
    if codex_src is not None:
        codex_pid = codex_src.provider_ids[0]
        codex_src.bind(host, [
            family for family, model in (catalog.get("models") or {}).items()
            if any(s.get("provider") == codex_pid for s in model.get("served_by") or [])
        ])
        observe = lambda sig: codex_src.ingest(codex_pid, sig)  # noqa: E731

    provider_rules = {
        pid: {"error_map": p["error_map"]}
        for pid, p in (catalog.get("providers") or {}).items()
        if isinstance(p, dict) and p.get("error_map")
    }
    # One reusable pool per router process. Long-running agent streams otherwise
    # create a client/socket per attempt; under bursts that is exactly the kind
    # of resource churn that turns latency into intermittent 502s.
    import httpx
    provider_http = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=100,
                            max_keepalive_connections=40))

    # Codex account refresh credentials remain on one RWO-PVC owner. A stateless
    # router uses the authenticated internal broker; a local install keeps the
    # original in-process store.
    codex_broker_url = os.getenv("CODEX_BROKER_URL", "").strip()
    codex_broker_token = os.getenv("CODEX_BROKER_TOKEN", "").strip()
    codex_store = None
    remote_codex = None
    if codex_broker_url:
        from remote_codex import RemoteCodexClient
        remote_codex = RemoteCodexClient(
            codex_broker_url, codex_broker_token,
            timeout_s=max(120.0, args.timeout_s))

        async def codex_call(request):
            return await remote_codex.call(request, observe=observe)

        async def codex_stream(request, emit):
            return await remote_codex.stream(request, emit, observe=observe)
        print(f"codex broker: {codex_broker_url}")
    else:
        from codex_auth import CodexAuthStore
        from codex_backend import make_codex_async_call_provider
        from streaming import stream_codex

        codex_store = CodexAuthStore(legacy_path=args.codex_auth)
        if codex_store.names():
            print(f"codex accounts: {', '.join(codex_store.names())}")
        codex_call = make_codex_async_call_provider(
            codex_store, observe=observe, client=provider_http)

        async def codex_stream(request, emit):
            account = codex_store.select_account()
            if account is None:
                from provider_adapters.common import _err
                return _err("auth_error", 0, 0,
                            "no codex access token (run `codex login`)")
            return await stream_codex(
                request, emit, auth=account, observe=observe,
                client=provider_http)
    # The native api_kind adapters (anthropic/bedrock/google) come from the
    # modular provider registry; codex is wired here because its backend takes
    # the `observe` hook that feeds its scarcity-price source.
    _native = providers.native_adapter_handlers(args.timeout_s)
    call_async = make_api_kind_dispatcher(
        default=make_async_call_provider(timeout_s=args.timeout_s,
                                         provider_rules=provider_rules,
                                         client=provider_http),
        handlers={**_native,
                  "openai_codex": codex_call},
    )
    host.set_async_call_hook(call_async)

    # Streaming twins of the same backends (stream: true requests).
    import functools

    from streaming import (
        make_streaming_dispatcher,
        stream_openai_compatible,
    )
    _native_streaming = providers.native_streaming_adapter_handlers(args.timeout_s)
    streaming_call = make_streaming_dispatcher(
        default=functools.partial(stream_openai_compatible,
                                  timeout_s=args.timeout_s,
                                  provider_rules=provider_rules,
                                  client=provider_http),
        # Native providers and Codex have real streaming twins; Codex also feeds
        # `observe` for quota/scarcity pricing.
        handlers={**_native_streaming,
                  "openai_codex": codex_stream},
    )

    from shim import create_app  # local import: keeps argparse errors fast
    # --default-max-tokens 0 means "forward nothing" (strict spec behaviour).
    app = create_app(host, default_profile=args.default_profile,
                     streaming_call=streaming_call,
                     default_max_tokens=args.default_max_tokens or None,
                     codex_store=codex_store)
    closeables = [provider_http]
    if remote_codex is not None:
        closeables.append(remote_codex)
    attach_sources(app, host, catalog=catalog, registry=registry,
                   closeables=closeables)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


def make_discover_hook(registry):
    """Sync discover hook for the core: discovery_id -> marketplace offers.
    Called from Lua inside rank — must be fast and never raise."""
    import time

    by_discovery_id = {}
    for source in registry:
        offers_sync = getattr(source, "offers_sync", None)
        if offers_sync is None:
            continue
        for pid in source.provider_ids:
            by_discovery_id[pid] = offers_sync

    def hook(discovery_id):
        fn = by_discovery_id.get(discovery_id)
        if fn is None:
            return {"ok": False, "error": "unknown discovery_id"}
        try:
            offers = fn(discovery_id)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if not offers:
            # not-ok so the core does NOT cache emptiness for the discovery
            # TTL — a router that starts before the first market dump should
            # pick offers up on the next rank, not minutes later.
            return {"ok": False, "error": "no offers"}
        return {"ok": True, "fetched_at_ms": int(time.time() * 1000),
                "offers": offers}

    return hook


def _enabled(name: str, default: bool = True) -> bool:
    import os
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().lower() not in {
        "0", "false", "no", "off"}


def attach_sources(app, host, catalog=None, registry=None,
                   closeables=()) -> None:
    """Wire the provider-sources refresh loop into the app's lifespan.

    Wraps any existing lifespan so both run. Uses the lifespan API directly:
    FastAPI 0.13x removed the on_event/add_event_handler path."""
    import asyncio
    import contextlib

    import providers
    import sources as sources_mod
    import wallet_keeper

    catalog = catalog if catalog is not None else host.catalog()
    registry = (registry if registry is not None
                else providers.build_source_registry(catalog))
    if any(getattr(s, "offers_sync", None) for s in registry):
        host.set_discover_hook(make_discover_hook(registry))
    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(app_):
        async with inner(app_):
            # COLD START, before the first request: publish last-known credits so
            # the host envelope's antseed funding clause — which fails CLOSED on
            # the engine's `credits` default of 0 — does not reject every antseed
            # candidate until the first balances tick. Reads durable state only.
            seeded = sources_mod.seed_credits(host, registry)
            if seeded:
                print(f"credits seeded for {seeded} provider(s)")
            tasks = (sources_mod.start_refresh_tasks(host, catalog, registry)
                     if _enabled("RUN_SOURCE_REFRESHERS") else [])
            # The autonomous AntSeed funding loop lives HERE, in the router: the
            # buyer identity + sqlite are on an RWO PVC bound to this pod and the
            # sidecar control server is pod-local. Ships dark — it re-reads
            # `antseed.keeper_enabled` (default 0) every cycle.
            keeper = (wallet_keeper.start(catalog)
                      if _enabled("RUN_WALLET_KEEPER") else None)
            if keeper is not None:
                tasks.append(keeper)
            try:
                yield
            finally:
                for t in tasks:
                    t.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                for closeable in closeables:
                    await closeable.aclose()

    app.router.lifespan_context = lifespan


if __name__ == "__main__":
    main()
