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

from llm_router_host import (  # noqa: E402
    LLMRouterHost,
    make_async_call_provider,
    make_api_kind_dispatcher,
)


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

    # Operator-managed provider keys persisted by the dashboard live on the PVC
    # (.env.secrets) and are the source of truth. Load them over the container
    # env BEFORE LLMRouterHost snapshots os.environ, so dashboard edits to
    # heurist/ionet/openrouter keys take effect and survive pod restarts.
    from env_secrets import load_env_secrets  # noqa: E402
    loaded = load_env_secrets()
    if loaded:
        print(f"env secrets loaded from PVC: {len(loaded)} keys")

    # api_kind=openai_codex is served by a dedicated backend (Codex Responses
    # endpoint + codex login token); everything else uses the OpenAI-compatible
    # backend. The Codex backend reads auth.json lazily on first use.
    from codex_auth import CodexAuthStore     # noqa: E402
    from codex_backend import make_codex_async_call_provider  # noqa: E402

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
    import sources as sources_mod
    catalog = host.catalog()
    registry = sources_mod.build_registry(catalog)
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
    # Multi-account: self-discovers /codex/accounts/*.json (managed from the
    # dashboard) and keeps the legacy single /codex/auth.json as the `default`
    # account. Drop-in for a single CodexAuth (serves the first account until
    # the policy drives per-call selection — a follow-up).
    codex_auth = CodexAuthStore(legacy_path=args.codex_auth)
    if codex_auth.names():
        print(f"codex accounts: {', '.join(codex_auth.names())}")
    call_async = make_api_kind_dispatcher(
        default=make_async_call_provider(timeout_s=args.timeout_s,
                                         provider_rules=provider_rules),
        handlers={"openai_codex": make_codex_async_call_provider(codex_auth, observe=observe)},
    )
    host.set_async_call_hook(call_async)

    # Streaming twins of the same backends (stream: true requests).
    import functools

    from streaming import (
        make_streaming_dispatcher,
        stream_codex,
        stream_openai_compatible,
    )
    streaming_call = make_streaming_dispatcher(
        default=functools.partial(stream_openai_compatible,
                                  timeout_s=args.timeout_s,
                                  provider_rules=provider_rules),
        handlers={"openai_codex": functools.partial(stream_codex,
                                                    auth=codex_auth,
                                                    observe=observe)},
    )

    from shim import create_app  # local import: keeps argparse errors fast
    # --default-max-tokens 0 means "forward nothing" (strict spec behaviour).
    app = create_app(host, default_profile=args.default_profile,
                     streaming_call=streaming_call,
                     default_max_tokens=args.default_max_tokens or None,
                     codex_store=codex_auth)
    attach_sources(app, host, catalog=catalog, registry=registry)

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


def attach_sources(app, host, catalog=None, registry=None) -> None:
    """Wire the provider-sources refresh loop into the app's lifespan.

    Wraps any existing lifespan so both run. Uses the lifespan API directly:
    FastAPI 0.13x removed the on_event/add_event_handler path."""
    import contextlib

    import sources as sources_mod

    catalog = catalog if catalog is not None else host.catalog()
    registry = registry if registry is not None else sources_mod.build_registry(catalog)
    if any(getattr(s, "offers_sync", None) for s in registry):
        host.set_discover_hook(make_discover_hook(registry))
    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(app_):
        async with inner(app_):
            tasks = sources_mod.start_refresh_tasks(host, catalog, registry)
            try:
                yield
            finally:
                for t in tasks:
                    t.cancel()

    app.router.lifespan_context = lifespan


if __name__ == "__main__":
    main()
