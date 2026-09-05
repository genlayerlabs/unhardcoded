"""Internal Codex credential/call broker for horizontally scaled routers.

Exactly one deployment owns the Codex account PVC. Stateless API replicas call
this service with an IaC-managed bearer token; raw account credentials never
leave the broker process or appear in its responses/logs.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from codex_auth import CodexAuthStore
from codex_backend import make_codex_async_call_provider
from provider_adapters.common import _err
from streaming import stream_codex


def _bearer(request: Request) -> str:
    value = request.headers.get("authorization") or ""
    return value[7:].strip() if value.lower().startswith("bearer ") else ""


def create_app(*, store: Any = None, token: str | None = None,
               client: Any = None) -> FastAPI:
    expected_token = (token if token is not None
                      else os.getenv("CODEX_BROKER_TOKEN", "")).strip()
    accounts_dir = Path(os.getenv("CODEX_ACCOUNTS_DIR", "/codex/accounts"))
    legacy_path_raw = os.getenv("CODEX_AUTH_PATH", "").strip()
    account_store = store or CodexAuthStore(
        accounts_dir, legacy_path=Path(legacy_path_raw) if legacy_path_raw else None)
    owns_client = client is None

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        if client is None:
            app_.state.http = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=50,
                                    max_keepalive_connections=20))
        else:
            app_.state.http = client
        try:
            yield
        finally:
            if owns_client:
                await app_.state.http.aclose()

    app = FastAPI(title="unhardcoded Codex broker", docs_url=None,
                  redoc_url=None, lifespan=lifespan)
    app.state.store = account_store
    app.state.http = client

    def authorized(request: Request) -> JSONResponse | None:
        if not expected_token:
            return JSONResponse(status_code=503, content={
                "error": "CODEX_BROKER_TOKEN is not configured"})
        supplied = _bearer(request)
        if not supplied or not hmac.compare_digest(supplied, expected_token):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return None

    async def request_body(request: Request) -> tuple[dict | None, JSONResponse | None]:
        try:
            body = await request.json()
        except Exception:
            return None, JSONResponse(status_code=400, content={
                "error": "request body must be valid JSON"})
        if not isinstance(body, dict):
            return None, JSONResponse(status_code=400, content={
                "error": "request body must be an object"})
        return body, None

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        configured = bool(expected_token)
        return JSONResponse(status_code=200 if configured else 503, content={
            "ok": configured,
            "accounts": len(account_store.names()),
        })

    @app.post("/v1/reload")
    async def reload_accounts(request: Request) -> JSONResponse:
        denied = authorized(request)
        if denied:
            return denied
        names = await asyncio.to_thread(account_store.reload)
        return JSONResponse(content={"ok": True, "accounts": names})

    @app.post("/v1/call")
    async def call(request: Request) -> JSONResponse:
        denied = authorized(request)
        if denied:
            return denied
        body, error = await request_body(request)
        if error:
            return error
        signals: list[dict] = []
        backend = make_codex_async_call_provider(
            account_store, observe=signals.append, client=app.state.http)
        try:
            result = await backend(body or {})
        except Exception as exc:  # noqa: BLE001
            result = _err("server_error", 0, 0,
                          f"Codex broker call failed: {type(exc).__name__}: {exc}")
        return JSONResponse(content={"result": result, "signals": signals})

    @app.post("/v1/stream")
    async def stream(request: Request):
        denied = authorized(request)
        if denied:
            return denied
        body, error = await request_body(request)
        if error:
            return error

        async def events():
            queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=128)
            signals: list[dict] = []

            async def emit(delta: str) -> None:
                await queue.put({"event": "delta", "delta": delta})

            async def run() -> None:
                account = account_store.select_account()
                if account is None:
                    result = _err("auth_error", 0, 0,
                                  "no Codex account is configured")
                else:
                    try:
                        result = await stream_codex(
                            body or {}, emit, auth=account,
                            observe=signals.append, client=app.state.http)
                    except Exception as exc:  # noqa: BLE001
                        result = _err(
                            "server_error", 0, 0,
                            f"Codex broker stream failed: {type(exc).__name__}: {exc}")
                await queue.put({"event": "result", "result": result,
                                 "signals": signals})

            task = asyncio.create_task(run())
            try:
                while True:
                    event = await queue.get()
                    yield json.dumps(event, separators=(",", ":")) + "\n"
                    if event.get("event") == "result":
                        break
            finally:
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        return StreamingResponse(events(), media_type="application/x-ndjson")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("CODEX_BROKER_HOST", "0.0.0.0"),
                port=int(os.getenv("CODEX_BROKER_PORT", "8090")))
