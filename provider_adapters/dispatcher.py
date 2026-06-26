"""api_kind dispatcher for non-streaming provider adapters."""
from __future__ import annotations

from provider_adapters.common import AsyncCallProviderHook


def make_api_kind_dispatcher(
    default: AsyncCallProviderHook,
    handlers: dict[str, AsyncCallProviderHook] | None = None,
) -> AsyncCallProviderHook:
    """Route each call to a per-api_kind async handler."""
    _handlers = dict(handlers or {})

    async def call(request: dict) -> dict:
        handler = _handlers.get(request.get("api_kind"), default)
        return await handler(request)

    return call
