"""Operator-configured paths through the ESLA caller sidecar."""
from __future__ import annotations

import json
import re
from urllib.parse import unquote, urlsplit, urlunsplit


class InvalidUpstreamPath(ValueError):
    """An upstream path could escape the configured sidecar route."""


def load_routes(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        entries = json.loads(raw)
        if not isinstance(entries, dict):
            raise ValueError
        routes = {}
        for provider, entry in entries.items():
            if not provider or not isinstance(entry, dict):
                raise ValueError
            if not re.fullmatch(r"0x[0-9a-fA-F]{40}", entry["agreement"]):
                raise ValueError
            url = urlsplit(entry["sidecar_url"])
            port = url.port  # Validate the port even before a request is made.
            if (url.scheme not in {"http", "https"} or not url.hostname
                    or url.username is not None or url.query or url.fragment
                    or port == 0 or not re.fullmatch(r"/[A-Za-z0-9_-]+/?", url.path)):
                raise ValueError
            routes[provider] = url.geturl().rstrip("/")
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError(
            "ESLA_ROUTES must map provider IDs to {agreement, sidecar_url} "
            "with an agreement address and a caller URL ending in /<route>"
        ) from None
    return routes


def upstream_url(request: dict, url: str) -> str:
    """Keep the adapter's wire path and query under the caller's route prefix."""
    sidecar = request.get("_esla_sidecar_url")
    if sidecar is None:
        return url
    target = urlsplit(url)
    path = unquote(target.path)
    if not path.startswith("/") or any(part in {".", ".."} for part in path.split("/")):
        raise InvalidUpstreamPath
    caller = urlsplit(sidecar)
    return urlunsplit((caller.scheme, caller.netloc, caller.path + target.path,
                       target.query, ""))
