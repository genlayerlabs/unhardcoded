"""Bounded transport/usage evidence, without prompts, credentials or reasoning text."""
from __future__ import annotations

import math
import time


TEXT_FIELDS = frozenset({
    "mode", "phase", "timeout_source", "upstream_id", "upstream_provider",
    "provider_id", "model_family", "error_kind", "requested_reasoning_effort",
})
NUMBER_FIELDS = frozenset({
    "http_status", "elapsed_ms", "request_sent_ms", "response_headers_ms",
    "body_complete_ms", "connect_ms", "tls_ms", "first_output_ms",
    "first_reasoning_ms", "tokens_reasoning", "requested_timeout_ms",
    "requested_max_tokens", "attempt",
})


def bounded_diagnostics(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    out = {key: value[key][:160] for key in TEXT_FIELDS
           if isinstance(value.get(key), str)}
    out.update({key: value[key] for key in NUMBER_FIELDS
                if isinstance(value.get(key), (int, float))
                and not isinstance(value[key], bool)
                and 0 <= value[key] <= 1e15 and math.isfinite(value[key])})
    if isinstance(value.get("requested_reasoning_enabled"), bool):
        out["requested_reasoning_enabled"] = value["requested_reasoning_enabled"]
    return out


def upstream_metadata(data: dict) -> dict:
    usage = data.get("usage") or {}
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details") or {}
    details = details if isinstance(details, dict) else {}
    return bounded_diagnostics({
        "upstream_id": data.get("id"),
        "upstream_provider": data.get("provider"),
        "tokens_reasoning": details.get("reasoning_tokens"),
    })


class ProviderTiming:
    def __init__(self, request: dict, timeout: float, mode: str):
        self.started = time.monotonic()
        self.starts: dict[str, float] = {}
        reasoning = request.get("reasoning")
        reasoning = reasoning if isinstance(reasoning, dict) else {}
        self.data = bounded_diagnostics({
            "mode": mode, "phase": "peer_capacity",
            "requested_timeout_ms": timeout * 1000,
            "requested_max_tokens": request.get("max_tokens"),
            "requested_reasoning_effort": reasoning.get("effort", request.get("reasoning_effort")),
            "requested_reasoning_enabled": reasoning.get("enabled"),
        })

    def elapsed(self) -> float:
        return round((time.monotonic() - self.started) * 1000, 3)

    async def trace(self, event: str, info: dict) -> None:
        # HTTP Core's trace info can contain URLs, headers and exceptions. Read
        # only the event name and numeric HTTP status; never retain info itself.
        operation, _, outcome = event.rpartition(".")
        operation = operation.rsplit(".", 1)[-1]
        now = time.monotonic()
        phases = {"connect_tcp": "connect", "start_tls": "tls",
                  "send_request_headers": "request_headers",
                  "send_request_body": "request_body",
                  "receive_response_headers": "response_headers",
                  "receive_response_body": "response_body"}
        if operation not in phases:
            return
        if outcome == "started":
            self.data["phase"] = phases[operation]
            self.starts[operation] = now
        elif outcome == "complete":
            if operation in ("connect_tcp", "start_tls") and operation in self.starts:
                key = "connect_ms" if operation == "connect_tcp" else "tls_ms"
                self.data[key] = round((now - self.starts[operation]) * 1000, 3)
            key = {"send_request_body": "request_sent_ms",
                   "receive_response_headers": "response_headers_ms",
                   "receive_response_body": "body_complete_ms"}.get(operation)
            if key:
                self.data[key] = self.elapsed()
            if operation == "receive_response_headers":
                value = info.get("return_value")
                if isinstance(value, tuple) and len(value) >= 2 and isinstance(value[1], int):
                    self.data["http_status"] = value[1]
                elif isinstance(value, tuple) and value and isinstance(value[0], int):
                    self.data["http_status"] = value[0]  # HTTP/2 has no HTTP version prefix.

    def http_options(self, client) -> dict:
        # Only HTTPX implements this tracing extension. Custom transports/test
        # clients retain their existing protocol and receive no extra arguments.
        import httpx
        if isinstance(client, httpx.AsyncClient):
            return {"extensions": {"trace": self.trace}}
        return {}

    def observe_chunk(self, chunk: dict) -> None:
        self.data.update(upstream_metadata(chunk))
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning") or delta.get("reasoning_details"):
                self.data.setdefault("first_reasoning_ms", self.elapsed())
            if delta.get("content") or delta.get("tool_calls"):
                self.data.setdefault("first_output_ms", self.elapsed())

    def attach(self, result: dict) -> dict:
        nested = result.get("diagnostics") or {}
        metadata = (result.get("response") or {}).get("upstream") or {}
        result["diagnostics"] = bounded_diagnostics({
            **self.data, **nested, **metadata, "elapsed_ms": self.elapsed(),
        })
        return result
