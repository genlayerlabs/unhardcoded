"""Request-local allowance for explicitly bounded provider dispatches."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, localcontext
from urllib.parse import urlsplit

PATH = "/_internal/shared-cost/chat"
LIMIT_HEADER = "x-unhardcoded-shared-cost-limit"
EXPIRY_HEADER = "x-unhardcoded-shared-cost-expiry"
CHARGE_HEADER = "x-unhardcoded-shared-cost-charge"
MAX_BODY_BYTES = 256 * 1024
MAX_EXECUTION_S = 120
_log = logging.getLogger(__name__)


def amount(value, *, reported=False) -> Decimal:
    """Match the store's NUMERIC(30,12) without silently rounding a ceiling."""
    try:
        number = Decimal(str(value))
        if not number.is_finite() or not 0 <= number < 10**18:
            raise ValueError("invalid cost")
        with localcontext() as context:
            context.prec = 32
            rounded = number.quantize(Decimal("0.000000000001"), rounding=ROUND_CEILING)
        if rounded >= 10**18 or (not reported and number != rounded):
            raise ValueError("invalid cost precision")
    except InvalidOperation as exc:
        raise ValueError("invalid cost") from exc
    return rounded


def supported_payload(payload):
    automatic = payload.get("_auto_contract")
    return (payload.get("stream", False) is False
            and (automatic is None or (isinstance(automatic, dict)
                                       and isinstance(automatic.get("automatic"), dict)))
            and all(payload.get(field) is None for field in (
                "flow_ir", "tools", "tool_choice", "response_format",
                "reasoning", "reasoning_effort"))
            and (payload.get("max_tokens") is None
                 or (type(payload["max_tokens"]) is int and payload["max_tokens"] > 0))
            and valid_messages(payload.get("messages")))


def valid_messages(messages):
    return (isinstance(messages, list) and bool(messages)
            and all(isinstance(msg, dict) and msg.keys() == {"role", "content"}
                    and msg["role"] in ("system", "user", "assistant", "developer")
                    and isinstance(msg["content"], str) for msg in messages))


@dataclass(frozen=True)
class AttemptLimit:
    max_input_bytes: int
    max_output_tokens: int | None
    max_cost_usd: Decimal


def load_limits():
    """Snapshot operator-attested bounds. Ranking data cannot supply these."""
    limits = {}
    for row in json.loads(os.getenv("SHARED_COST_ATTEMPT_LIMITS", "[]")):
        provider, base_url, model = (row[key] for key in ("provider", "base_url", "model"))
        url = urlsplit(base_url)
        protocol, peer = row.get("protocol", "chat"), row.get("peer_id")
        key = (protocol, provider, base_url, model, peer)
        input_bytes, output_tokens = row["max_input_bytes"], row.get("max_output_tokens")
        if (protocol not in ("chat", "decisions")
                or (peer is not None and (protocol != "chat" or not isinstance(peer, str)
                                          or not peer or not peer.isascii() or not peer.isprintable()))
                or url.scheme != "https" or not url.hostname or url.username is not None
                or url.query or url.fragment or base_url.endswith("/") or key in limits
                or type(input_bytes) is not int or not 0 < input_bytes <= MAX_BODY_BYTES
                or (protocol == "chat" and (type(output_tokens) is not int or output_tokens <= 0))
                or (protocol == "decisions" and output_tokens is not None)):
            raise ValueError("invalid shared cost attempt configuration")
        limits[key] = AttemptLimit(input_bytes, output_tokens, amount(row["max_cost_usd"]))
    return limits


class Allowance:
    def __init__(self, ceiling, limits, output_tokens):
        self.remaining = ceiling
        self.charge = Decimal(0)
        self.limits = limits
        self.output_tokens = output_tokens
        self.dispatched = False
        self.violated = False
        self.denial = None

    def prepare(self, request):
        offer = request.get("offer")
        if offer is not None and not isinstance(offer, dict):
            self.denial = "shared_cost_unbounded"
            return None
        offer = offer or {}
        model = offer.get("wire_model_id") if offer else request.get("served_model_id")
        endpoint, peer = request.get("base_url"), offer.get("peer_id")
        if (not isinstance(model, str) or not model or not isinstance(endpoint, str)
                or (peer is not None and not isinstance(peer, str))
                or (offer and offer.get("seller_endpoint") != endpoint)):
            self.denial = "shared_cost_unbounded"
            return None
        key = ("chat", request.get("provider_id"), endpoint, model, peer)
        limit = self.limits.get(key)
        messages = request.get("messages")
        tokens = request.get("max_tokens")
        if (limit is None or request.get("api_kind") != "openai_compatible"
                or request.get("protocol") not in (None, "chat")
                or not valid_messages(messages) or type(tokens) is not int
                or not 0 < tokens <= min(limit.max_output_tokens, self.output_tokens)):
            self.denial = "shared_cost_unbounded"
            return None
        body = {"model": model, "messages": messages, "max_tokens": tokens}
        try:
            wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except UnicodeError:
            self.denial = "shared_cost_unbounded"
            return None
        bound = self._check(limit, wire)
        if bound is not None:
            return endpoint + "/chat/completions", wire, bound
        return None

    def prepare_decision(self, provider, endpoint, model, wire):
        return self._check(self.limits.get(("decisions", provider, endpoint, model, None)), wire)

    def _check(self, limit, wire):
        if limit is None or len(wire) > limit.max_input_bytes:
            self.denial = "shared_cost_unbounded"
            return None
        if self.violated or limit.max_cost_usd > self.remaining:
            self.denial = "shared_cost_ceiling_exceeded"
            return None
        return limit.max_cost_usd

    def dispatch(self, bound):
        with localcontext() as context:
            context.prec = 32
            self.remaining -= bound
            self.charge += bound
        self.dispatched = True

    def reconcile(self, reported, bound):
        try:
            actual = amount(reported, reported=True)
        except ValueError:
            return
        if actual > bound:
            _log.error("Provider reported a charge above its configured shared cost attempt bound")
            self.violated = True
        with localcontext() as context:
            context.prec = 32
            self.charge += actual - bound
