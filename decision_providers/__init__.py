"""Typed, provider-independent choices. Application code owns authorization."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Protocol


@dataclass(frozen=True)
class DecisionChoice:
    id: str
    description: str


@dataclass(frozen=True)
class DecisionRequest:
    decision_id: str
    instruction: str
    state: dict
    choices: tuple[DecisionChoice, ...]

    def validate(self):
        ids = [c.id for c in self.choices]
        if (not 1 <= len(ids) <= 32 or len(set(ids)) != len(ids)
                or any(not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", i) for i in ids)):
            raise ValueError("Invalid decision choices")
        if any(not c.description or len(c.description) > 1000 for c in self.choices):
            raise ValueError("Invalid choice description")
        if not self.instruction or len(self.instruction) > 2000 or not isinstance(self.state, dict):
            raise ValueError("Invalid decision request")


@dataclass(frozen=True)
class DecisionResult:
    selected_id: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    normalized_entropy: float | None = None
    provider: str = "deterministic"
    model: str = ""
    latency_ms: float = 0
    cost_usd: float | None = 0
    input_tokens: int | None = None
    attempts: int = 0


class DecisionError(RuntimeError):
    def __init__(self, reason, *, cost_usd=None, attempts=0):
        super().__init__(reason)  # fixed reason codes only, never upstream text
        self.reason, self.cost_usd, self.attempts = reason, cost_usd, attempts


class DecisionProvider(Protocol):
    async def decide(self, request: DecisionRequest, *, deadline: float) -> DecisionResult: ...


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def probability_decimals_for_model(model):
    """TypeSafe displays probabilities to two decimals; other models stay strict."""
    if isinstance(model, str) and re.fullmatch(r"jev-(?:latest|\d[\w.-]*)", model.rsplit('/', 1)[-1], re.I):
        return 2
    return None


def probability_sum_valid(values, *, decimal_places=None):
    values = list(values)
    if not values or any(not probability(p) for p in values):
        return False
    if abs(math.fsum(values) - 1) <= 1e-6:
        return True
    # Permit only the declared display precision. Do not apply a blanket loose
    # tolerance to arbitrary high-precision or malformed distributions.
    if decimal_places != 2 or any(abs(p * 100 - round(p * 100)) > 1e-8 for p in values):
        return False
    half_step = .005
    lower = math.fsum(max(0., p - half_step) for p in values)
    upper = math.fsum(min(1., p + half_step) for p in values)
    return lower <= 1 + 1e-12 and upper >= 1 - 1e-12


def validate_answer(answer, choices, *, probability_decimals=None):
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise DecisionError("invalid_response")
    selected = answer.get("choice")
    if not isinstance(selected, str) or selected not in choices:
        raise DecisionError("unknown_choice")
    confidence = answer.get("confidence")
    if confidence is not None and not probability(confidence):
        raise DecisionError("invalid_confidence")
    distribution = answer.get("probabilities")
    entropy = None
    if distribution is not None:
        if (not isinstance(distribution, dict) or set(distribution) != set(choices)
                or any(not probability(p) for p in distribution.values())
                or not probability_sum_valid(distribution.values(), decimal_places=probability_decimals)
                or distribution[selected] + 1e-9 < max(distribution.values())):
            raise DecisionError("invalid_probabilities")
        # Preserve the reported probabilities; normalize only the derived entropy.
        total = math.fsum(distribution.values())
        entropy = (-sum((p / total) * math.log(p / total) for p in distribution.values() if p > 0)
                   / math.log(len(choices))) if len(choices) > 1 else 0.0
        entropy = max(0.0, min(1.0, entropy))
    return selected, distribution, confidence, entropy


@dataclass
class DeterministicDecisionProvider:
    """Explicit selection for tests or an application-owned fallback."""
    selected_id: str

    async def decide(self, request, *, deadline):
        request.validate()
        if self.selected_id not in {c.id for c in request.choices}:
            raise DecisionError("unknown_choice")
        return DecisionResult(self.selected_id)


@dataclass
class RecordedDecisionProvider:
    """Replay by opaque decision ID. Fixtures do not require prompt retention."""
    answers: dict[str, dict] = field(default_factory=dict)

    async def decide(self, request, *, deadline):
        request.validate()
        answer = self.answers.get(request.decision_id)
        selected, probs, confidence, entropy = validate_answer(answer, [c.id for c in request.choices])
        return DecisionResult(selected, probs, confidence, entropy, provider="recorded")
