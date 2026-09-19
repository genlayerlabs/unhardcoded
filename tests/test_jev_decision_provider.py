import asyncio
import json
import time

import httpx
import pytest

from decision_providers import DecisionChoice, DecisionError, DecisionRequest, validate_answer
from decision_providers.jev import JevDecisionProvider


def request():
    return DecisionRequest("opaque-id", "Choose the workload", {"task": "Write code"}, (
        DecisionChoice("general", "General help"), DecisionChoice("coding", "Code work")))


def response(**changes):
    return {"model": "typesafe/jev-1.13-20260917", "answers": {"selection": {
        "type": "choice", "choice": "coding", "probabilities": {"general": .1, "coding": .9},
        "confidence": .7, **changes}}, "usage": {"input_tokens": 100, "cost": .0000042}}


@pytest.mark.asyncio
async def test_official_openrouter_choice_contract():
    async def handler(req):
        assert str(req.url) == "https://openrouter.ai/api/alpha/decisions"
        assert req.headers["authorization"] == "Bearer tenant-key"
        body = json.loads(req.content)
        assert body == {"model": "typesafe/jev-1.13", "state": {"task": "Write code"},
            "questions": {"selection": {"type": "choice", "instructions": "Choose the workload",
                "criteria": {"general": "General help", "coding": "Code work"}}}}
        return httpx.Response(200, json=response())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JevDecisionProvider("tenant-key", client=client).decide(request(), deadline=time.monotonic() + 1)
    assert result.selected_id == "coding" and result.cost_usd == .0000042
    assert result.confidence == .7  # not max(probabilities)
    assert 0 < result.normalized_entropy < 1 and result.attempts == 1


@pytest.mark.parametrize("changes", [{"choice": "shell"}, {"probabilities": {"coding": 1}},
    {"probabilities": {"coding": .2, "general": .1}}, {"confidence": float("nan")},
    {"probabilities": {"coding": -.1, "general": 1.1}},
    {"probabilities": {"coding": .1, "general": .9}}, {"confidence": True}])
def test_malformed_choices_are_rejected(changes):
    with pytest.raises(DecisionError):
        validate_answer(response(**changes)["answers"]["selection"], ["general", "coding"])


def test_selection_only_does_not_fabricate_entropy():
    result = validate_answer({"type": "choice", "choice": "coding"}, ["general", "coding"])
    assert result == ("coding", None, None, None)


@pytest.mark.asyncio
async def test_timeout_and_cancellation_do_not_leave_work_running():
    cancelled = asyncio.Event()
    async def handler(req):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = JevDecisionProvider("key", client=client)
        with pytest.raises(DecisionError, match="timeout"):
            await provider.decide(request(), deadline=time.monotonic() + .01)
        assert cancelled.is_set()
        task = asyncio.create_task(provider.decide(request(), deadline=time.monotonic() + 10))
        await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [(400, 1), (401, 1), (429, 2), (503, 2), (302, 1)])
async def test_only_transient_errors_retry_and_redirects_are_not_followed(status, expected):
    calls = []
    async def handler(req):
        calls.append(req)
        return httpx.Response(status, headers={"Location": "https://untrusted.test"}, json={"error": "secret"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DecisionError) as error:
            await JevDecisionProvider("key", client=client, max_retries=1).decide(request(), deadline=time.monotonic() + 1)
    assert len(calls) == expected and "secret" not in str(error.value)


@pytest.mark.asyncio
async def test_duplicate_json_and_oversize_responses_are_refused():
    for raw in (b'{"model":"one","model":"two"}', b"x" * 65537):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=raw))) as client:
            with pytest.raises(DecisionError):
                await JevDecisionProvider("key", client=client).decide(request(), deadline=time.monotonic() + 1)


@pytest.mark.asyncio
async def test_missing_credentials_never_use_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "operator-secret")
    with pytest.raises(DecisionError, match="missing_credentials"):
        await JevDecisionProvider("").decide(request(), deadline=time.monotonic() + 1)
