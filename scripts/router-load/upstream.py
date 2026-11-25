"""Local OpenAI-compatible fixture with a delay before the first response."""

import asyncio
import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
DELAY = float(os.environ["LOAD_LATENCY_MS"]) / 1000
FIELDS = {"id": "load", "created": 1, "model": "load-model"}
USAGE = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}


@app.post("/v1/chat/completions")
async def chat(body: dict):
    await asyncio.sleep(DELAY)
    if not body.get("stream"):
        return {
            **FIELDS,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                         "finish_reason": "stop"}],
            "usage": USAGE,
        }

    async def events():
        for delta, reason in [({"role": "assistant", "content": "OK"}, None), ({}, "stop")]:
            chunk = {
                **FIELDS,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        if (body.get("stream_options") or {}).get("include_usage"):
            yield f"data: {json.dumps({**FIELDS, 'choices': [], 'usage': USAGE})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
