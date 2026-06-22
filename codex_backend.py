"""
codex_backend.py — call_provider for api_kind="openai_codex": the ChatGPT
subscription path via the Codex Responses endpoint
(POST https://chatgpt.com/backend-api/codex/responses), authenticated with the
token from `codex login` (see codex_auth.CodexAuth).

UNOFFICIAL / ToS-RISKY and undocumented — the endpoint shape can change without
notice. See docs/OPENAI-CODEX.md. The pure translation/aggregation helpers are
unit-tested; the live streaming call is not (no subscription in CI).
"""
from __future__ import annotations

import json
from typing import Any, Iterable

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _err(kind: str, status: int, latency_ms: int, message: str) -> dict:
    return {"ok": False, "error_kind": kind, "http_status": status,
            "latency_ms": latency_ms, "error_message": message}


def _content_to_text(content: Any) -> str:
    """Coerce a chat message `content` (str | None | content-parts list) to a
    plain string for the Responses API string-content shorthand."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


def _messages_to_input(messages: list[dict]) -> list[dict]:
    """Chat-completions messages → Responses API `input` items.

    A plain message maps to `{role, content}` (string-content shorthand). Two
    cases need STRUCTURAL translation or the Codex Responses endpoint 400s the
    whole request (turning every tool-using turn after the first into a 502):

      - an assistant message carrying `tool_calls` → one `function_call` item per
        call (`{type, call_id, name, arguments}`); any text content is kept as a
        separate assistant message item. (The Responses API has no `tool_calls`
        field on a message — it models the call as its own input item.)
      - a `tool` result message → a `function_call_output` item
        (`{type, call_id, output}`); the Responses API has no `tool` role.

    Without this, a conversation whose history includes assistant tool_calls +
    their tool results is rejected, so a multi-turn agent replies once and then
    goes silent.
    """
    out: list[dict] = []
    for m in messages or []:
        role = m.get("role") or "user"
        tool_calls = m.get("tool_calls")

        if role == "tool":
            out.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id") or m.get("call_id") or "",
                "output": _content_to_text(m.get("content")),
            })
            continue

        if role == "assistant" and tool_calls:
            text = _content_to_text(m.get("content"))
            if text:
                out.append({"role": "assistant", "content": text})
            for tc in tool_calls:
                fn = tc.get("function") or {}
                out.append({
                    "type": "function_call",
                    "call_id": tc.get("id") or "",
                    "name": fn.get("name") or "",
                    "arguments": fn.get("arguments") or "{}",
                })
            continue

        out.append({"role": role, "content": _content_to_text(m.get("content"))})
    return out


def _to_responses_tools(tools: Any) -> "list[dict] | None":
    """Chat-completions tools -> Responses API tools. Chat-completions nests the
    schema under "function" ({type:"function", function:{name,...}}); the Responses
    API wants it FLAT ({type:"function", name, description, parameters})."""
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if t.get("type") == "function" and isinstance(fn, dict):
            flat: dict = {"type": "function", "name": fn.get("name")}
            if fn.get("description") is not None:
                flat["description"] = fn["description"]
            if fn.get("parameters") is not None:
                flat["parameters"] = fn["parameters"]
            out.append(flat)
        else:
            out.append(t)  # already flat / non-function tool — pass through
    return out or None


def _to_responses_tool_choice(tc: Any) -> Any:
    """Strings ("auto"/"required"/"none") pass through. The named form differs:
    chat-completions {type:"function", function:{name}} -> Responses {type:"function", name}."""
    if tc is None or isinstance(tc, str):
        return tc
    if isinstance(tc, dict) and tc.get("type") == "function":
        fn = tc.get("function")
        name = fn.get("name") if isinstance(fn, dict) else tc.get("name")
        if name:
            return {"type": "function", "name": name}
    return None


def build_codex_body(request: dict) -> dict:
    """Build the Responses API request body from a router request."""
    body: dict = {
        "model":  request["served_model_id"],
        "instructions": request.get("instructions") or "You are a concise assistant.",
        "input":  _messages_to_input(request.get("messages") or []),
        "stream": True,   # the Codex endpoint streams SSE
        "store": False,   # ChatGPT-account Codex endpoint requires this.
    }
    # Forward tool definitions so the model emits NATIVE function calls
    # (response.function_call_arguments.*) instead of describing the call as text.
    # Without this the model improvises "<tool_call>{...}" prose with finish="stop"
    # and every tool-using agent breaks. tool_choice is optional (auto by default).
    tools = _to_responses_tools(request.get("tools"))
    if tools:
        body["tools"] = tools
        tc = _to_responses_tool_choice(request.get("tool_choice"))
        if tc is not None:
            body["tool_choice"] = tc
    # The ChatGPT-account Codex endpoint rejects some public Responses API
    # params even though they are accepted elsewhere. Do not forward max_tokens
    # as max_output_tokens, and do not forward temperature; live endpoint errors
    # include: {"detail":"Unsupported parameter: temperature"}.
    return body


def build_codex_headers(token: str, account_id: str | None,
                        extra: dict[str, str] | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "text/event-stream",
        "originator":    "codex_cli_rs",
        "User-Agent":    "codex_cli_rs",
    }
    if account_id:
        headers["chatgpt-account-id"] = account_id
    if extra:
        headers.update(extra)
    return headers


def aggregate_codex_sse(lines: Iterable[str], latency_ms: int) -> dict:
    """Fold a Codex Responses SSE stream into the router's response shape.

    Recognized events (by the `type` field of each `data:` JSON object):
      - response.output_text.delta             -> append `delta` to the text
      - response.output_item.added (function_call) -> begin a native tool call
      - response.function_call_arguments.delta -> accumulate that call's arguments
      - response.function_call_arguments.done  -> the call's full arguments string
      - response.completed                     -> capture usage + finish
      - response.failed / error                -> map to an error

    A function call makes the result `finish_reason: "tool_calls"` with a native
    OpenAI-shaped `tool_calls` array, so the shim presents it like any other
    function-calling model. Pure text streams are unchanged.
    """
    text_parts: list[str] = []
    finish_reason = "stop"
    usage: dict = {}
    err: dict | None = None
    # function_call items keyed by their streaming item id, in arrival order.
    fcalls: dict = {}
    fcorder: list = []

    for line in lines:
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            ev = json.loads(payload)
        except ValueError:
            continue
        etype = ev.get("type")
        if etype == "response.output_text.delta":
            if ev.get("delta"):
                text_parts.append(ev["delta"])
        elif etype == "response.output_item.added":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                iid = item.get("id")
                fcalls[iid] = {
                    "call_id": item.get("call_id") or iid,
                    "name": item.get("name"),
                    "args": [],
                    "done": None,
                }
                fcorder.append(iid)
        elif etype == "response.function_call_arguments.delta":
            iid = ev.get("item_id")
            if iid in fcalls and ev.get("delta"):
                fcalls[iid]["args"].append(ev["delta"])
        elif etype == "response.function_call_arguments.done":
            iid = ev.get("item_id")
            if iid in fcalls and ev.get("arguments") is not None:
                fcalls[iid]["done"] = ev["arguments"]
        elif etype == "response.completed":
            resp = ev.get("response") or {}
            usage = resp.get("usage") or usage
            if resp.get("status") == "incomplete":
                finish_reason = "length"
        elif etype in ("response.failed", "error"):
            msg = (ev.get("response") or ev).get("error") or ev.get("message") or "codex stream failed"
            err = _err("server_error", 0, latency_ms, str(msg))

    if err is not None:
        return err

    tool_calls: "list[dict] | None" = None
    if fcorder:
        tool_calls = []
        for iid in fcorder:
            c = fcalls[iid]
            args = c["done"] if c["done"] is not None else "".join(c["args"])
            tool_calls.append({
                "id": c["call_id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": args or "{}"},
            })
        finish_reason = "tool_calls"

    return {
        "ok": True,
        "latency_ms": latency_ms,
        "response": {
            "text":          "".join(text_parts),
            "tool_calls":    tool_calls,
            "finish_reason": finish_reason,
            "tokens_in":     usage.get("input_tokens"),
            "tokens_out":    usage.get("output_tokens"),
            "tokens_total":  usage.get("total_tokens"),
            "raw_model":     None,
        },
    }


def make_codex_async_call_provider(
    auth,
    base_url: str = CODEX_BASE_URL,
    timeout_s: float = 120.0,
    extra_headers: dict[str, str] | None = None,
    observe=None,
):
    """Async call_provider for api_kind="openai_codex". `auth` is a
    codex_auth.CodexAuth (or anything with access_token()/account_id()).

    `observe(signal)` is the passive quota feed: called once per attempt with
    {"status", "headers" (ratelimit/usage/quota only), "ts"} — polling the
    unofficial endpoint would burn the quota it measures, so observation of
    real traffic is the only safe signal."""
    import re as _re
    import time
    import httpx

    def _notify(status: int, headers=None) -> None:
        if observe is None:
            return
        try:
            hdrs = {k.lower(): v for k, v in dict(headers or {}).items()
                    if _re.search(r"ratelimit|usage|quota|percent", k, _re.I)}
            observe({"status": status, "headers": hdrs, "ts": int(time.time())})
        except Exception:
            pass

    async def call(request: dict) -> dict:
        # Pick the account ONCE per call (advances round-robin in balanced mode),
        # so token + account_id come from the same account.
        acct = auth.select_account() if hasattr(auth, "select_account") else auth
        token = acct.access_token() if acct else None
        if not token:
            _notify(0)
            return _err("auth_error", 0, 0, "no codex access token (run `codex login`)")
        body = build_codex_body(request)
        headers = build_codex_headers(token, acct.account_id(), extra_headers)
        url = (request.get("base_url") or base_url).rstrip("/") + "/responses"
        timeout = (request.get("timeout_ms") or int(timeout_s * 1000)) / 1000.0

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                async with c.stream("POST", url, json=body, headers=headers) as resp:
                    _notify(resp.status_code, resp.headers)
                    latency = int((time.monotonic() - t0) * 1000)
                    if resp.status_code == 401:
                        return _err("auth_error", 401, latency, "codex token rejected")
                    if resp.status_code == 429:
                        return _err("rate_limit", 429, latency, "codex rate limited")
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                        return _err("server_error", resp.status_code, latency, detail)
                    lines = [line async for line in resp.aiter_lines()]
            return aggregate_codex_sse(lines, int((time.monotonic() - t0) * 1000))
        except httpx.TimeoutException:
            _notify(0)
            return _err("timeout", 0, int((time.monotonic() - t0) * 1000), "codex request timed out")
        except (httpx.NetworkError, httpx.RequestError) as e:
            _notify(0)
            return _err("network_error", 0, int((time.monotonic() - t0) * 1000), str(e))

    return call
