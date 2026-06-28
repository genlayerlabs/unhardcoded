"""Unit tests for responses_api.py — the inbound Responses-API translation
(mirror of codex_backend.py). Pure, no host."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import codex_backend as cb  # noqa: E402
import responses_api as ra  # noqa: E402


# ---- input_to_messages -------------------------------------------------

def test_input_string_becomes_user_message():
    assert ra.input_to_messages("hi") == [{"role": "user", "content": "hi"}]


def test_instructions_become_leading_system_message():
    msgs = ra.input_to_messages("hi", instructions="be terse")
    assert msgs == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_input_items_plain_messages():
    items = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert ra.input_to_messages(items) == items


def test_input_content_parts_are_flattened():
    items = [{"role": "user", "content": [
        {"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]}]
    assert ra.input_to_messages(items) == [{"role": "user", "content": "ab"}]


def test_message_type_wrapper_is_unwrapped():
    items = [{"type": "message", "role": "user", "content": "hi"}]
    assert ra.input_to_messages(items) == [{"role": "user", "content": "hi"}]


def test_function_call_output_becomes_tool_message():
    items = [{"type": "function_call_output", "call_id": "c1", "output": "42"}]
    assert ra.input_to_messages(items) == [
        {"role": "tool", "tool_call_id": "c1", "content": "42"}]


def test_developer_role_is_normalized_to_system():
    # Codex sends a `developer` role item; map it to system for provider portability.
    items = [{"type": "message", "role": "developer",
              "content": [{"type": "input_text", "text": "rules"}]}]
    assert ra.input_to_messages(items) == [{"role": "system", "content": "rules"}]


def test_reasoning_and_unknown_items_skipped():
    items = [{"type": "reasoning", "summary": []},
             {"type": "weird_future_type"},
             {"role": "user", "content": "hi"}]
    assert ra.input_to_messages(items) == [{"role": "user", "content": "hi"}]


def test_consecutive_function_calls_merge_onto_one_assistant_message():
    items = [
        {"role": "user", "content": "weather?"},
        {"type": "function_call", "call_id": "c1", "name": "get_weather",
         "arguments": '{"city":"BCN"}'},
        {"type": "function_call", "call_id": "c2", "name": "get_time",
         "arguments": "{}"},
    ]
    assert ra.input_to_messages(items) == [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"BCN"}'}},
            {"id": "c2", "type": "function",
             "function": {"name": "get_time", "arguments": "{}"}},
        ]},
    ]


def test_assistant_text_then_function_call_merge_into_one_message():
    items = [
        {"role": "assistant", "content": "checking…"},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
    ]
    assert ra.input_to_messages(items) == [
        {"role": "assistant", "content": "checking…", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
    ]


def test_roundtrip_inverse_of_messages_to_input():
    # The canonical tool-history conversation that 400d live (test_codex.py).
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "weather in BCN?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"BCN"}'}},
            {"id": "call_2", "type": "function",
             "function": {"name": "get_time", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "22C sunny"},
        {"role": "tool", "tool_call_id": "call_2", "content": "14:00"},
        {"role": "assistant", "content": "It's 22C."},
        {"role": "user", "content": "and tomorrow?"},
    ]
    # item-level inverse: _messages_to_input keeps the system message as an
    # input ITEM (instructions is a separate Responses field, covered by its own
    # test), so the clean round-trip passes no instructions.
    items = cb._messages_to_input(msgs)
    rebuilt = ra.input_to_messages(items)
    assert rebuilt == msgs


# ---- tools_to_chat / tool_choice_to_chat -------------------------------

def test_tools_to_chat_nests_function_tools():
    flat = [{"type": "function", "name": "shell", "description": "run",
             "parameters": {"type": "object", "properties": {}}}]
    assert ra.tools_to_chat(flat) == [{"type": "function", "function": {
        "name": "shell", "description": "run",
        "parameters": {"type": "object", "properties": {}}}}]


def test_tools_to_chat_is_inverse_of_to_responses_tools():
    nested = [{"type": "function", "function": {
        "name": "shell", "description": "run a shell command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}]
    flat = cb._to_responses_tools(nested)
    assert ra.tools_to_chat(flat) == nested


def test_tools_to_chat_drops_unknown_tool_types():
    # native/unknown tools would 400 a chat provider -> dropped, not forwarded.
    flat = [{"type": "local_shell"}, {"type": "web_search"}]
    assert ra.tools_to_chat(flat) is None


def test_tools_to_chat_keeps_function_drops_native_when_mixed():
    flat = [{"type": "function", "name": "shell"}, {"type": "local_shell"}]
    assert ra.tools_to_chat(flat) == [
        {"type": "function", "function": {"name": "shell"}}]


def test_dropped_tool_types_reports_dropped():
    flat = [{"type": "function", "name": "shell"}, {"type": "local_shell"},
            {"type": "web_search"}, {"type": "function"}]  # nameless function drops too
    assert ra.dropped_tool_types(flat) == ["local_shell", "web_search", "function"]
    assert ra.dropped_tool_types(None) == []
    assert ra.dropped_tool_types([{"type": "function", "name": "ok"}]) == []


def test_tools_to_chat_empty_is_none():
    assert ra.tools_to_chat(None) is None
    assert ra.tools_to_chat([]) is None


def test_tool_choice_to_chat_passthrough_and_named():
    assert ra.tool_choice_to_chat("auto") == "auto"
    assert ra.tool_choice_to_chat(None) is None
    assert ra.tool_choice_to_chat({"type": "function", "name": "shell"}) == {
        "type": "function", "function": {"name": "shell"}}


def test_tool_choice_to_chat_is_inverse_of_to_responses():
    nested = {"type": "function", "function": {"name": "shell"}}
    flat = cb._to_responses_tool_choice(nested)
    assert ra.tool_choice_to_chat(flat) == nested


# ---- result_to_responses_object ----------------------------------------

def _result(text="", tool_calls=None, finish="stop",
            tin=7, tout=3, ttot=10, raw_model="mock-model"):
    return {"ok": True, "response": {
        "text": text, "tool_calls": tool_calls, "finish_reason": finish,
        "tokens_in": tin, "tokens_out": tout, "tokens_total": ttot,
        "raw_model": raw_model}, "chosen": {"served_model_id": "srv"}}


def test_object_text_only():
    obj = ra.result_to_responses_object(_result(text="hello"), "policy:auto",
                                        response_id="resp_x", created_at=111,
                                        msg_id="msg_x")
    assert obj["id"] == "resp_x"
    assert obj["object"] == "response"
    assert obj["created_at"] == 111
    assert obj["status"] == "completed"
    assert obj["model"] == "mock-model"
    assert obj["output"] == [{
        "type": "message", "id": "msg_x", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hello", "annotations": []}]}]
    assert obj["usage"] == {"input_tokens": 7, "output_tokens": 3,
                            "total_tokens": 10}


def test_object_tool_calls_only_has_no_message_item():
    tcs = [{"id": "call_1", "type": "function",
            "function": {"name": "shell", "arguments": '{"command":"ls"}'}}]
    obj = ra.result_to_responses_object(_result(text="", tool_calls=tcs),
                                        response_id="r", created_at=1)
    types = [o["type"] for o in obj["output"]]
    assert types == ["function_call"]
    fc = obj["output"][0]
    assert fc["call_id"] == "call_1" and fc["name"] == "shell"
    assert fc["arguments"] == '{"command":"ls"}'
    assert fc["status"] == "completed" and fc["id"].startswith("fc_")


def test_object_text_and_tool_calls():
    tcs = [{"id": "c1", "type": "function",
            "function": {"name": "f", "arguments": "{}"}}]
    obj = ra.result_to_responses_object(_result(text="ok", tool_calls=tcs),
                                        response_id="r", created_at=1)
    assert [o["type"] for o in obj["output"]] == ["message", "function_call"]


def test_object_length_finish_is_incomplete():
    obj = ra.result_to_responses_object(_result(text="x", finish="length"),
                                        response_id="r", created_at=1)
    assert obj["status"] == "incomplete"


def test_object_usage_carries_cached_tokens():
    res = _result(text="x")
    res["response"]["tokens_cached"] = 4
    obj = ra.result_to_responses_object(res, response_id="r", created_at=1)
    assert obj["usage"]["input_tokens_details"] == {"cached_tokens": 4}


def test_object_omits_usage_when_unknown():
    obj = ra.result_to_responses_object(
        _result(text="x", tin=None, tout=None, ttot=None),
        response_id="r", created_at=1)
    assert "usage" not in obj


def test_object_uses_now_when_created_at_absent():
    obj = ra.result_to_responses_object(_result(text="x"), response_id="r",
                                        now=lambda: 555)
    assert obj["created_at"] == 555


# ---- SSE encoders ------------------------------------------------------

def _parse_sse(frames):
    """[(event_name, data_dict), ...] from a list of 'event:..\\ndata:..\\n\\n'."""
    out = []
    for f in frames:
        lines = [ln for ln in f.split("\n") if ln]
        ev = next(ln[len("event:"):].strip() for ln in lines if ln.startswith("event:"))
        data = next(ln[len("data:"):].strip() for ln in lines if ln.startswith("data:"))
        out.append((ev, json.loads(data)))
    return out


def test_created_event_shape():
    obj = ra.result_to_responses_object(_result(text="x"), response_id="resp_1",
                                        created_at=1)
    (ev, data), = _parse_sse([ra.responses_created_event(obj)])
    assert ev == "response.created"
    assert data["type"] == "response.created"
    assert data["response"]["id"] == "resp_1"
    assert data["response"]["status"] == "in_progress"


def test_failed_event_shape():
    (ev, data), = _parse_sse([ra.responses_failed_event("resp_1", "boom", "server_error")])
    assert ev == "response.failed"
    assert data["response"]["status"] == "failed"
    assert data["response"]["error"]["message"] == "boom"
    assert data["response"]["error"]["code"] == "server_error"


def test_sse_events_text_sequence():
    obj = ra.result_to_responses_object(_result(text="Hello"), response_id="r",
                                        created_at=1, msg_id="msg_1")
    events = _parse_sse(list(ra.responses_sse_events(obj)))
    names = [e for e, _ in events]
    assert names == [
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.completed",
    ]
    delta = dict(events)["response.output_text.delta"]
    assert delta["delta"] == "Hello" and delta["item_id"] == "msg_1"
    completed = dict(events)["response.completed"]
    assert completed["response"]["output"] == obj["output"]
    assert completed["response"]["usage"]["total_tokens"] == 10


def test_sse_events_function_call_sequence():
    tcs = [{"id": "call_1", "type": "function",
            "function": {"name": "shell", "arguments": '{"command":"ls"}'}}]
    obj = ra.result_to_responses_object(_result(text="", tool_calls=tcs),
                                        response_id="r", created_at=1)
    events = _parse_sse(list(ra.responses_sse_events(obj)))
    names = [e for e, _ in events]
    assert names == [
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    done = dict(events)["response.function_call_arguments.done"]
    assert done["arguments"] == '{"command":"ls"}'


def test_sse_events_sequence_numbers_increase():
    obj = ra.result_to_responses_object(_result(text="hi"), response_id="r",
                                        created_at=1)
    seqs = [d["sequence_number"] for _, d in _parse_sse(list(ra.responses_sse_events(obj)))]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
