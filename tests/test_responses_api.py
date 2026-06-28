"""Unit tests for responses_api.py — the inbound Responses-API translation
(mirror of codex_backend.py). Pure, no host."""
from __future__ import annotations

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


def test_tools_to_chat_passes_through_unknown_tool_types():
    flat = [{"type": "local_shell"}, {"type": "web_search"}]
    assert ra.tools_to_chat(flat) == flat


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
