# Bedrock tool protocol review — 24 September 2026

PR #123 is a general OpenAI-to-Converse adapter fix. No agent name, trading rule,
wallet, or application-specific branch exists in the adapter.

## Independent model review

Unhardcoded reported `openrouter_market`, served model
`anthropic/claude-opus-5.5`. The first non-streaming request returned 504 without
review text. A streaming review was truncated (`finish_reason: length`); it
identified adjacent user turns after tool results. A second streaming review
completed (`finish_reason: stop`) and recommended changes. Reported costs for
the two successful requests: $0.106366 and $0.105636. The failed request's cost
is unknown; conservative reservations remain in TraderSwarm's ledger.

The reviewer received the complete adapter and diff. Its second review preceded
the final corrections below; this is independent review evidence, not model
approval of the final commit or live AWS validation.

Accepted and implemented:

- Merge consecutive user blocks, with tool results preceding ordinary text.
- Reject results without a matching preceding assistant tool use, in both call paths.
- Return tool calls only after `tool_use` and closure of every tool block.
- Normalize absent arguments only for completed calls; reject malformed,
  non-object, and non-finite JSON.
- Report `stream_interrupted` when text was already emitted before an incomplete
  tool response; return no executable tool calls from the failed batch.

Not adopted: defaulting an absent contentBlockStop index to zero. The API requires
that index; malformed input should fail closed rather than invent a block identity.

Pre-existing, outside this PR: full non-streaming response validation, duplicate
model-generated tool IDs, and preservation of reasoning blocks/signatures.
These are not proved correct by this patch.

## Validation

56 native-provider, streaming and provider tests pass. Eight negative cases fail
against the initial PR and pass after hardening. Tests include parallel results
in different orders, result/text mixtures, an orphan result, incomplete parallel
batches, missing completion, token limits, invalid JSON and emitted text before
failure. Provider calls are mocked; the corrected adapter is not yet deployed or
verified with live Bedrock.

References:

- https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_Scenario_ToolUse_section.html
- https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_MessageStopEvent.html
- https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlockStopEvent.html
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
