# Jev probability display precision

TypeSafe displays probabilities and scores to two decimal places. The maintained
[AI SDK TypeSafe adapter](https://github.com/vercel/ai/blob/main/packages/typesafe-ai/src/typesafe-ai-evaluation-model.ts)
declares `probabilityDecimals: 2` and preserves the native numbers. Consequently,
a valid distribution can arrive as `0.33, 0.33, 0.33`, summing to `0.99`.

The router previously required a sum within `1e-6` of one. That can reject a valid
HTTP200 decision response and expose a public502. Live architecture tests found
`bad_response` failures; their discarded response bodies do not establish that
every observed failure had this cause. This patch fixes the independently
reproducible compatibility defect, not generative-provider timeouts.

For versioned Jev model identifiers and the mutable `jev-latest` alias
(including vendor-prefixed forms of both), distributions whose values are displayed at two decimal places are accepted when their clipped
rounding intervals jointly contain a unit mass. Each value still must be finite
and within0..1; keys must match all requested choices; the selected choice must
have maximal displayed probability. Unexplained mass errors and distributions
with other precision retain strict validation. Other model families remain strict.

Raw probabilities, scores, confidence and usage are unchanged. Only the derived
normalized entropy uses the normalized displayed weights. The rule is shared by
public Choice/Score responses, direct Jev selection and automatic-routing
revalidation. Requests, deadlines, model policies and retry behavior do not change.
