# Decision models

Decision models answer typed questions about a state instead of generating chat
text. The catalog labels them **Decision models**, exposes `type: "decision"`,
and supports `GET /v1/models?type=decisions`. The dashboard Catalog has a matching
category filter. Existing chat requests select only chat candidates, even when
a caller pins a decision model or supplies a policy without `meets_req`.

## API and policies

Use an existing **normal-router consumer key** with `POST /v1/decisions`.
`POST /v1/systemone` is an equivalent endpoint for System One clients.
These endpoints do not change Cloud's published-route contract; Cloud tenant
keys continue to use their existing supported endpoints.

```json
{
  "model": "family:jev-1.13",
  "state": {"ticket": "I was charged twice."},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which department should handle this ticket?",
      "criteria": {"billing": "Payments and refunds", "technical": "Software issues"}
    }
  }
}
```

`choice`, `score`, and `noul` questions may be batched. Responses preserve
`model`, `answers` and `usage`, adding `x_router` with the selected provider,
peer, prices, cost and attempt trace. Responses are not streamed. Payloads are
bounded to 32 KB, 32 questions, and 32 criteria per question. Unsupported chat
fields, invalid answers, oversized replies and redirects fail closed.

Set `policy_ir` to the contents of `policies/jev-value-v1.json` for the qualified
Jev policy: exact `jev-1.13` family, known prices <= $0.10/M input and output,
success-rate floor, healthy/observed-fast routes first, 70% input-cost and 30%
latency utility, at most four Jev attempts. The mutable `jev-latest` family is
kept separate: it is not evidence of a second backend or a pinned version.
Unknown success rate retains the engine's optimistic default; unknown latency
is ranked behind observed-fast routes. These are selection preferences, not
proof of model quality or guaranteed savings.

Without a policy, the endpoint selects up to four decision candidates by input
price, preferring closed breakers, with a 2.5-second per-attempt timeout.
Use `model` (`family:…`, `pin:provider/family`, or a bare model name) to constrain
that selection. Omit `model` when the policy itself owns all model restrictions.
The server's configured overall request deadline bounds
the cascade. A supplied `first_token_timeout_ms` bounds the complete decision
response, since decision models do not emit tokens incrementally.

Preview a decision policy without spending:

```json
{"protocol": "decisions", "policy_ir": ["policy", ["meets_req"], ["zero"], ["argmax"], ["id"], ["always", {"action":"next_candidate"}]]}
```

Send this to `POST /x/rank`. Omitting `protocol` previews chat candidates.
Authentication, revocation, consumer route restrictions, rate limits, request
capacity and budget admission remain in the existing ingress. Input/output
token usage and provider-reported cost are recorded by that same ingress.
As for existing router routes, `cost_usd` describes the winning attempt;
unreported charges from failed upstream attempts are not known.

## Discovery and transport

- OpenRouter: query `/api/v1/models?output_modalities=decisions` as well as the
  normal model catalog; call `/api/alpha/decisions` with the configured provider
  credential. A failure of decision discovery does not discard chat discovery.
- AntSeed: CLI `0.1.161`, router-local `0.1.46`, node `0.2.121`; query the buyer's
  `/v1/models?type=decisions` and require advertised `typesafe-systemone` support
  on each peer offer. Send to `/v1/systemone`, preserving the exact service ID
  and `x-antseed-pin-peer`. The router owns peer selection and fallback, so it
  deliberately uses the buyer's supported pin header instead of the standalone
  skill's recommendation to let the buyer choose. Existing funds, reputation,
  allow/deny, cooldown and concurrency checks remain in force.

AntSeed protocol announcements are persisted in the additive nullable
`peer_offers.protocols` column. Missing metadata in a later announcement keeps
the last known protocol. Unknown future decision services cannot enter chat
while the buyer's typed catalog is unavailable. Decision catalog snapshots
expire using the existing source staleness bound. Known Jev names are also
quarantined when old sidecars have no protocol metadata.

## Merge and deployment

1. Merge the engine protocol PR.
2. Ensure this host PR's core submodule points to the merged engine revision;
   merge this host PR after its Python, Lua and image checks pass.
3. **Deployment remains a separate manual step.** Start the updated router
   first so its additive schema initialization completes, then update the
   AntSeed sidecar. An old sidecar continues serving chat; its unsupported
   decision catalog leaves AntSeed decision routes unavailable.
4. Qualify the deployed catalog, authenticated decision inference and actual
   provider failover. Do not infer live AntSeed availability from the existence
   of the upstream plugin. No production deployment is part of these PRs.

The current legacy control deployment has one replica and uses Recreate; its
public ingress still points there. This change does not itself provide a
zero-downtime deployment path.

Upstream contracts:
[AntSeed decisions](https://github.com/AntSeed/antseed/blob/main/skills/antseed-decisions/SKILL.md),
[AntSeed buyer transport](https://github.com/AntSeed/antseed/blob/main/apps/cli/src/proxy/buyer-proxy.ts),
[OpenRouter decision catalog](https://openrouter.ai/api/v1/models?output_modalities=decisions),
[TypeSafe API](https://docs.typesafe.ai/api).

## Qualification

The new local ingress and router completed three real OpenRouter calls using
the existing consumer/provider credentials kept in memory. All three selected
the expected support department: 628, 432 and 426 ms, with total reported cost
$0.000041034. The local ingress admitted the consumer key using an isolated
configuration; production was not changed. See
[`jev-new-local-router-20260920.json`](qualification/jev-new-local-router-20260920.json).

AntSeed→OpenRouter fallback is covered through the real Lua engine and HTTP
adapters with simulated upstreams, including peer pins and usage recording.
Live AntSeed serving and actual cross-provider failover must still be qualified
after the newer sidecar is deployed. The older production-router baseline is
retained separately in `qualification/JEV-POLICY-20260920.md`.
