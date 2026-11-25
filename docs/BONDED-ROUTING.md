# Bonded-only routing

Use this Σ_pol term as a route's policy or as the `policy_ir` of a
policy-routed request. It keeps the standard eligibility checks and admits
only candidates whose provider has a qualifying ESLA agreement.

```json
["policy",
  ["and", ["meets_req"], ["not", ["is", "disabled"]], ["is", "bonded"]],
  ["lit", 0],
  ["argmax"],
  ["id"],
  ["always", {"action": "next_candidate"}]]
```

Fallback stays within the filtered candidates. If none qualify, the router
returns its usual `no_candidates` error. For direct requests, select a
policy-driven route rather than the engine's explicit `pin:` override.

The router process reads these environment variables at startup.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROUTER_MIN_PROVIDER_BOND` | `1000000000000000000` | Minimum bond in GEN base units (1 GEN), inclusive |
| `ROUTER_PROVIDER_BOND_MAX_AGE_S` | `300` | Maximum age of a check in seconds, exclusive |

The router requires an `active` observation with sufficient bond and a check
timestamp at or before the ranking time. Missing rows, future or stale
timestamps, invalid settings and an unavailable registry provide no bonded
candidates. The registry is keyed by the router's exact provider ID. All model
candidates under that ID share the fact. Policies that never evaluate
`bonded` do not query the registry.

## Recording observations

The ingress serves `PUT /internal/provider-agreements/{provider_id}` and
`GET /internal/provider-agreements/{provider_id}`. Both use the
`x-internal-secret` header set to the configured
`CONTROL_PLANE_INTERNAL_SECRET`. Without a configured secret the endpoints
return 404. An incorrect or missing header returns 403.

Example PUT body:

```json
{
  "agreement_address": "0x1234567890123456789012345678901234567890",
  "network": "example-network",
  "bond_amount": "1000000000000000000",
  "status": "active",
  "checked_at_ms": 1760000000000
}
```

Use the Unix time in milliseconds at which the agreement was checked.
`bond_amount` is a decimal string so the entire unsigned 256-bit range
round-trips without precision loss. Status is one of `active`, `exiting`,
`defaulted` and `none`. For `none` the address and network may be null or
omitted. For other statuses both are required. The writer is responsible for
checking the correct network and agreement. The router does not query the
chain.

PUT returns the stored observation with its provider ID. An older observation
returns 409. An observation with the same timestamp may replace the row.
GET returns the row, or 404 if the provider has no observation.
