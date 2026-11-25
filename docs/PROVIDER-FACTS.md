# Provider facts

An operator service can publish named Boolean observations for a catalog provider
ID. Each fact applies to all candidates with that provider ID. The router treats
the optional JSON detail as opaque.

## Publish an observation

Set `CONTROL_PLANE_INTERNAL_SECRET` on the ingress. Send it via the
`x-internal-secret` header on both `PUT` and `GET` to
`/internal/provider-facts/{provider_id}/{name}`. For example, to record that
`example-provider` is hosted in the EU, send this body with `PUT /internal/provider-facts/example-provider/eu_hosted`:

```json
{"value": true, "observed_at_ms": 1700000000000, "detail": {"region": "eu"}}
```

Use the actual observation time as Unix epoch milliseconds. A future timestamp
is rejected. An older write returns HTTP 409. A write at the stored timestamp
replaces the stored observation. Omitted detail is returned as `null`. A
`GET` returns the latest observation even after it expires, or HTTP 404 if
none exists. Fact names are case-sensitive and use 1–64 ASCII letters, digits,
or underscores, with dots and hyphens also allowed after the first character.

## Require a fact in a policy

Add `["is", "fact:eu_hosted"]` to the filter in a raw `policy_ir` term:

```json
["policy",
  ["and", ["meets_req"], ["not", ["is", "disabled"]], ["is", "fact:eu_hosted"]],
  ["lit", 0], ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]
```

Set `ROUTER_PROVIDER_FACT_MAX_AGE_S` on the router to a positive integer number
of seconds (default 300). Missing, false, stale, or unavailable observations
evaluate to false. If no candidate passes the filter, the router reports
`no_candidates`. Retries use only candidates admitted at ranking time. Policies
that do not refer to facts perform no fact lookup.

Ranking reads a local snapshot. Until its first successful refresh, or once the
snapshot exceeds the same maximum age, all facts evaluate to false. A failed
refresh retains the previous snapshot until it expires.

The router starts refreshes at most `ROUTER_PROVIDER_FACT_REFRESH_S` seconds
apart under normal operation (default 5, with up to 20% downward jitter). A
successful PUT updates the receiving router's snapshot before responding. Other
replicas see the observation on their next successful refresh, normally within
one interval plus the query time. Configure the same internal secret on ingress
and its router so the ingress can forward writes.

Each refresh loads up to 10,000 true facts in fact-name and provider-ID order and
logs when it reaches the limit. Facts outside the snapshot evaluate to false.
A successful local write can temporarily displace a cached fact at capacity.
