# Horizontal scaling

The public router can run as a stateless, autoscaled pool without copying a
wallet identity or Codex refresh credentials. PostgreSQL is the shared admission
authority; one stateful control replica retains the existing RWO volumes and
serves private provider brokers.

## Runtime topology

```text
public Service / ALB
        |
        v
llm-router-api (2..N replicas, no PVC)
  nginx -> auth proxy -> router
             |            |---- normal providers
             |            |---- AntSeed private service
             |            `---- Codex private broker
             `---- PostgreSQL (key digests, rate buckets, ledger/settings)

llm-router-control (exactly 1 replica, existing RWO PVC owner)
  dashboard + synthetic probes + wallet keeper
  AntSeed sidecar (funded identity/SQLite)
  Codex broker (account files)
```

The control replica is deliberately not part of the autoscaled public Service.
Scaling it would duplicate money-moving keeper loops and contend for the RWO
volume. The API pool scales independently.

## Invariants

- Dashboard-created consumer keys are stored as exact SHA-256 digests in
  PostgreSQL. Plaintext is never added to the database. A key created or revoked
  on the control replica takes effect on every API replica in the same database
  transaction as its metadata.
- Rate limiting is a PostgreSQL-backed token bucket per consumer. Adding
  replicas does not multiply `RATE_PER_MIN` or `BURST`.
- Each API pod admits at most `MAX_INFLIGHT_REQUESTS` upstream calls and queues
  at most `MAX_PENDING_REQUESTS` briefly. Excess load receives `503
  router_overloaded` with `Retry-After`; callers can retry another ready replica
  instead of driving the pod into probe timeouts and 502s.
- Codex credentials stay on the control PVC. `codex_broker.py` selects an
  account and performs calls behind `CODEX_BROKER_TOKEN`; responses contain only
  canonical call results and quota headers, never tokens.
- AntSeed API replicas use the private AntSeed Service URL. Only the control
  replica runs the wallet keeper (`RUN_WALLET_KEEPER=1`); every API replica sets
  it to `0`.
- AntSeed seller concurrency is enforced with renewable PostgreSQL leases on
  both JSON and streaming calls. A peer's advertised cap therefore stays
  global as replicas are added, and a crashed pod's lease expires automatically.
- Synthetic probes have one owner. API replicas set
  `DASHBOARD_SYNTHETIC_PROBES_ENABLED=0`, so scaling does not multiply paid
  requests.
- Startup cost backfill has one owner. API replicas set `RUN_COST_BACKFILL=0`
  so a scale-out event cannot fan out the same maintenance query.

## Autoscaling signal

The auth proxy exposes low-cardinality Prometheus metrics on its private port:

- `llm_router_inflight_requests`
- `llm_router_pending_requests`
- `llm_router_capacity_requests`
- request, rejection, store-error, latency, and uptime metrics

Scrape `/metrics` directly with a PodMonitor; do not expose it through the
public nginx listener. KEDA should scale on the sum of active and pending calls,
with a target below the per-pod active cap. For example, with an active cap of
30, a target of 20 gives a new replica time to become Ready before the short
queue fills. Keep at least two replicas and add CPU as a secondary trigger.

Autoscaling does not create upstream quota. Consumer rate limits and provider
fallbacks remain the hard admission controls even when KEDA allows more pods.

## Daily upstream spend ceilings

Set the same `ROUTER_PROVIDER_DAILY_SPEND` JSON on every router replica. Each key
is a stable upstream account identifier. By default it also names a catalog
provider. Use `provider_ids` to specify multiple catalog IDs for the account.
For example:

```json
{"example-provider":{"daily_usd":"50.00","max_attempt_usd":"0.25","provider_ids":["example-provider","example-provider_market"]}}
```

Set `max_attempt_usd` to an upper bound on the price of any allowed attempt,
including its maximum input and output, upstream retries, and all extras the
provider can charge. Use upstream request and account limits to establish that
bound. A zero `daily_usd` pauses dispatch by default. Amounts have up to 12
decimal places. The ceiling applies to routed inference using operator
credentials, including shared tenant routes. Tenant-supplied credentials have
their own upstream accounts.

Each admitted attempt commits its upper bound against the UTC day it starts,
before sending anything upstream. A confirmed provider-reported cost on a
successful response replaces the bound. Unmetered failures, cancellation,
missing usage, and dropped settlement writes keep the full charge. Requests
that cannot reserve take the next ranked candidate. The synchronous embedding
requires the async driver to use capped providers.

On the private router admin listener,
`GET /x/providers/{upstream_account}/daily-spend` returns today's effective ceiling,
committed spend (including in-flight charges), and reset time. Use `PUT` on the
same path with `{"override_usd":"75.00"}` to replace today's ceiling across
replicas or `{"override_usd":null}` to restore its configured value. The
override preserves all charges and expires at 00:00 UTC.

## Zero-downtime migration

1. Deploy the compatible application release to the existing single replica.
   It creates the new PostgreSQL tables idempotently and backfills configured
   key hashes.
2. Create `CODEX_BROKER_TOKEN` in the platform secret manager through IaC. Add
   the Codex broker and private AntSeed/Codex Service to the existing PVC-owning
   deployment.
3. Deploy the stateless API Deployment, PodMonitor, autoscaler, and PDB behind a
   shadow Service. Keep public traffic on the original Service while health,
   consumer-key auth, streaming, Codex, and AntSeed are exercised.
4. Route the dashboard to a control-only Service, then switch the public Service
   selector to the already-Ready API pods. No PVC is detached or handed over.
5. Observe overload rejections, p95 latency, replica count, database health, and
   provider errors before raising the maximum replica count.

Rollback is a Service-selector change back to the original control pod. The old
pod and PVC stay running during the migration, and the schema additions are
backward-compatible, so rollback does not require restoring storage.

All cluster resources, selectors, secrets, and scaling thresholds belong in the
GitOps/Terraform repositories. Do not patch the live Deployment or create the
broker token manually.

## Shared-scope request cost

The ingress reserves the published per-request ceiling in PostgreSQL before
forwarding a shared-scope request. The router deducts the configured worst-case
cost of each physical provider attempt from that request's remaining allowance
before dispatch. A retry or fallback must fit the same remaining allowance.
Unknown attempt costs retain their full bound. The trusted router returns the
aggregate charge for background settlement. A missing result keeps the ingress
precharge.

Set `SHARED_COST_ATTEMPT_LIMITS` on the router to a JSON array such as:

```json
[
  {
    "provider": "direct",
    "base_url": "https://inference.example/v1",
    "model": "text-model-version",
    "max_input_bytes": 65536,
    "max_output_tokens": 1024,
    "max_cost_usd": "0.05"
  },
  {
    "provider": "openrouter",
    "protocol": "decisions",
    "base_url": "https://openrouter.ai/api/alpha/decisions",
    "model": "typesafe/jev-1.13",
    "max_input_bytes": 16000,
    "max_cost_usd": "0.01"
  }
]
```

Each chat entry attests an upper bound for one POST to the exact HTTPS endpoint
and wire model using the operator's shared provider credentials. The limit covers
every accepted serialized JSON body up to `max_input_bytes` and all billable
output up to the enforced `max_tokens` limit. Include protocol overhead,
reasoning tokens if applicable, and any other charges in the bound. Configure
only endpoints that enforce the output limit, return trustworthy usage costs
when present, and charge within the bound even on failed or canceled calls.
Update the bound before changing the endpoint's charging or token semantics.
Keep configuration consistent on all router replicas.

Marketplace offers must match the configured provider, HTTPS seller endpoint
and wire model. Peer offers also require an exact `peer_id` match and a gateway
that honors that pin and enforces the attested bound. Peer attempts retain the
full bound at settlement because the seller can control the reported usage.
Discovery prices are used by ranking. They never authorize dispatch.

For an automatic route, a `decisions` entry attests the total charge for one
selection POST, including all output and overhead. Its `base_url` is the full
decision endpoint. The selector still requires the tenant's decision credential.
If the decision cannot be dispatched, the ordinary general-policy fallback runs
under the same guard. Selection and inference consume the same request allowance.
Reported costs can reduce settlement. The sum of the dispatch bounds must still
fit the original request ceiling.

Shared scopes support buffered text chat. Streaming, multimodal input, tools and
flow execution remain outside the supported request shape. The route's
first-token deadline applies to completion of the buffered provider call.
The default configuration allows no provider attempts for shared scopes.

During a UTC month rollover, a scope waits for the previous month's live
commitments to settle or expire. Admission uses PostgreSQL's UTC clock and a
transaction lock on the stable scope identity. A lost settlement retains the
precharge in the original month. The trusted bridge expires a request by the
reservation deadline and limits execution to 120 seconds.

Enable shared scopes only after ingress and router replicas run the version
with guarded dispatch. An upgraded ingress uses a dedicated internal endpoint
so a router that predates the guard returns an error.
