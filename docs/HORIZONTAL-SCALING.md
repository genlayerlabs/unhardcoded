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

## Offline load check

With Python 3.10+ and Docker Compose v2, prepare the images once. Image
preparation can use the network.

```sh
git submodule update --init
docker build -t unhardcoded:local .
docker pull postgres:16
docker pull nginx:alpine
```

Run the load check in one command once those images are cached.

```sh
python3 scripts/router-load-check.py unhardcoded:local > load.json
```

To check a change, build or tag a local image from `main` as
`unhardcoded:main` and build the working tree as `unhardcoded:local`. Run both
against identical fixtures and settings.

```sh
python3 scripts/router-load-check.py unhardcoded:main \
  --compare unhardcoded:local --margin-pct 30 > comparison.json
```

Use `--compare unhardcoded:main` with the same baseline tag to measure
run-to-run variation on your machine. The manual **Router load check** workflow
runs a standalone check and two same-image comparisons, checks the margin in
both directions, and uploads the JSON.

Each run starts two replicas behind round-robin nginx on an internal Docker
network with no published ports or image pulls. The harness uses the baseline
image's Python dependencies for both runs. PostgreSQL and the fake provider
are fresh for each image, and the script removes the containers and data at
the end. The defaults are ten seconds of warmup and 45 seconds of measurement
per image, up to eight concurrent clients, 80 requests per second, and 40 ms
before the fake provider replies. Adjust these settings with `--duration`,
`--concurrency`, `--rate`, and `--latency-ms`. The driver evenly spaces request
starts and skips missed slots if requests take too long. The workload repeats
ten requests at a time with four valid buffered calls, four valid streaming
calls, one call using an exhausted rate-limited key, and one with an invalid
key. The limited key refills once a minute and can occasionally succeed.

The summary table goes to stderr and JSON goes to stdout. Overhead is
successful valid-key request time through nginx, less the configured fake
provider delay. It includes the local proxy and fixture processing. Stream
latency measures time until the first SSE chunk carrying a content token.
Memory and connections are sampled peaks. Docker stats samples take roughly
a second or longer, so short spikes can fall between samples. The connection
report includes both total open and active router sessions. The PostgreSQL
statement count uses the change in `pg_stat_statements` for the router role,
including transaction control and background writes observed through a
one-second drain period, divided by all completed requests.

Compare mode exits nonzero if p99 overhead, queries per request, or either
replica's peak memory increases by more than the margin (30% by default).
Both modes fail if valid calls fail or the expected rate-limit and
authentication refusals are absent.
