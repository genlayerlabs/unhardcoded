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
