# Providers

Operational notes for the provider catalog in `../config.live.lua`. The router
itself is auth-agnostic: each provider declares how it authenticates and the
**host** resolves it (see `_resolve_auth_headers` in
`hosts/python/llm_router_host.py`). Three auth kinds:

| `auth`                                   | Header sent                | Used by              |
|------------------------------------------|----------------------------|----------------------|
| `auth_env = "X"` (or `{kind="bearer"}`)  | `Authorization: Bearer $X` | heurist, io_net, openrouter |
| `{ kind = "none" }`                      | *(none)*                   | antseed              |
| `{ kind = "oauth", provider = "codex" }` | `Authorization: Bearer <refreshed token>` | openai (codex) |

## Bearer providers (heurist, io.net, openrouter)

Standard OpenAI-compatible gateways. Put the key in the shim's process
environment under the provider's `auth_env` name. Clients hitting the shim do
**not** carry these keys.

## AntSeed (local node, no auth)

> ⚠️ **Spends real money on-chain.** The AntSeed provider pays peers in **real
> USDC on Base mainnet** from a funded hot wallet you control. It is **opt-in**:
> it ships behind the `antseed` Compose profile (off by default) and the wallet
> control server self-disables unless you set `ANTSEED_CONTROL_TOKEN`. Treat
> `ANTSEED_IDENTITY_HEX` as a private key — never commit it. Only enable this if
> you understand you are funding and spending a live mainnet wallet.

AntSeed is a decentralized meta-router: a local node speaks OpenAI Chat
Completions with **no Authorization header**, reachable on the container
network at `http://antseed:8378/v1` (the daemon binds `127.0.0.1:8377`; socat
exposes it — see `antseed/entrypoint.sh`).

**Quality is per-model, not per-provider.** AntSeed serves the same model
families as everyone else (`minimax-m2.7`, `claude-*`, `qwen3-235b-a22b`, …),
so each offer inherits that family's OpenRouter benchmark from `model_meta.lua`
exactly like an OpenRouter offer of the same family. AntSeed is therefore a
`marketplace` tier that **competes head-to-head on benchmark + price** (and,
being cheaper, often wins) — *not* a quality-blind fallback. The only signal
OpenRouter's per-model benchmark can't give you about an AntSeed peer is its
latency/reliability for that model; that is learned from your own call history
(the EMA). Services that don't map to a curated family are still exposed (under
their raw wire name) and score on price + learned latency alone.

The host pins the policy-selected peer per request via `x-antseed-pin-peer`
(the browse-mode buyer disables auto-selection), keeping peer choice inside
Σ_pol rather than an opaque buyer-side router.

### Local dev wallet (testing)

For local testing use a **dedicated dev wallet**, never your production key.
`./scripts/gen-dev-wallet.sh` prints a fresh `ANTSEED_IDENTITY_HEX` +
`ANTSEED_CONTROL_TOKEN` to paste into `.env`; bring the sidecar up
(`docker compose --profile antseed up -d`), read the derived address with
`docker compose exec antseed antseed buyer balance --json`, fund it with a little
USDC + ETH (gas) on Base, then **Deposit** into escrow from the dashboard Catalog
(wallet cell). Keep dev and prod wallet secrets separate. See `.env.example`.

> Note: the AntSeed deposits contract **locks** deposited funds — an immediate
> `withdraw` after a `deposit` reverts. Funds are safe in escrow and become
> withdrawable later, or are spent as the buyer routes paid calls.

### Running the node (vendored sidecar)

Built from `Dockerfile.antseed` (pinned `@antseed/cli`, `socat`) and run by
`antseed/entrypoint.sh` under the `antseed` compose profile — **not** a runtime
`npm install`, and **not** `network_mode: service:router` (whose orphaned netns
silently zeroed discovery on every router recreate). The entrypoint runs the
buyer proxy in browse mode, the socat forwarder, and a 300 s loop that validates
each `network browse --json` dump before writing it (the CLI prints a non-JSON
"No peers found" line even with `--json`) into `/market` for `sources/antseed.py`.

```bash
docker compose --profile antseed up -d --build antseed
```

- **Identity + wallet:** the buyer needs a secp256k1 identity and a **funded
  wallet** (USDC + a little ETH for gas on Base mainnet) to pay peers; staking
  is only for *selling*. Set `ANTSEED_IDENTITY_HEX` (compose env) so the funded
  wallet is durable — otherwise the CLI generates an ephemeral key in the
  `antseed-data` volume and losing the volume loses the funded address.
- **Wallet vs deposits:** the dashboard shows `depositsAvailable` — USDC moved
  into the AntSeed **deposits contract (escrow)**, which is what the buyer spends
  — NOT raw wallet USDC. Funding the wallet is not enough; you must `deposit`
  into escrow. (`reserved` is escrow locked in active payment channels — in use,
  not lost; it returns to available as channels settle.) The catalog wallet cell
  also shows the **raw wallet balance** — USDC sitting in the wallet, plus ETH for
  gas — read on-chain from Base, so you can tell at a glance whether you have funds
  to deposit and gas to move them. The RPC defaults to a public Base endpoint;
  override with `ANTSEED_WALLET_RPC_URL`, or set it to `off` to show escrow only.
- **Self-service (no kubectl):** set `ANTSEED_CONTROL_TOKEN` (shared by the
  `router` and `antseed` services) to enable the sidecar wallet control server;
  the catalog then offers **Deposit / Withdraw / Refresh** buttons. Unset → those
  endpoints return 503 and you fund via `antseed buyer deposit <amt>` over
  `kubectl exec`. In k8s the router + antseed share a pod, so set
  `ANTSEED_CONTROL_URL=http://127.0.0.1:8379`; the control port is pod-local
  (no Service/Ingress).
- **Per replica:** if you scale the shim horizontally, each replica needs a
  reachable AntSeed node — a sidecar per replica (own funded identity) or one
  shared instance. Decide this when you add replicas.
- **Funding autonomy (`wallet_keeper.py`) — off by default.** Escrow *ratchets*:
  every channel open reserves ~1 USDC, and a channel that never settles never
  gives it back, so a buyer whose `depositsAvailable` falls below one reserve
  answers 402 `insufficient_deposits` to *every* call until a human deposits or
  reclaims. Two things now cover that:
  - a **tourniquet** in `sources/antseed.py`: below
    `antseed.min_available_usdc` (1.1) the source stops offering AntSeed routes
    at all, rather than spending callers' requests on certain failures. It fails
    OPEN — an unreadable buyer status is a read error, not an empty escrow — and
    it does *not* bound the row's age, deliberately: the belt that does is the
    envelope. A status row older than 15 minutes (the sidecar rewrites it every
    60s) makes `balances()` publish **0** credits, which closes the envelope's
    `credits >= 1.0` clause and takes AntSeed out of ranking anyway.
  - the **wallet keeper**, a loop in the router (the buyer identity + sqlite are
    on an RWO PVC bound to this pod, and the control server is pod-local) that
    reclaims stuck channels and then tops the escrow up. It **reclaims before it
    tops up** — refilling a ratchet you have not unwound just feeds the leak —
    and only force-closes channels when the provider is provably not using them:
    either the offer tourniquet is suppressing every route right now (so there
    *cannot* be traffic), or there was traffic and none of it succeeded for an
    hour. Never when it is merely idle. Reclaim also fires when the escrow has
    been topped up but the reserve dwarfs it, so a deposit cannot lock the
    recovery path out permanently.

  It **ships dark**: `antseed.keeper_enabled` defaults to `0`, and arming it is a
  deliberate operator act (Config tab or `ANTSEED_KEEPER_ENABLED=1`). Its limits
  are the `antseed.topup_*` / `antseed.reclaim_min_usdc` knobs, cross-validated
  on load. It will **never** run `buyer withdraw` (escrow → wallet): that is the
  exfiltration path if `ANTSEED_CONTROL_TOKEN` leaks and stays human-only. Every
  action is written to `wallet_ops` as an INTENT *before* it fires — that table
  is the audit trail, the daily-cap ledger and the `/x/runtime` feed at once.

  Two breakers hard-halt top-ups until an operator clears the flag:

  - two deposits in a row that fail to raise the **escrow** (`depositsAvailable`
    *plus* `depositsReserved` — a deposit immediately reserved by an opening
    channel did arrive, it just moved one column right); and
  - three in a row that could not be *completed*. A deposit whose HTTP call
    timed out, reset, or came back `502`/`504` is recorded `unknown` and
    **counts against the daily cap and the cooldown** — it may have put a
    transaction on Base mainnet. A response that proves **no transaction could
    have been broadcast** is recorded `failed`: it consumes no cap and no
    cooldown, but it is **not free** — it still counts toward this breaker and
    still backs the retry interval off exponentially, or a misconfigured
    endpoint would be retried every 60s forever.

    That proof is the sidecar's to give, never the keeper's: the keeper only
    ever sees `(stderr || stdout)[:600]` of the run, one stream and truncated,
    while the buyer CLI prints its transaction hash on the other one. So
    `attempted: false` comes either from a request that never reached the CLI (a
    `400` from the amount validator, a `401` after a token rotation, a `429`
    from the sidecar's queue gate) or from `antseed/broadcast.js` recognising a
    CLI failure as provably pre-RPC — a process that never spawned, a module
    graph that would not load.

    It is a **narrow** set on purpose. `@antseed/cli` discards the transaction
    hash when a deposit fails *after* broadcasting, so a receipt poll that 403s
    is byte-identical to a 403 before signing; every failure from the moment the
    CLI's on-chain step starts therefore stays `unknown`. Deciding otherwise
    would move real USDC with the ledger recording nothing. Resolving that class
    needs evidence from outside the CLI's stdio (the wallet nonce around the
    run, or the escrow delta a status cycle later) and is not implemented.

    Because every `unknown` consumes the cap, three strikes are not always
    reachable — at the shipped knobs (cap 10, amount 5) only two deposits fit in
    a 24h window. So the breaker also halts on **two** strikes once the cap can
    no longer admit another attempt: at that point the keeper is not done
    spending for the day, it is wedged, and the halt costs nothing it could
    still have done.

  Both halts are cleared by an operator via `POST /x/wallet/clear-halt`
  (`{"kind": "topup"}` or `{"kind": "reclaim"}`); `GET /x/wallet/halts` shows
  what is set and why. Nothing self-clears — a breaker that re-arms itself is
  not a breaker — but a breaker with no reset is a trap, so there is a way back.

  Reclaim has its own cooldown (one challenge window) and its own breaker, and
  names the exact channels the sidecar may act on, so the per-cycle transaction
  cap and the dust filter bind on-chain rather than only in the log line. The
  sidecar echoes back what it acted on and the keeper **verifies** it: router
  and sidecar are separate images, and one that predates the channel-id list
  would silently act on every eligible channel, so a mismatch halts reclaim. A
  top-up halt does **not** stop reclaim: it means "stop putting money in", and
  reclaim moves money the other way.

  Dashboard-initiated deposits and withdrawals are ledgered too, under
  `topup_manual` / `withdraw_manual` — same wallet, same CLI, same need for an
  audit trail — but they do not consume the keeper's daily cap: two actors, two
  budgets, one wallet.

  > **`ANTSEED_WALLET_RPC_URL=off` disables top-ups.** The hot-wallet floor is
  > checked against a public Base RPC read, and an *absent* read vetoes just as a
  > low one does — otherwise taking the endpoint down removes the floor. With the
  > read off there is no floor to check, so every top-up is declined and reported
  > as `wallet_unreadable` on `/x/runtime`. Reclaim is unaffected.

The `model` field we send is the offer's wire id (the peer's service name),
forwarded verbatim; AntSeed translates protocols and serves it.

## OpenAI via ChatGPT subscription (Codex proxy)

See [`docs/OPENAI-CODEX.md`](./OPENAI-CODEX.md). **Unofficial and ToS-risky** —
the Apps SDK OAuth does not grant inference on a subscription; only the Codex
login + local proxy path works, and OpenAI may close it (as Anthropic and Google
closed their equivalents in 2026). Use a normal OpenAI API key
(`auth_env = "OPENAI_API_KEY"`, `api_kind = "openai_compatible"`) if you want a
supported path instead.
