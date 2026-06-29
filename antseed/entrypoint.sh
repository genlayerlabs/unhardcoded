#!/bin/sh
# AntSeed buyer sidecar — vendored CLI, robust market writer, network-reachable
# proxy. Replaces the old inline `npm install + shell loop` in compose.yml.
#
# Why each piece exists (all four were live failure modes):
#   * vendored CLI (Dockerfile)  -> no runtime `npm install` (registry outage
#     killed cold starts; version drift rotted the DHT bootstrap nodes).
#   * socat forwarder            -> the buyer proxy binds 127.0.0.1 only (no
#     --host flag), so on its own Docker network the router can't reach it;
#     socat exposes it on the container network. This lets us DROP
#     `network_mode: service:router`, whose orphaned netns (router recreated,
#     antseed not) silently zeroed discovery on every deploy.
#   * validate-before-write      -> the writers reject the CLI's non-JSON
#     "No peers found" line (exit non-zero) instead of wiping the store row.
#   * --top high                 -> the CLI default is 20; we want the whole
#     peer set.
#
# The market book and buyer status are written to the host store (Postgres)
# by write-market.js / write-status.js — no market.json / status-*.json files.
# $MARKET_DIR is now just container-local scratch for the raw CLI dumps.
set -eu

PORT_PROXY="${ANTSEED_PROXY_PORT:-8377}"      # buyer proxy (binds 127.0.0.1)
PORT_PUBLIC="${ANTSEED_PUBLIC_PORT:-8378}"    # socat, on the container network
MARKET_DIR="${ANTSEED_MARKET_DIR:-/market}"
TOP="${ANTSEED_BROWSE_TOP:-500}"
INTERVAL="${ANTSEED_BROWSE_INTERVAL:-60}"     # browse cadence (s); 60s feeds the
                                              # observed_at sliding window in peer_offers
MAXIN="${ANTSEED_MAX_INPUT:-1000}"            # buyer spend rail; Σ_pol policy is
MAXOUT="${ANTSEED_MAX_OUTPUT:-1000}"          # the real per-call price ceiling
CLI_TIMEOUT="${ANTSEED_CLI_TIMEOUT:-45}"      # hard cap per browse/status CLI call.
                                              # WITHOUT it a single hung `antseed`
                                              # invocation (e.g. a concurrent buyer
                                              # command grabbing the sqlite store)
                                              # blocks the writer loop FOREVER — it
                                              # has no self-restart — so the
                                              # peer_offers / buyer_status rows
                                              # silently freeze and the catalog/wallet go stale.
LIB=/usr/local/lib/antseed

mkdir -p "$MARKET_DIR"

# Buyer proxy in browse mode (no --peer: the host pins per request). A funded
# wallet is needed to transact; discovery/pricing work unfunded.
antseed buyer start -p "$PORT_PROXY" \
    --max-input-usd-per-million "$MAXIN" \
    --max-output-usd-per-million "$MAXOUT" &
PROXY_PID=$!

# Expose the localhost-only proxy on the container network for the router.
socat "TCP-LISTEN:${PORT_PUBLIC},fork,reuseaddr" "TCP:127.0.0.1:${PORT_PROXY}" &

# Wallet control server (deposit/withdraw/status from the dashboard, no kubectl).
# Self-disables when ANTSEED_CONTROL_TOKEN is unset. See antseed/control.js.
node "$LIB/control.js" &

write_market() {
    raw="$MARKET_DIR/.market.raw.$$"
    timeout -k 5 "$CLI_TIMEOUT" antseed network browse --services --top "$TOP" --json > "$raw" 2>/dev/null || true
    # upsert this browse into the host store's peer_offers (and validate: a
    # non-dump / DB error exits non-zero -> keep the last good window, no write)
    if ! node "$LIB/write-market.js" < "$raw"; then
        # record what we got for debugging; keep the last good window
        cp "$raw" "$MARKET_DIR/market.err" 2>/dev/null || true
    fi
    rm -f "$raw"
}

write_status() {
    raw="$MARKET_DIR/.status.raw.$$"
    timeout -k 5 "$CLI_TIMEOUT" antseed buyer status --json > "$raw" 2>/dev/null || true
    # upsert the buyer status into the host store's buyer_status (validate + DB
    # error -> exit non-zero, keep the last good row)
    node "$LIB/write-status.js" < "$raw" || true
    rm -f "$raw"
}

(
    sleep 10   # let the buyer join the DHT before the first browse
    while true; do
        write_market
        write_status
        sleep "$INTERVAL"
    done
) &

wait "$PROXY_PID"
