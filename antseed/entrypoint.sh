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
#   * validate-before-write      -> never clobber market.json with the CLI's
#     non-JSON "No peers found" line.
#   * --top high                 -> the CLI default is 20; we want the whole
#     peer set.
set -eu

PORT_PROXY="${ANTSEED_PROXY_PORT:-8377}"      # buyer proxy (binds 127.0.0.1)
PORT_PUBLIC="${ANTSEED_PUBLIC_PORT:-8378}"    # socat, on the container network
MARKET_DIR="${ANTSEED_MARKET_DIR:-/market}"
TOP="${ANTSEED_BROWSE_TOP:-500}"
INTERVAL="${ANTSEED_BROWSE_INTERVAL:-60}"     # browse cadence (s); 60s feeds the
                                              # sliding window in merge-market.js
MAXIN="${ANTSEED_MAX_INPUT:-1000}"            # buyer spend rail; Σ_pol policy is
MAXOUT="${ANTSEED_MAX_OUTPUT:-1000}"          # the real per-call price ceiling
CLI_TIMEOUT="${ANTSEED_CLI_TIMEOUT:-45}"      # hard cap per browse/status CLI call.
                                              # WITHOUT it a single hung `antseed`
                                              # invocation (e.g. a concurrent buyer
                                              # command grabbing the sqlite store)
                                              # blocks the writer loop FOREVER — it
                                              # has no self-restart — so market.json
                                              # and status-antseed.json silently
                                              # freeze and the catalog/wallet go stale.
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

atomic_write() {  # <validator-cmd> <dest> ; reads producer stdout on fd via $1
    dest="$1"; tmp="${dest}.$$.tmp"
    if cat > "$tmp" && [ -s "$tmp" ]; then
        mv "$tmp" "$dest"
    else
        rm -f "$tmp"
        return 1
    fi
}

write_market() {
    raw="$MARKET_DIR/.market.raw.$$"
    timeout -k 5 "$CLI_TIMEOUT" antseed network browse --services --top "$TOP" --json > "$raw" 2>/dev/null || true
    # merge this browse into the rolling window (and validate: a non-dump exits
    # non-zero -> keep the last good market.json)
    if node "$LIB/merge-market.js" "$MARKET_DIR/market.json" < "$raw" | atomic_write "$MARKET_DIR/market.json"; then
        :
    else
        # record what we got for debugging; keep the last good window
        cp "$raw" "$MARKET_DIR/market.err" 2>/dev/null || true
    fi
    rm -f "$raw"
}

write_status() {
    raw="$MARKET_DIR/.status.raw.$$"
    timeout -k 5 "$CLI_TIMEOUT" antseed buyer status --json > "$raw" 2>/dev/null || true
    node -e 'const fs=require("fs");let d;try{d=JSON.parse(fs.readFileSync(0,"utf8"))}catch(e){process.exit(2)}if(d===null||typeof d!=="object")process.exit(3);d.fetched_at_ms=Date.now();process.stdout.write(JSON.stringify(d))' \
        < "$raw" | atomic_write "$MARKET_DIR/status-antseed.json" || true
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
