// Marketplace book writer (replaces merge-market.js). Each browse sees only the
// peers DHT-reachable from this node *right now* (~68); the network has more
// (~83) that flit in and out of our view. We keep a SLIDING window so a peer
// silent for a few browses still routes: instead of a hand-rolled union of the
// previous file, every peer/service we see is UPSERT-ed into the host store's
// `peer_offers` table stamped `observed_at = now`, and sources/antseed.py reads
// only rows within the window (WHERE observed_at >= now - window). The window
// thus lives as a read-time filter; here we just record what we saw and prune
// rows older than the window so the table can't grow without bound.
//
// One RAW row per (peerId, service): the seller's announced prices/cap/reputation
// as columns, no interpretation — ranking/admission stays in offers_sync (host)
// and the Σ_pol policy (core). Type-cleaning (positive-int cap, numeric
// reputation, non-null cached price) happens here, at the write, mirroring the
// coercion sources/antseed.py used to do at the read.
//
// Validation: a non-dump (the CLI prints a human "No peers found" line even with
// --json) exits non-zero so entrypoint.sh keeps the last good window (no write).
const { Client } = require("pg");
const { pgConfig } = require("./db.js");

const WINDOW_MS = Number(process.env.ANTSEED_PEER_WINDOW_MS || 15 * 60 * 1000);
const now = Date.now();

function numOr0(v) { const n = Number(v); return Number.isFinite(n) ? n : 0; }
function numOrNull(v) {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
function posIntOrNull(v) {  // maxConcurrency: positive int, else ungated (null)
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? Math.trunc(n) : null;
}

let fresh;
try { fresh = JSON.parse(require("fs").readFileSync(0, "utf8")); }
catch (e) { process.exit(2); }                            // not JSON
if (fresh === null || typeof fresh !== "object" || !Array.isArray(fresh.peers))
  process.exit(3);                                        // JSON but not a dump

// Flatten the nested browse dump (peer -> providerPricing -> services) to one
// row per (peerId, service). No services -> the peer contributes no rows.
const rows = [];
for (const peer of fresh.peers) {
  if (!peer || !peer.peerId) continue;
  const maxc = posIntOrNull(peer.maxConcurrency);
  const rep = numOrNull(peer.onChainReputationScore);
  const lastSeen = numOrNull(peer.lastSeen);
  for (const pricing of Object.values(peer.providerPricing || {})) {
    for (const [service, sp] of Object.entries((pricing || {}).services || {})) {
      rows.push([
        peer.peerId, service,
        numOr0(sp.inputUsdPerMillion), numOr0(sp.outputUsdPerMillion),
        numOrNull(sp.cachedInputUsdPerMillion),
        maxc, rep, lastSeen, now, now, now,
      ]);
    }
  }
}

const UPSERT = `INSERT INTO peer_offers
  (peer_id, service, price_in, price_out, price_cached_in, max_concurrency,
   reputation, last_seen, observed_at, first_seen, fetched_at)
  VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
  ON CONFLICT (peer_id, service) DO UPDATE SET
    price_in=EXCLUDED.price_in, price_out=EXCLUDED.price_out,
    price_cached_in=EXCLUDED.price_cached_in,
    max_concurrency=EXCLUDED.max_concurrency, reputation=EXCLUDED.reputation,
    last_seen=EXCLUDED.last_seen, observed_at=EXCLUDED.observed_at,
    fetched_at=EXCLUDED.fetched_at`;  // first_seen preserved across conflicts

(async () => {
  const client = new Client(pgConfig(process.env.DATABASE_URL));
  try {
    await client.connect();
    for (const r of rows) await client.query(UPSERT, r);
    await client.query("DELETE FROM peer_offers WHERE observed_at < $1",
                       [now - WINDOW_MS]);
  } catch (e) {
    process.stderr.write(`write-market: ${e.message}\n`);
    process.exitCode = 4;                                 // DB error -> keep window
  } finally {
    await client.end().catch(() => {});
  }
})();
