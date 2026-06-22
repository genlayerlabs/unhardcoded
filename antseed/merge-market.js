// Rolling-window market writer (replaces validate-market.js; subsumes its
// validate step). Each browse sees only the peers DHT-reachable from this node
// *right now* (~68); the network has more (~83) that flit in and out of our
// view. So instead of overwriting market.json with the instantaneous snapshot,
// we keep a SLIDING window: union the new browse with the previously written
// peers (keyed by peerId, freshest pricing/lastSeen win) and keep any peer we
// have seen within the last ANTSEED_PEER_WINDOW_MS of OUR OWN browsing. With a
// 60 s browse cadence and a 15 min window that's ~15 browses unioned, sliding
// forward every minute — a continuous crawl from this node's vantage, with
// steady coverage (no tumbling sawtooth) and no peer surfaced once it has been
// silent past the window.
//
// `_seen_at_ms` is OUR observation stamp (when this node last saw the peer in a
// browse) — distinct from the network's `lastSeen`. It drives the window;
// sources/antseed.py ignores unknown peer fields, so it's inert downstream.
//
// Validation: a non-dump (the CLI prints a human "No peers found" line even
// with --json) exits non-zero so the caller keeps the last good file.
const fs = require("fs");

const PREV_PATH = process.argv[2];                       // existing market.json
const WINDOW_MS = Number(process.env.ANTSEED_PEER_WINDOW_MS || 15 * 60 * 1000);
const now = Date.now();

let fresh;
try { fresh = JSON.parse(fs.readFileSync(0, "utf8")); }
catch (e) { process.exit(2); }                           // not JSON
if (fresh === null || typeof fresh !== "object" || !Array.isArray(fresh.peers))
  process.exit(3);                                       // JSON but not a dump

// seed from the previously accumulated window (first boot: empty)
const byId = new Map();
try {
  const prev = JSON.parse(fs.readFileSync(PREV_PATH, "utf8"));
  for (const p of prev.peers || []) if (p && p.peerId) byId.set(p.peerId, p);
} catch (e) { /* missing/unreadable -> start fresh */ }

// upsert this browse's peers; stamp them as seen-now (freshest data wins)
const seenNow = new Set();
for (const p of fresh.peers) if (p && p.peerId) {
  p._seen_at_ms = now;
  byId.set(p.peerId, p);
  seenNow.add(p.peerId);
}

// slide: keep only peers seen within the last WINDOW_MS of our browsing
const peers = [...byId.values()]
  .filter(p => Number(p._seen_at_ms) >= now - WINDOW_MS);

process.stdout.write(JSON.stringify({
  ...fresh,
  peers,
  fetched_at_ms: now,
  peers_seen_now: seenNow.size,
  peers_accumulated: peers.length,
  window_ms: WINDOW_MS,
}));
