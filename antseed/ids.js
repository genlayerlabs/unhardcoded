// Payment-channel id validation for the control server's reclaim phases.
//
// Extracted from control.js for the same reason amount.js and db.js were:
// control.js pulls in `pg`, which only exists inside the sidecar image, so
// anything left in that file cannot be unit-tested from the repo. This module
// has NO dependencies, so ids.test.js runs wherever the suite runs.
//
// WHY THE CONTROL SERVER TAKES AN ID LIST AT ALL. reclaim.mjs used to act on
// EVERY eligible channel per invocation, and the caller could only name a phase.
// The router's wallet keeper filters channels (skipping dust) and caps the batch
// at MAX_TX_PER_CYCLE, but neither bound reached the sidecar — 100 eligible
// channels with 1 worth reclaiming meant the keeper logged "1 channel" while the
// sidecar fired 100 transactions. The keeper now names the channels it decided
// on, so its filter and its per-cycle transaction cap are the truth.
//
// The ids reach `node reclaim.mjs <phase> <ids>` through execFile with array
// args (no shell), so this validator is defence in depth rather than the only
// thing between an id and a command line — but a channel id is an opaque token
// from a sqlite store, and an unvalidated one has no business on an argv.
"use strict";

// Session ids as @antseed/node writes them: opaque, url-safe-ish tokens. Kept
// deliberately narrow — anything outside this is not an id we produced. The
// first character must be alphanumeric so an "id" can never be read as an
// OPTION by whatever ends up parsing the argv (`--phase`, `-e`).
const CHANNEL_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;
// Mirrors wallet_keeper.MAX_TX_PER_CYCLE with headroom. A caller asking for more
// channels than this in one invocation is not the keeper.
const MAX_IDS = 32;

function validChannelId(s) {
  return typeof s === "string" && CHANNEL_ID_RE.test(s);
}

// Normalize a caller-supplied list (JSON body) into the argv string reclaim.mjs
// parses back. Returns { ok, ids, value } or { ok: false, error }.
// `undefined`/`null` means "no selection" — the legacy act-on-everything path,
// which stays reachable for a human running the script by hand.
function encodeIds(raw) {
  if (raw === undefined || raw === null) return { ok: true, ids: null, value: null };
  if (!Array.isArray(raw)) return { ok: false, error: "ids must be an array of channel ids" };
  if (raw.length === 0) return { ok: false, error: "ids must not be empty (omit it to act on every eligible channel)" };
  if (raw.length > MAX_IDS) return { ok: false, error: "at most " + MAX_IDS + " channel ids per call" };
  const ids = [];
  for (const id of raw) {
    if (!validChannelId(id)) return { ok: false, error: "malformed channel id" };
    if (!ids.includes(id)) ids.push(id);
  }
  return { ok: true, ids, value: ids.join(",") };
}

// The reclaim.mjs half: argv string -> Set, or null for "no selection".
function parseIds(argv) {
  if (argv === undefined || argv === null || argv === "") return null;
  const ids = String(argv).split(",").filter((s) => s !== "");
  return new Set(ids.filter(validChannelId));
}

module.exports = { validChannelId, encodeIds, parseIds, CHANNEL_ID_RE, MAX_IDS };
