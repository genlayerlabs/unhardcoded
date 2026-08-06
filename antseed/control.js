// AntSeed sidecar control server — lets the operator dashboard run wallet ops
// (deposit / withdraw / status) without `kubectl exec`. The buyer CLI + funded
// identity live ONLY in this container, so the dashboard reaches them through:
//
//   catalog button -> auth_proxy /dashboard/api/wallet/* -> router /x/wallet/*
//                  -> THIS server (:8379) -> `antseed buyer <cmd>`
//
// Guarded by a shared token (ANTSEED_CONTROL_TOKEN); if unset the server does
// not start (feature disabled, router degrades to 503). Listens on the pod/
// container network only — never published. Subcommands are whitelisted and run
// via execFile with array args (no shell); amounts are strictly validated.
'use strict';
const http = require('http');
const { execFile } = require('child_process');
const { Pool } = require('pg');
const { pgConfig } = require('./db.js');
const { UPSERT_BUYER_STATUS, buyerStatusRow } = require('./store.js');
// Amount validation (incl. the hard per-deposit ceiling) lives in its own
// dependency-free module so it can be unit-tested outside the sidecar image.
const { validAmount, MAX_AMOUNT_USDC } = require('./amount.js');
const { encodeIds } = require('./ids.js');
const { createQueue } = require('./queue.js');

const path = require('path');

const PORT = parseInt(process.env.ANTSEED_CONTROL_PORT || '8379', 10);
const TOKEN = process.env.ANTSEED_CONTROL_TOKEN || '';
const PID = process.env.ANTSEED_BUYER_PID || 'antseed';
const DEPOSIT_TIMEOUT_MS = 120000; // on-chain tx
const STATUS_TIMEOUT_MS = 30000;
// The post-op buyer_status write. It used to be unbounded, which made the
// server's worst case unbounded too — and a caller cannot pick a timeout that
// strictly exceeds an unbounded budget, so every deposit was one slow query away
// from a client-side timeout on a request the CLI had already executed.
const DB_TIMEOUT_MS = 10000;
// How long a mutation may sit in the serialize() queue before it is REJECTED
// rather than run. Without it a deposit queued behind a 240s reclaim phase blows
// the caller's timeout while still executing afterwards — the caller gives up,
// the CLI spends anyway, and nothing in the ledger says so. Rejecting is safe
// precisely because it happens before the CLI runs: nothing was attempted.
const QUEUE_WAIT_BUDGET_MS = 30000;
// Channel reclaim runs @antseed/node's ChannelsClient via reclaim.mjs (no CLI
// verb exists). Scans are read-only RPC; request-close/withdraw send one tx per
// channel, so the on-chain phases get the longer, deposit-grade budget.
const RECLAIM_PATH = path.join(__dirname, 'reclaim.mjs');
const RECLAIM_SCAN_TIMEOUT_MS = 90000;
const RECLAIM_TX_TIMEOUT_MS = 240000;

// The server's OWN worst case per endpoint, in ms. Published on /budgets so the
// caller's timeout can be derived from it instead of guessed: a client timeout
// below the server's worst case is not a timeout, it is a lie — the sidecar goes
// on executing a request the caller has already written off.
// wallet_keeper.py mirrors these numbers and asserts against them.
const BUDGETS_MS = {
  deposit: QUEUE_WAIT_BUDGET_MS + DEPOSIT_TIMEOUT_MS + STATUS_TIMEOUT_MS + DB_TIMEOUT_MS,
  status: STATUS_TIMEOUT_MS + DB_TIMEOUT_MS,
  'reclaim/scan': RECLAIM_SCAN_TIMEOUT_MS,
  'reclaim/tx': QUEUE_WAIT_BUDGET_MS + RECLAIM_TX_TIMEOUT_MS + STATUS_TIMEOUT_MS + DB_TIMEOUT_MS,
};

// One pool for the long-lived control server (write-status.js, the poll-loop
// twin, is one-shot and uses a plain Client instead).
const pool = new Pool(pgConfig(process.env.DATABASE_URL));

if (!TOKEN) {
  console.error('[control] ANTSEED_CONTROL_TOKEN unset — control server disabled');
  return;
}

function run(args, timeout) {
  return new Promise((resolve) => {
    execFile('antseed', args, { timeout, maxBuffer: 4 * 1024 * 1024 },
      (err, stdout, stderr) => {
        resolve({ code: err ? (err.code || 1) : 0,
                  // `killed` means WE SIGTERMed the CLI on the timeout above —
                  // which says nothing about whether it had already broadcast
                  // the transaction. Surfaced so the response can admit that
                  // instead of reporting a plain failure.
                  killed: Boolean(err && (err.killed || err.signal)),
                  stdout: String(stdout || ''), stderr: String(stderr || '') });
      });
  });
}

// Run reclaim.mjs <phase> [ids] and parse its single-JSON-object stdout. The
// script always emits a JSON envelope (even on error), so a non-JSON stdout
// means the node process itself died (import/crash) — surfaced as a 502.
function runReclaim(phase, timeout, ids) {
  const args = ids ? [RECLAIM_PATH, phase, ids] : [RECLAIM_PATH, phase];
  return new Promise((resolve) => {
    execFile('node', args, { timeout, maxBuffer: 8 * 1024 * 1024 },
      (err, stdout, stderr) => {
        let data = null;
        try { data = JSON.parse(String(stdout || '')); } catch (_) {}
        resolve({ code: err ? (err.code || 1) : 0, data,
                  killed: Boolean(err && (err.killed || err.signal)),
                  stderr: String(stderr || ''), stdout: String(stdout || '') });
      });
  });
}

// Upsert the buyer's status into the host store (buyer_status) from a fresh
// `buyer status --json`, so the router's source picks up the new escrow balance
// on its next read — the post-wallet-op twin of write-status.js. Returns the
// fresh status object for the HTTP response even if the persist fails (the poll
// loop will retry the write); null only when the CLI output isn't a status.
async function refreshStatus() {
  const r = await run(['buyer', 'status', '--json'], STATUS_TIMEOUT_MS);
  let data;
  try { data = JSON.parse(r.stdout); } catch (_) { return null; }
  if (data === null || typeof data !== 'object') return null;
  data.fetched_at_ms = Date.now();
  try {
    // Bounded: this write is on the critical path of a caller that has already
    // spent, and an unbounded query here is what put the server's worst case
    // above every timeout the caller could reasonably pick.
    await Promise.race([
      pool.query(UPSERT_BUYER_STATUS, buyerStatusRow(data, PID)),
      new Promise((_, rej) => setTimeout(
        () => rej(new Error('db write exceeded ' + DB_TIMEOUT_MS + 'ms')),
        DB_TIMEOUT_MS).unref()),
    ]);
  } catch (e) {
    console.error('[control] buyer_status upsert failed:', e.message);
  }
  return data;
}

// Serialize wallet mutations: two concurrent deposits would race the buyer's
// sqlite store / nonce. Bounded IN TIME, so the server's worst case is finite
// and a caller can pick a timeout that strictly exceeds it (see queue.js).
const serialize = createQueue(QUEUE_WAIT_BUDGET_MS);

function send(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve) => {
    let b = '';
    req.on('data', (c) => { b += c; if (b.length > 1e6) req.destroy(); });
    req.on('end', () => { try { resolve(JSON.parse(b || '{}')); } catch (_) { resolve(null); } });
    req.on('error', () => resolve(null));
  });
}

// `attempted` tells the CALLER whether the buyer CLI ran. It is the difference
// between "nothing happened, retry freely" and "a transaction may be on Base
// mainnet right now" — the router's keeper records the first as `failed` (costs
// nothing) and the second as `unknown` (counts as spent). Getting it wrong in
// the optimistic direction moves real USDC with the ledger recording nothing, so
// every branch below states it explicitly rather than defaulting.
function refuse(res, status, error) {          // provably nothing was attempted
  return send(res, status, { ok: false, error, attempted: false });
}
function inconclusive(res, status, error) {    // the CLI ran; outcome unknowable
  return send(res, status, { ok: false, error, attempted: true });
}

const server = http.createServer(async (req, res) => {
  if (req.headers['x-antseed-control-token'] !== TOKEN) {
    return refuse(res, 401, 'unauthorized');
  }
  const url = (req.url || '').split('?')[0];

  // The server's own worst-case budgets, so a caller can size its timeout to
  // strictly exceed them instead of guessing (read-only, no side effects).
  if (req.method === 'POST' && url === '/budgets') {
    return send(res, 200, { ok: true, budgets_ms: BUDGETS_MS });
  }

  if (req.method === 'POST' && (url === '/deposit' || url === '/withdraw')) {
    const verb = url.slice(1);
    const body = await readBody(req);
    const amount = body && body.amount != null ? String(body.amount) : '';
    if (!validAmount(amount)) {
      return refuse(res, 400, 'amount must be a positive USDC value (<=6 decimals, <=' + MAX_AMOUNT_USDC + ')');
    }
    return serialize(async () => {
      const r = await run(['buyer', verb, amount], DEPOSIT_TIMEOUT_MS);
      if (r.code !== 0) {
        const why = (r.stderr || r.stdout || 'cli failed').slice(0, 600);
        // A CLI we KILLED on the timeout may already have broadcast the
        // transaction; only a CLI that exited on its own proves it did not.
        return r.killed
          ? inconclusive(res, 504, 'buyer ' + verb + ' timed out after ' +
              DEPOSIT_TIMEOUT_MS + 'ms and was killed — the transaction may have ' +
              'been broadcast: ' + why)
          : inconclusive(res, 502, why);
      }
      const status = await refreshStatus();
      return send(res, 200, { ok: true, attempted: true, action: verb, amount, stdout: r.stdout.slice(0, 600), status });
    }, (waited) => refuse(res, 429,
      'control queue busy: waited ' + waited + 'ms without starting (budget ' +
      QUEUE_WAIT_BUDGET_MS + 'ms) — nothing was executed'));
  }

  if (req.method === 'POST' && url === '/status') {
    const status = await refreshStatus();
    if (!status) return inconclusive(res, 502, 'status unavailable');
    return send(res, 200, { ok: true, attempted: true, status });
  }

  // Read-only: enumerate payment channels and their on-chain reclaimable USDC.
  // Not serialized — it sends no transaction, so it cannot race the nonce.
  if (req.method === 'POST' && url === '/reclaim/scan') {
    const r = await runReclaim('list', RECLAIM_SCAN_TIMEOUT_MS);
    if (!r.data) {
      return inconclusive(res, 502, (r.stderr || 'reclaim scan failed').slice(0, 600));
    }
    return send(res, r.data.ok ? 200 : 502, { ...r.data, attempted: true });
  }

  // On-chain, one tx per SELECTED eligible channel (set-operator is a single tx).
  // Serialized with deposits/withdraws: concurrent buyer wallet txs race the nonce.
  if (req.method === 'POST' && (url === '/reclaim/set-operator' || url === '/reclaim/request-close' || url === '/reclaim/withdraw')) {
    const phase = url === '/reclaim/set-operator' ? 'set-operator'
      : url === '/reclaim/request-close' ? 'request-close' : 'withdraw';
    const body = await readBody(req);
    // Optional channel-id list. It is what makes the caller's per-cycle
    // transaction cap real — without it reclaim.mjs acts on EVERY eligible
    // channel and the caller's own filter is decorative. Omitted = act on all
    // (the human-by-hand path); malformed = refuse, never silently widen.
    const sel = encodeIds(body && body.ids != null ? body.ids : undefined);
    if (!sel.ok) return refuse(res, 400, sel.error);
    return serialize(async () => {
      const r = await runReclaim(phase, RECLAIM_TX_TIMEOUT_MS, sel.value);
      if (!r.data) {
        const why = (r.stderr || 'reclaim ' + phase + ' failed').slice(0, 600);
        return r.killed
          ? inconclusive(res, 504, 'reclaim ' + phase + ' timed out after ' +
              RECLAIM_TX_TIMEOUT_MS + 'ms and was killed — transactions may have ' +
              'been broadcast: ' + why)
          : inconclusive(res, 502, why);
      }
      // A fresh status write so the router picks up the freed escrow on withdraw.
      if (phase === 'withdraw') { try { await refreshStatus(); } catch (_) {} }
      return send(res, r.data.ok ? 200 : 502, { ...r.data, attempted: true });
    }, (waited) => refuse(res, 429,
      'control queue busy: waited ' + waited + 'ms without starting (budget ' +
      QUEUE_WAIT_BUDGET_MS + 'ms) — nothing was executed'));
  }

  return refuse(res, 404, 'not found');
});

server.listen(PORT, () => console.error('[control] listening on :' + PORT));
