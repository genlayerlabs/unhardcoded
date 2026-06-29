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
const { UPSERT_BUYER_STATUS, buyerStatusRow } = require('./store.js');

const PORT = parseInt(process.env.ANTSEED_CONTROL_PORT || '8379', 10);
const TOKEN = process.env.ANTSEED_CONTROL_TOKEN || '';
const PID = process.env.ANTSEED_BUYER_PID || 'antseed';
const DEPOSIT_TIMEOUT_MS = 120000; // on-chain tx
const STATUS_TIMEOUT_MS = 30000;

// One pool for the long-lived control server (write-status.js, the poll-loop
// twin, is one-shot and uses a plain Client instead).
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

if (!TOKEN) {
  console.error('[control] ANTSEED_CONTROL_TOKEN unset — control server disabled');
  return;
}

// USDC amount: human units, up to 6 decimals, strictly positive.
const AMOUNT_RE = /^\d+(\.\d{1,6})?$/;
function validAmount(s) {
  return typeof s === 'string' && AMOUNT_RE.test(s) && parseFloat(s) > 0;
}

function run(args, timeout) {
  return new Promise((resolve) => {
    execFile('antseed', args, { timeout, maxBuffer: 4 * 1024 * 1024 },
      (err, stdout, stderr) => {
        resolve({ code: err ? (err.code || 1) : 0,
                  stdout: String(stdout || ''), stderr: String(stderr || '') });
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
    await pool.query(UPSERT_BUYER_STATUS, buyerStatusRow(data, PID));
  } catch (e) {
    console.error('[control] buyer_status upsert failed:', e.message);
  }
  return data;
}

// Serialize wallet mutations: two concurrent deposits would race the buyer's
// sqlite store / nonce.
let chain = Promise.resolve();
function serialize(fn) {
  const next = chain.then(fn, fn);
  chain = next.catch(() => {});
  return next;
}

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

const server = http.createServer(async (req, res) => {
  if (req.headers['x-antseed-control-token'] !== TOKEN) {
    return send(res, 401, { ok: false, error: 'unauthorized' });
  }
  const url = (req.url || '').split('?')[0];

  if (req.method === 'POST' && (url === '/deposit' || url === '/withdraw')) {
    const verb = url.slice(1);
    const body = await readBody(req);
    const amount = body && body.amount != null ? String(body.amount) : '';
    if (!validAmount(amount)) {
      return send(res, 400, { ok: false, error: 'amount must be a positive USDC value (<=6 decimals)' });
    }
    return serialize(async () => {
      const r = await run(['buyer', verb, amount], DEPOSIT_TIMEOUT_MS);
      if (r.code !== 0) {
        return send(res, 502, { ok: false, error: (r.stderr || r.stdout || 'cli failed').slice(0, 600) });
      }
      const status = await refreshStatus();
      return send(res, 200, { ok: true, action: verb, amount, stdout: r.stdout.slice(0, 600), status });
    });
  }

  if (req.method === 'POST' && url === '/status') {
    const status = await refreshStatus();
    if (!status) return send(res, 502, { ok: false, error: 'status unavailable' });
    return send(res, 200, { ok: true, status });
  }

  return send(res, 404, { ok: false, error: 'not found' });
});

server.listen(PORT, () => console.error('[control] listening on :' + PORT));
