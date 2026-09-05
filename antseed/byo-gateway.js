// Optional customer-owned BYO gateway. Separate from wallet control: this token
// can spend through inference, but cannot deposit, withdraw or export keys.
// Terminate HTTPS at the customer's reverse proxy; do not expose the raw proxy.
'use strict';
const http = require('http');
const { timingSafeEqual } = require('crypto');

function createGateway({ token, snapshot, proxyPort = 8377 }) {
  if (!token || token.length < 32) throw new Error('ANTSEED_BYO_TOKEN must have at least 32 characters');
  const expected = Buffer.from('Bearer ' + token);
  return http.createServer(async (req, res) => {
    const auth = Buffer.from(req.headers.authorization || '');
    const send = (code, data) => {
      res.writeHead(code, { 'content-type': 'application/json', 'cache-control': 'no-store' });
      res.end(JSON.stringify(data));
    };
    if (auth.length !== expected.length || !timingSafeEqual(auth, expected)) {
      return send(401, { error: 'unauthorized' });
    }
    if (req.method === 'GET' && req.url === '/snapshot') {
      try { return send(200, await snapshot()); }
      catch (_) { return send(503, { error: 'buyer_state_unavailable' }); }
    }
    if (req.method !== 'POST' || req.url !== '/v1/chat/completions') {
      return send(404, { error: 'unsupported_endpoint' });
    }
    // Fixed local upstream, no caller-chosen URL or forwarded credentials.
    const chunks = [];
    let bytes = 0;
    try {
      for await (const chunk of req) {
        bytes += chunk.length;
        if (bytes > 1024 * 1024) return send(413, { error: 'request_too_large' });
        chunks.push(chunk);
      }
    } catch (_) { return res.destroy(); }
    const headers = { 'content-type': 'application/json' };
    const peer = req.headers['x-antseed-pin-peer'];
    if (typeof peer !== 'string' || !peer || peer.length > 256) {
      return send(400, { error: 'peer_pin_required' });
    }
    headers['x-antseed-pin-peer'] = peer;
    const upstream = http.request({ hostname: '127.0.0.1', port: proxyPort,
      path: '/v1/chat/completions', method: 'POST', headers, timeout: 120000 }, response => {
      res.writeHead(response.statusCode, {
        'content-type': response.headers['content-type'] || 'application/json',
        'cache-control': 'no-store',
      });
      response.on('error', () => res.destroy());
      response.pipe(res);
    });
    upstream.on('timeout', () => upstream.destroy());
    upstream.on('error', () => {
      if (!res.headersSent) send(502, { error: 'buyer_unavailable' });
      else res.destroy();
    });
    res.on('close', () => upstream.destroy());
    upstream.end(Buffer.concat(chunks));
  });
}

if (require.main === module && process.env.ANTSEED_BYO_TOKEN) {
  const { Pool } = require('pg');
  const { pgConfig } = require('./db.js');
  const pool = new Pool({ ...pgConfig(process.env.DATABASE_URL), statement_timeout: 5000 });
  const pid = process.env.ANTSEED_BUYER_PID || 'antseed';
  const snapshot = async () => {
    const [offers, status] = await Promise.all([
      pool.query(`SELECT peer_id, service, price_in, price_out, price_cached_in,
                         max_concurrency, reputation, last_seen, last_reached_at, observed_at
                  FROM peer_offers WHERE observed_at >= $1 LIMIT 10000`, [Date.now() - 900000]),
      pool.query(`SELECT pinned_peer_id, deposits_available, fetched_at
                  FROM buyer_status WHERE pid = $1`, [pid]),
    ]);
    return { peer_offers: offers.rows.map(row => ({ ...row, observed_at: Number(row.observed_at),
        last_reached_at: row.last_reached_at == null ? null : Number(row.last_reached_at),
        last_seen: row.last_seen == null ? null : Number(row.last_seen) })),
      buyer_status: status.rows[0] ? { ...status.rows[0], fetched_at: Number(status.rows[0].fetched_at) } : null };
  };
  createGateway({ token: process.env.ANTSEED_BYO_TOKEN, snapshot,
    proxyPort: Number(process.env.ANTSEED_PROXY_PORT || 8377) })
    .listen(Number(process.env.ANTSEED_BYO_PORT || 8380), '0.0.0.0');
}

module.exports = { createGateway };
