// Authenticated network face of the buyer proxy. The buyer binds 127.0.0.1:8377
// and spends the funded wallet on every call, so whatever exposes it beyond the
// pod must check a credential: with ANTSEED_PROXY_TOKEN set, entrypoint.sh runs
// THIS instead of the bare socat forwarder and every request needs
// `Authorization: Bearer <token>` (the router sends it via the antseed
// provider's auth). Transparent otherwise: any method/path, streamed both ways.
'use strict';
const http = require('http');
const { tokenMatches } = require('./auth.js');

function createPublicProxy({ token, proxyPort = 8377 }) {
  if (!token || token.length < 32) throw new Error('ANTSEED_PROXY_TOKEN must have at least 32 characters');
  return http.createServer((req, res) => {
    if (!tokenMatches(req.headers.authorization, 'Bearer ' + token)) {
      res.writeHead(401, { 'content-type': 'application/json', 'cache-control': 'no-store' });
      return res.end(JSON.stringify({ error: 'unauthorized' }));
    }
    const headers = { ...req.headers };
    delete headers.authorization;     // the buyer proxy never needs our credential
    const upstream = http.request({ hostname: '127.0.0.1', port: proxyPort,
      path: req.url, method: req.method, headers }, response => {
      res.writeHead(response.statusCode, response.headers);
      response.on('error', () => res.destroy());
      response.pipe(res);
    });
    upstream.on('error', () => {
      if (!res.headersSent) {
        res.writeHead(502, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'buyer_unavailable' }));
      } else res.destroy();
    });
    res.on('close', () => upstream.destroy());
    req.pipe(upstream);
  });
}

if (require.main === module) {
  createPublicProxy({ token: process.env.ANTSEED_PROXY_TOKEN,
    proxyPort: Number(process.env.ANTSEED_PROXY_PORT || 8377) })
    .listen(Number(process.env.ANTSEED_PUBLIC_PORT || 8378), '0.0.0.0');
}

module.exports = { createPublicProxy };
