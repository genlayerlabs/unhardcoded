const { test } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const { createPublicProxy } = require('./public-proxy.js');

test('network face of the buyer proxy demands the bearer and forwards transparently', async () => {
  const received = [];
  const buyer = http.createServer((req, res) => {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => {
      received.push({ method: req.method, url: req.url, headers: req.headers, body });
      res.setHeader('content-type', 'text/event-stream');
      res.end('data: {"ok":true}\n\n');
    });
  });
  await new Promise(resolve => buyer.listen(0, '127.0.0.1', resolve));
  const token = 'p'.repeat(40);
  assert.throws(() => createPublicProxy({ token: 'short' }));
  const proxy = createPublicProxy({ token, proxyPort: buyer.address().port });
  await new Promise(resolve => proxy.listen(0, '127.0.0.1', resolve));
  const base = 'http://127.0.0.1:' + proxy.address().port;
  try {
    assert.equal((await fetch(base + '/v1/chat/completions', { method: 'POST', body: '{}' })).status, 401);
    assert.equal((await fetch(base + '/v1/chat/completions', { method: 'POST', body: '{}',
      headers: { authorization: 'Bearer ' + 'q'.repeat(40) } })).status, 401);
    assert.equal(received.length, 0);
    const r = await fetch(base + '/v1/chat/completions', { method: 'POST', body: '{"a":1}',
      headers: { authorization: 'Bearer ' + token, 'x-antseed-pin-peer': 'peer-a' } });
    assert.equal(r.status, 200);
    assert.equal(await r.text(), 'data: {"ok":true}\n\n');
    assert.equal(received[0].url, '/v1/chat/completions');
    assert.equal(received[0].body, '{"a":1}');
    assert.equal(received[0].headers['x-antseed-pin-peer'], 'peer-a');
    assert.equal(received[0].headers.authorization, undefined);
  } finally {
    await Promise.all([new Promise(r => proxy.close(r)), new Promise(r => buyer.close(r))]);
  }
});

test('entrypoint uses the authenticated forwarder whenever a proxy token is configured', () => {
  const src = fs.readFileSync(__dirname + '/entrypoint.sh', 'utf8');
  assert.match(src, /if \[ -n "\$\{ANTSEED_PROXY_TOKEN:-\}" \]; then\n\s+node "\$LIB\/public-proxy\.js" &\nelse/);
});
