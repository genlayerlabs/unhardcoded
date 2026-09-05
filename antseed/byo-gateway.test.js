const { test } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const { createGateway } = require('./byo-gateway.js');

test('BYO token permits snapshot/inference, never wallet ops or unpinned traffic', async () => {
  const received = [];
  const proxy = http.createServer((req, res) => {
    received.push(req.headers);
    res.setHeader('content-type', 'text/event-stream');
    res.end('data: {"ok":true}\n\n');
  });
  await new Promise(resolve => proxy.listen(0, '127.0.0.1', resolve));
  const token = 't'.repeat(40);
  const gateway = createGateway({token, proxyPort: proxy.address().port,
    snapshot: async () => ({ peer_offers: [], buyer_status: { deposits_available: 3 } })});
  await new Promise(resolve => gateway.listen(0, '127.0.0.1', resolve));
  const base = 'http://127.0.0.1:' + gateway.address().port;
  const headers = { authorization: 'Bearer ' + token };
  try {
    assert.equal((await fetch(base + '/snapshot')).status, 401);
    assert.equal((await (await fetch(base + '/snapshot', {headers})).json()).buyer_status.deposits_available, 3);
    for (const path of ['/deposit','/withdraw','/reclaim','/status','/v1/models']) {
      assert.equal((await fetch(base + path, {method:'POST', headers})).status, 404);
    }
    assert.equal((await fetch(base + '/v1/chat/completions', {method:'POST',headers,body:'{}'})).status, 400);
    const result = await fetch(base + '/v1/chat/completions', {method:'POST',
      headers:{...headers,'x-antseed-pin-peer':'peer-a'},body:'{}'});
    assert.equal(await result.text(), 'data: {"ok":true}\n\n');
    assert.equal(received[0]['x-antseed-pin-peer'], 'peer-a');
    assert.equal(received[0].authorization, undefined);
  } finally {
    await Promise.all([new Promise(resolve => gateway.close(resolve)), new Promise(resolve => proxy.close(resolve))]);
  }
});
