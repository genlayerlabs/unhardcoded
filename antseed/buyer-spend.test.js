'use strict';
const assert = require('node:assert/strict');
const { once } = require('node:events');
const http = require('node:http');
const test = require('node:test');
const { createSpendServer, SETTLED_TOPIC } = require('./buyer-spend.js');

const buyer = '0x' + 'ab'.repeat(20);
const channels = '0x' + 'cd'.repeat(20);
const buyerTopic = '0x' + buyer.slice(2).padStart(64, '0');
const channelTopic = '0x' + '12'.repeat(32);
const midnight = Date.parse('2026-11-25') / 1000;
const clock = () => (midnight + 2 * 86400) * 1000;
const hex = (number) => '0x' + number.toString(16);
const timestamps = [midnight - 86400, midnight - 2, midnight, midnight + 2, midnight + 86400];

function settlement(block, cumulative, delta) {
  return {
    address: channels, blockNumber: hex(block), logIndex: '0x2',
    topics: [SETTLED_TOPIC, channelTopic, buyerTopic, '0x' + '0'.repeat(64)],
    data: '0x' + [cumulative, delta, cumulative, 0n, 160n, 0n]
      .map((word) => word.toString(16).padStart(64, '0')).join(''),
  };
}

async function fixture(t) {
  let tip = 2;
  let available = true;
  const events = [
    settlement(1, 9007199254740993n, 9007199254740993n),
    settlement(2, 9007199254741000n, 7n),
    settlement(3, 9007199254741011n, 11n),
  ];
  let calls = 0;
  async function rpc(method, params) {
    calls++;
    if (method === 'eth_chainId') return hex(8453);
    if (method === 'eth_getBlockByNumber') {
      const block = params[0] === 'finalized' ? tip : Number(params[0]);
      return { number: hex(block), timestamp: hex(timestamps[block]) };
    }
    assert.equal(method, 'eth_getLogs');
    const [filter] = params;
    assert.deepEqual(filter.topics, [SETTLED_TOPIC, null, buyerTopic]);
    assert.equal(filter.address, channels);
    if (!available) throw new Error('logs unavailable');
    return events.filter((event) => Number(event.blockNumber) >= Number(filter.fromBlock) &&
      Number(event.blockNumber) <= Number(filter.toBlock));
  }
  const upstream = http.createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const { id, method, params } = JSON.parse(Buffer.concat(chunks));
    let reply;
    try {
      reply = { result: await rpc(method, params) };
    } catch (error) {
      reply = { error: { code: -32000, message: error.message } };
    }
    res.end(JSON.stringify({ jsonrpc: '2.0', id, ...reply }));
  });
  upstream.listen(0, '127.0.0.1');
  await once(upstream, 'listening');
  t.after(() => upstream.close());
  const upstreamUrl = 'http://127.0.0.1:' + upstream.address().port;
  const token = 'r'.repeat(32);
  const server = createSpendServer({ token, rpcUrl: upstreamUrl, buyer, channels, chainId: 8453, clock });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => server.close());
  const url = 'http://127.0.0.1:' + server.address().port;
  const headers = { 'x-antseed-spend-token': token };
  return {
    read: (date) => fetch(url + '/daily-spend?date=' + date, { headers }),
    request: (path, options) => fetch(url + path, options),
    headers,
    finalize: () => { tip = 4; },
    failLogs: () => { available = false; },
    calls: () => calls,
  };
}

test('one channel settles on both sides of midnight; late finalization fills the original day', async (t) => {
  const feed = await fixture(t);
  const before = await (await feed.read('2026-11-24')).json();
  assert.equal(before.spend_micro_usdc, '9007199254740993');
  assert.equal(before.complete, true);
  const initial = await (await feed.read('2026-11-25')).json();
  assert.equal(initial.spend_micro_usdc, '7');
  assert.equal(initial.complete, false);
  assert.equal(initial.finalized_through, '2026-11-25T00:00:00.000Z');
  feed.finalize();
  assert.deepEqual(await (await feed.read('2026-11-25')).json(), {
    buyer, channels, chain_id: 8453, date: '2026-11-25', basis: 'settlement',
    spend_micro_usdc: '18', complete: true,
    finalized_through: '2026-11-26T00:00:00.000Z',
    observed_at: '2026-11-27T00:00:00.000Z',
  });
  assert.equal((await (await feed.read('2026-11-24')).json()).spend_micro_usdc, before.spend_micro_usdc);
});

test('requires the monitoring token, validates the date, and only serves the read endpoint', async (t) => {
  const feed = await fixture(t);
  assert.equal((await feed.request('/daily-spend?date=2026-11-25')).status, 401);
  assert.equal((await feed.request('/deposit', { method: 'POST', headers: feed.headers })).status, 404);
  assert.equal((await feed.request('/daily-spend?date=2026-11-25',
    { method: 'POST', headers: feed.headers })).status, 404);
  for (const date of ['', '2026-02-30', '2026-11-28', '2026-11-25T00:00:00Z']) {
    assert.equal((await feed.read(date)).status, 400);
  }
  assert.equal(feed.calls(), 0);
});

test('an RPC failure returns unavailable without a partial amount', async (t) => {
  const feed = await fixture(t);
  feed.failLogs();
  const response = await feed.read('2026-11-25');
  assert.equal(response.status, 502);
  assert.deepEqual(await response.json(), { error: 'settlement data unavailable' });
});
