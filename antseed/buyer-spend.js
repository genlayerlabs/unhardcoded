// Read-only AntSeed settlement feed. Uses a public buyer address and JSON-RPC
// reads only. It never loads the buyer CLI, identity, or channel database.
'use strict';
const http = require('http');
const { listenHost, tokenMatches } = require('./auth.js');

const DAY_SECONDS = 86400;
const MAX_DAY_BLOCKS = 50000; // Base produces about 43,200 blocks/day.
const LOG_BATCH_BLOCKS = 2000;
// keccak256 of ChannelSettled(bytes32,address,address,uint128,uint128,uint128,uint256,bytes).
// The contract emits this for every charge, including charges in topUp and close.
const SETTLED_TOPIC = '0x0b287f37d8bd14ef37f2966734ab387c243cc1a1663616a25a4cc259877736b1';
const BASE_CHANNELS = '0xba66d3b4fbcf472f6f11d6f9f96aace96516f09d';
const ADDRESS = /^0x[0-9a-f]{40}$/i;

function quantity(value) {
  if (typeof value !== 'string' || !/^0x[0-9a-f]+$/i.test(value)) {
    throw new Error('invalid RPC quantity');
  }
  const result = Number(value);
  if (!Number.isSafeInteger(result)) throw new Error('RPC quantity out of range');
  return result;
}

const hex = (number) => '0x' + number.toString(16);

async function dailySpend(rpc, { buyer, channels, chainId }, start) {
  if (quantity(await rpc('eth_chainId', [])) !== chainId) throw new Error('wrong chain');
  const finalized = await rpc('eth_getBlockByNumber', ['finalized', false]);
  const through = quantity(finalized.timestamp);
  const tip = quantity(finalized.number);
  const end = start + DAY_SECONDS;

  // Find the first finalized block at or after a UTC boundary. A boundary
  // beyond the finalized head maps to the exclusive end of the scanned range.
  async function boundary(timestamp) {
    if (timestamp > through) return tip + 1;
    let low = 0;
    let high = tip;
    while (low < high) {
      const mid = low + Math.floor((high - low) / 2);
      const block = await rpc('eth_getBlockByNumber', [hex(mid), false]);
      if (quantity(block.number) !== mid) throw new Error('wrong block');
      if (quantity(block.timestamp) < timestamp) low = mid + 1;
      else high = mid;
    }
    return low;
  }

  const [from, to] = await Promise.all([boundary(start), boundary(end)]);
  if (to < from || to - from > MAX_DAY_BLOCKS) throw new Error('invalid block window');
  const buyerTopic = '0x' + buyer.slice(2).toLowerCase().padStart(64, '0');
  let total = 0n;
  for (let first = from; first < to; first += LOG_BATCH_BLOCKS) {
    const last = Math.min(first + LOG_BATCH_BLOCKS, to) - 1;
    const logs = await rpc('eth_getLogs', [{
      address: channels, topics: [SETTLED_TOPIC, null, buyerTopic],
      fromBlock: hex(first), toBlock: hex(last),
    }]);
    if (!Array.isArray(logs)) throw new Error('invalid settlement logs');
    const seen = new Set();
    for (const log of logs) {
      const block = quantity(log.blockNumber);
      const position = block + '/' + quantity(log.logIndex);
      if (log.removed || block < first || block > last ||
          log.address?.toLowerCase() !== channels.toLowerCase() ||
          log.topics?.[0] !== SETTLED_TOPIC || log.topics?.[2] !== buyerTopic ||
          typeof log.data !== 'string' || !/^0x(?:[0-9a-f]{64}){6,}$/i.test(log.data)) {
        throw new Error('invalid settlement event');
      }
      if (seen.has(position)) continue;
      seen.add(position);
      // ABI word 1 is the newly charged delta. The other amount words are cumulative.
      const delta = BigInt('0x' + log.data.slice(66, 130));
      if (delta <= 0n || delta >= (1n << 128n)) throw new Error('invalid settlement delta');
      total += delta;
    }
  }
  return {
    spend_micro_usdc: total.toString(),
    finalized_through: new Date(through * 1000).toISOString(),
    complete: through >= end,
  };
}

function createSpendServer({ token, rpcUrl, buyer, channels, chainId, clock = Date.now }) {
  let busy = false;
  return http.createServer(async (req, res) => {
    function send(status, body) {
      res.writeHead(status, { 'content-type': 'application/json', 'cache-control': 'no-store' });
      res.end(JSON.stringify(body));
    }
    if (!tokenMatches(req.headers['x-antseed-spend-token'], token)) {
      return send(401, { error: 'unauthorized' });
    }
    if (!URL.canParse(req.url, 'http://localhost')) return send(400, { error: 'invalid URL' });
    const url = new URL(req.url, 'http://localhost');
    if (req.method !== 'GET' || url.pathname !== '/daily-spend') {
      return send(404, { error: 'not found' });
    }
    const date = url.searchParams.get('date');
    const start = /^\d{4}-\d{2}-\d{2}$/.test(date || '') ? Date.parse(date) / 1000 : NaN;
    const today = Math.floor(clock() / 1000 / DAY_SECONDS) * DAY_SECONDS;
    if (!Number.isFinite(start) || new Date(start * 1000).toISOString().slice(0, 10) !== date ||
        start > today) {
      return send(400, { error: 'date must be a UTC date (YYYY-MM-DD) no later than today' });
    }
    if (busy) return send(429, { error: 'settlement read in progress' });
    busy = true;
    const scan = new AbortController();
    try {
      const signal = AbortSignal.any([scan.signal, AbortSignal.timeout(20000)]);
      const result = await dailySpend(rpcClient(rpcUrl, signal), { buyer, channels, chainId }, start);
      return send(200, { buyer, channels, chain_id: chainId, date, basis: 'settlement',
        ...result, observed_at: new Date(clock()).toISOString() });
    } catch {
      return send(502, { error: 'settlement data unavailable' });
    } finally {
      scan.abort();
      busy = false;
    }
  });
}

function rpcClient(url, signal) {
  return async (method, params) => {
    const response = await fetch(url, {
      method: 'POST', redirect: 'error', headers: { 'content-type': 'application/json' }, signal,
      body: JSON.stringify({ jsonrpc: '2.0', id: 1, method, params }),
    });
    if (!response.ok) throw new Error('settlement RPC request failed');
    // Bound the decoded response too, including for RPCs that ignore log filters.
    const chunks = [];
    let size = 0;
    for await (const chunk of response.body) {
      size += chunk.length;
      if (size > 4 * 1024 * 1024) throw new Error('settlement RPC response too large');
      chunks.push(chunk);
    }
    const data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    if (data.error || data.id !== 1 || !Object.hasOwn(data, 'result')) {
      throw new Error('invalid settlement RPC response');
    }
    return data.result;
  };
}

if (require.main === module && process.env.ANTSEED_SPEND_TOKEN) {
  const env = process.env;
  const token = env.ANTSEED_SPEND_TOKEN;
  const buyer = env.ANTSEED_SPEND_BUYER || '';
  const channels = env.ANTSEED_SPEND_CHANNELS || BASE_CHANNELS;
  const chainId = Number(env.ANTSEED_SPEND_CHAIN_ID || 8453);
  if (token.length < 32 || [env.ANTSEED_CONTROL_TOKEN, env.ANTSEED_PROXY_TOKEN,
    env.ANTSEED_BYO_TOKEN].includes(token) || !ADDRESS.test(buyer) || !ADDRESS.test(channels) ||
      !Number.isSafeInteger(chainId) || chainId <= 0 ||
      (chainId !== 8453 && !env.ANTSEED_SPEND_CHANNELS) || !env.ANTSEED_SPEND_RPC_URL) {
    throw new Error('invalid AntSeed spend feed configuration');
  }
  if (!URL.canParse(env.ANTSEED_SPEND_RPC_URL)) throw new Error('invalid AntSeed spend RPC URL');
  const rpcUrl = new URL(env.ANTSEED_SPEND_RPC_URL);
  if (!['http:', 'https:'].includes(rpcUrl.protocol) || rpcUrl.username || rpcUrl.password) {
    throw new Error('invalid AntSeed spend RPC URL');
  }
  createSpendServer({ token, rpcUrl, buyer, channels, chainId })
    .listen(Number(env.ANTSEED_SPEND_PORT || 8381), listenHost(env, 'ANTSEED_SPEND_HOST'));
}

module.exports = { createSpendServer, SETTLED_TOPIC };
