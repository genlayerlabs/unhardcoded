'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('browse writer preserves decision protocols and supports older announcements', async () => {
  const queries = [];
  const peer = (id, protocols) => ({ peerId: id, providerPricing: {
    typesafe: { services: { 'jev-1.13': {inputUsdPerMillion: 0.05, outputUsdPerMillion: 0} } } },
    ...(protocols ? { providerServiceApiProtocols: {typesafe: {services: {'jev-1.13': protocols}}} } : {}) });
  const input = JSON.stringify({peers: [peer('modern', ['typesafe-systemone']), peer('legacy'), peer('malformed', [7])]});
  class Client {
    async connect() {}
    async end() {}
    async query(sql, params) { queries.push({sql, params}); }
  }
  const processMock = {env: {}, stderr: {write() {}}, exit(code) {throw new Error('exit ' + code);} };
  vm.runInNewContext(fs.readFileSync(__dirname + '/write-market.js', 'utf8'), {
    process: processMock, console,
    require(name) {
      if (name === 'pg') return {Client};
      if (name === './db.js') return {pgConfig: () => ({})};
      if (name === 'fs') return {readFileSync: () => input};
      throw new Error('unexpected import');
    },
  });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(processMock.exitCode, undefined);
  const inserts = queries.filter(q => q.sql.startsWith('INSERT'));
  assert.equal(inserts.length, 3);
  assert.deepEqual(Array.from(inserts[0].params[12]), ['typesafe-systemone']);
  assert.equal(inserts[1].params[12], null);
  assert.equal(inserts[2].params[12], null);
  assert.match(inserts[0].sql, /protocols=COALESCE\(EXCLUDED.protocols, peer_offers.protocols\)/);
});

test('browse writer drops peers and services with oversized or non-id names', async () => {
  const queries = [];
  const svc = name => ({ [name]: { inputUsdPerMillion: 1, outputUsdPerMillion: 2 } });
  const peer = (id, services) => ({ peerId: id, providerPricing: { p: { services } } });
  const input = JSON.stringify({ peers: [
    peer('4668854ba3e8b094e6f48fbeb59cec1cfde162f2', { ...svc('anthropic/claude-opus-4.8'), ...svc('x'.repeat(129)),
      ...svc('evil\r\nx-injected: 1'), ...svc('qwen3-235b-a22b@fast') }),
    peer('a'.repeat(129), svc('ok-service')),
    peer('bad peer\n', svc('ok-service')),
    peer({ nested: true }, svc('ok-service')),
  ] });
  class Client {
    async connect() {}
    async end() {}
    async query(sql, params) { queries.push({ sql, params }); }
  }
  const processMock = { env: {}, stderr: { write() {} }, exit(code) { throw new Error('exit ' + code); } };
  vm.runInNewContext(fs.readFileSync(__dirname + '/write-market.js', 'utf8'), {
    process: processMock, console,
    require(name) {
      if (name === 'pg') return { Client };
      if (name === './db.js') return { pgConfig: () => ({}) };
      if (name === 'fs') return { readFileSync: () => input };
      throw new Error('unexpected import');
    },
  });
  await new Promise(resolve => setImmediate(resolve));
  const inserts = queries.filter(q => q.sql.startsWith('INSERT'));
  assert.deepEqual(inserts.map(q => q.params[1]), ['anthropic/claude-opus-4.8', 'qwen3-235b-a22b@fast']);
});
