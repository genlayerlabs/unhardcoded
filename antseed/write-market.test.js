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
