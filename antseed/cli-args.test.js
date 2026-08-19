'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { walletCommandArgs } = require('./cli-args.js');

test('direct deposits use the post-0.1.137 --onchain syntax', () => {
  assert.deepEqual(walletCommandArgs('deposit', '1.25'),
    ['buyer', 'deposit', '--onchain', '1.25']);
});

test('withdraw syntax remains positional', () => {
  assert.deepEqual(walletCommandArgs('withdraw', '2'),
    ['buyer', 'withdraw', '2']);
});

test('unknown wallet verbs fail closed', () => {
  assert.throws(() => walletCommandArgs('sweep', '1'), /unsupported wallet command/);
});
