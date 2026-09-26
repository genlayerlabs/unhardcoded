const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { tokenMatches, listenHost } = require('./auth.js');

test('token compare is exact and total', () => {
  const t = 'x'.repeat(32);
  assert.equal(tokenMatches(t, t), true);
  assert.equal(tokenMatches(t + 'y', t), false);
  assert.equal(tokenMatches(t.slice(1), t), false);
  assert.equal(tokenMatches(undefined, t), false);
  assert.equal(tokenMatches(['a'], t), false);
  assert.equal(tokenMatches('', ''), false);        // unset token never matches
  assert.equal(tokenMatches('é'.repeat(16), t), false);  // multibyte, same char count
});

test('control server compares in constant time and binds pod-local by default', () => {
  const src = fs.readFileSync(__dirname + '/control.js', 'utf8');
  assert.doesNotMatch(src, /!==\s*TOKEN/);
  assert.match(src, /tokenMatches\(req\.headers\['x-antseed-control-token'\], TOKEN\)/);
  assert.match(src, /server\.listen\(PORT, HOST,/);
  assert.equal(listenHost({}, 'ANTSEED_CONTROL_HOST'), '127.0.0.1');
  assert.equal(listenHost({ ANTSEED_CONTROL_HOST: '0.0.0.0' }, 'ANTSEED_CONTROL_HOST'), '0.0.0.0');
});
