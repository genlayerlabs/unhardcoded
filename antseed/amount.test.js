// The control server's deposit-amount guard. It used to accept ANY positive
// value, which was tolerable while every deposit was typed by a human into the
// dashboard. It is not tolerable now: the router's wallet keeper calls /deposit
// autonomously, so this endpoint is reachable by a loop, a bad operator knob, or
// anyone who gets hold of ANTSEED_CONTROL_TOKEN. The ceiling has to hold
// server-side, next to the CLI that actually spends — not only in the caller.
//
// Run: node --test antseed/amount.test.js
"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { validAmount, MAX_AMOUNT_USDC } = require("./amount.js");

test("accepts well-formed USDC amounts at or under the cap", () => {
  for (const ok of ["1", "0.000001", "5", "5.5", "49.999999", "50"]) {
    assert.equal(validAmount(ok), true, `${ok} should be accepted`);
  }
});

test("rejects amounts over the 50 USDC cap (the regression this closes)", () => {
  assert.equal(MAX_AMOUNT_USDC, 50);
  for (const bad of ["50.000001", "51", "100", "1000", "999999"]) {
    assert.equal(validAmount(bad), false, `${bad} should be rejected`);
  }
});

test("rejects non-positive, malformed and non-string amounts", () => {
  for (const bad of ["0", "0.0", "-1", "1e3", "abc", "", "1.1234567", " 5"]) {
    assert.equal(validAmount(bad), false, `${bad} should be rejected`);
  }
  // A JSON number must not slip past: the endpoint stringifies its input, and a
  // loose coercion here would defeat both the shape check and the cap.
  for (const bad of [5, null, undefined, {}, []]) {
    assert.equal(validAmount(bad), false);
  }
});
