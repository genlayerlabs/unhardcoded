// USDC amount validation for the control server's wallet ops.
//
// Extracted from control.js for the same reason db.js was: control.js pulls in
// `pg`, which only exists inside the sidecar image, so anything left in that
// file cannot be unit-tested from the repo. This module has NO dependencies, so
// amount.test.js runs wherever the suite runs.
//
// The cap is the server-side half of the wallet keeper's limits. It used to be
// absent — any positive value went straight to `antseed buyer deposit` — which
// was tolerable while every deposit was typed by a human into the dashboard. It
// is not tolerable now that the router's keeper calls /deposit autonomously: a
// bad knob, a bug in the loop, or anyone holding ANTSEED_CONTROL_TOKEN would
// otherwise reach an unbounded spend. Mirrors wallet_keeper.MAX_TOPUP_USDC.
"use strict";

// Human units, up to 6 decimals (USDC's precision), strictly positive.
const AMOUNT_RE = /^\d+(\.\d{1,6})?$/;
const MAX_AMOUNT_USDC = 50;

function validAmount(s) {
  if (typeof s !== "string" || !AMOUNT_RE.test(s)) return false;
  const v = parseFloat(s);
  return v > 0 && v <= MAX_AMOUNT_USDC;
}

module.exports = { validAmount, MAX_AMOUNT_USDC, AMOUNT_RE };
