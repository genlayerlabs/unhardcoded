// The reclaim channel-id selector. This is the piece that makes the router's
// per-cycle transaction cap and its dust filter TRUE rather than decorative:
// before it existed the keeper could decide "1 channel is worth closing" and the
// sidecar would still fire one transaction per eligible channel, because the
// only thing crossing the wire was a phase name.
//
// Run: node --test antseed/ids.test.js
"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { validChannelId, encodeIds, parseIds, MAX_IDS } = require("./ids.js");

test("accepts the opaque session ids @antseed/node writes", () => {
  for (const ok of ["c1", "abc-123", "a.b:c_d", "0xdeadBEEF", "x".repeat(128)]) {
    assert.equal(validChannelId(ok), true, `${ok} should be accepted`);
  }
});

test("rejects ids that have no business on an argv", () => {
  // execFile passes array args with no shell, so none of these could inject a
  // command today — but an id is an opaque token from a sqlite store, and the
  // narrow charset is what keeps that true if the call path ever changes.
  for (const bad of ["a b", "a,b", "a;rm -rf /", "$(id)", "a\nb", "--phase",
                     "", "x".repeat(129), "a/b"]) {
    assert.equal(validChannelId(bad), false, `${bad} should be rejected`);
  }
  for (const bad of [1, null, undefined, {}, ["c1"]]) {
    assert.equal(validChannelId(bad), false);
  }
});

test("encodeIds turns a caller's array into the argv reclaim.mjs parses", () => {
  const r = encodeIds(["c1", "c2"]);
  assert.deepEqual(r, { ok: true, ids: ["c1", "c2"], value: "c1,c2" });
  assert.deepEqual([...parseIds(r.value)], ["c1", "c2"]);
});

test("omitting ids stays the act-on-everything path (a human by hand)", () => {
  // The keeper ALWAYS names its channels; the unbounded path is what a human
  // running `node reclaim.mjs withdraw` gets, and it must keep working.
  assert.deepEqual(encodeIds(undefined), { ok: true, ids: null, value: null });
  assert.deepEqual(encodeIds(null), { ok: true, ids: null, value: null });
  assert.equal(parseIds(null), null);
  assert.equal(parseIds(""), null);
});

test("an EMPTY list is an error, never 'act on everything'", () => {
  // The dangerous coercion: a caller that filtered its batch down to nothing
  // must not have that read as "no selection" and blow away every channel.
  assert.equal(encodeIds([]).ok, false);
});

test("rejects a batch larger than the per-call ceiling", () => {
  const many = Array.from({ length: MAX_IDS + 1 }, (_, i) => `c${i}`);
  assert.equal(encodeIds(many).ok, false);
  assert.equal(encodeIds(many.slice(0, MAX_IDS)).ok, true);
});

test("rejects a malformed id rather than silently dropping it", () => {
  // Dropping it would shrink the batch without telling the caller, and the
  // caller is the thing counting transactions.
  assert.equal(encodeIds(["c1", "a b"]).ok, false);
  assert.equal(encodeIds(["c1", 7]).ok, false);
  assert.equal(encodeIds("c1").ok, false, "a bare string is not a list");
});

test("de-duplicates so one channel cannot consume two of the cap's slots", () => {
  assert.deepEqual(encodeIds(["c1", "c1", "c2"]).ids, ["c1", "c2"]);
});
