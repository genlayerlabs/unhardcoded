// The control server's mutation queue. Its admission bound is what makes the
// sidecar's worst case FINITE, and a finite server budget is the precondition
// for the router's keeper picking a client timeout that strictly exceeds it.
// Without that, a deposit queued behind a long reclaim phase times out on the
// caller while still executing here — real USDC moves and the ledger records
// nothing.
//
// Run: node --test antseed/queue.test.js
"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { createQueue } = require("./queue.js");

// A controllable clock, so the tests assert on the BOUND rather than on wall time.
function fakeClock(start = 0) {
  let t = start;
  return { now: () => t, advance: (ms) => { t += ms; } };
}

test("mutations run one at a time, in order", async () => {
  const serialize = createQueue(1000);
  const order = [];
  const busy = (id) => () => new Promise((res) => setImmediate(() => {
    order.push(id);
    res(id);
  }));
  const fail = () => assert.fail("should not have timed out");
  await Promise.all([serialize(busy(1), fail), serialize(busy(2), fail),
                     serialize(busy(3), fail)]);
  assert.deepEqual(order, [1, 2, 3]);
});

test("a request that waited past the budget is REJECTED WITHOUT RUNNING", async () => {
  // The load-bearing property: the rejection happens when its turn arrives, so
  // it is a proof of non-execution, not a guess. That is what lets the caller
  // record it as "nothing was attempted" and retry freely.
  const clock = fakeClock();
  const serialize = createQueue(30000, clock.now);
  let ran = 0;

  // A long mutation ahead of us — a 240s reclaim phase, say. The clock advances
  // while it holds the queue, which is exactly the real shape.
  const first = serialize(() => { clock.advance(31000); return "slow"; },
                          () => assert.fail("first should not time out"));
  const second = serialize(() => { ran += 1; return "ran"; },
                           (waited) => ({ rejected: true, waited }));

  await first;
  assert.deepEqual(await second, { rejected: true, waited: 31000 });
  assert.equal(ran, 0, "the CLI must never be invoked for a rejected request");
});

test("a request served within the budget runs normally", async () => {
  const clock = fakeClock();
  const serialize = createQueue(30000, clock.now);
  const first = serialize(() => { clock.advance(29999); return "slow"; }, () => {});
  const second = serialize(() => "ran", () => assert.fail("within budget"));
  await first;
  assert.equal(await second, "ran");
});

test("a failed mutation does not wedge the ones behind it", async () => {
  // The chain is kept alive across a rejection; otherwise one CLI blow-up would
  // permanently break every later deposit and reclaim.
  const serialize = createQueue(1000);
  await assert.rejects(serialize(() => Promise.reject(new Error("cli blew up")),
                                 () => {}));
  assert.equal(await serialize(() => "still works", () => {}), "still works");
});

test("the queue is bounded in TIME, not in depth", async () => {
  // Depth is a poor proxy: these ops have budgets an order of magnitude apart,
  // so "3 ahead of me" says nothing useful about how long the wait will be.
  // Many fast mutations must all succeed.
  const clock = fakeClock();
  const serialize = createQueue(30000, clock.now);
  const results = await Promise.all(
    Array.from({ length: 50 }, (_, i) =>
      serialize(() => i, () => assert.fail("no wall time passed"))));
  assert.deepEqual(results, Array.from({ length: 50 }, (_, i) => i));
});
