// The control server's mutation queue. Its wait bound is what makes the
// sidecar's worst RESPONSE time finite, and a finite server budget is the
// precondition for the router's keeper picking a client timeout that strictly
// exceeds it. Without that, a deposit queued behind a long reclaim phase times
// out on the caller while still executing here — real USDC moves and the ledger
// records nothing.
//
// Note the bound has to be armed at ENQUEUE. Checking elapsed time at dequeue
// bounds only which requests are ADMITTED, not when the caller hears back: the
// request still sits behind the whole 240s phase and is refused at the end of
// it, long after the caller gave up and wrote down an outcome it could not rule
// out. See the "ANSWERED ON TIME" test.
//
// Run: node --test antseed/queue.test.js
"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { createQueue } = require("./queue.js");

// A controllable clock AND timer queue, so the tests assert on the BOUND rather
// than on wall time. `advance` moves the clock and fires any timer due by then,
// which is what lets a test prove the caller is answered while a long mutation
// still holds the queue.
function fakeClock(start = 0) {
  let t = start;
  let seq = 0;
  const timers = new Map();
  return {
    opts: {
      now: () => t,
      setTimer: (fn, ms) => { const id = ++seq; timers.set(id, { at: t + ms, fn }); return id; },
      clearTimer: (id) => { timers.delete(id); },
    },
    advance(ms) {
      t += ms;
      for (const [id, timer] of [...timers]) {
        if (timer.at <= t) { timers.delete(id); timer.fn(); }
      }
    },
    pending: () => timers.size,
  };
}

// Let queued microtasks run, so the chain advances between assertions.
const drain = () => new Promise((res) => setImmediate(res));

test("mutations run one at a time, in order", async () => {
  const serialize = createQueue(1000, fakeClock().opts);
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

test("a queued request is ANSWERED ON TIME while a long mutation still holds the queue", async () => {
  // THE load-bearing property, and the one an at-dequeue check does not give
  // you. A /deposit queued behind a 240s reclaim phase must be refused at the
  // 30s budget — not held for 240s and refused then. Held that long, the
  // caller's own timeout fires first and it records an outcome it cannot rule
  // out (`unknown`: counts as spent, burns a cap slot, feeds the hard-halt
  // breaker) for a request that never ran. Answering on time is what makes the
  // server's published budget true.
  const clock = fakeClock();
  const serialize = createQueue(30000, clock.opts);
  let ran = 0;

  // A reclaim phase that holds the queue far longer than the queue budget.
  let releaseFirst;
  const first = serialize(() => new Promise((res) => { releaseFirst = res; }),
                          () => assert.fail("first should not time out"));
  const second = serialize(() => { ran += 1; return "ran"; },
                           (waited) => ({ rejected: true, waited }));
  await drain();

  clock.advance(30001);                        // the budget elapses...
  assert.deepEqual(await second, { rejected: true, waited: 30001 },
                   "the caller must be answered without waiting for the queue");
  assert.equal(ran, 0, "the CLI must never be invoked for a refused request");

  // ...and the long mutation is still running, untouched.
  releaseFirst("slow");
  assert.equal(await first, "slow");
  await drain();
  assert.equal(ran, 0, "its turn arriving later must NOT run it after all");
});

test("a request served within the budget runs normally", async () => {
  const clock = fakeClock();
  const serialize = createQueue(30000, clock.opts);
  let releaseFirst;
  const first = serialize(() => new Promise((res) => { releaseFirst = res; }), () => {});
  const second = serialize(() => "ran", () => assert.fail("within budget"));
  await drain();
  clock.advance(29999);
  releaseFirst("slow");
  await first;
  assert.equal(await second, "ran");
  assert.equal(clock.pending(), 0, "the timer must be cleared once admitted");
});

test("a mutation that outlives the budget is not cancelled mid-flight", async () => {
  // The budget bounds the WAIT, not the execution: a deposit admitted in time
  // must be allowed to finish, however long the CLI takes.
  const clock = fakeClock();
  const serialize = createQueue(30000, clock.opts);
  let release;
  const only = serialize(() => new Promise((res) => { release = res; }),
                         () => assert.fail("an admitted mutation must not be refused"));
  await drain();
  clock.advance(600000);
  release("finished after 10 minutes");
  assert.equal(await only, "finished after 10 minutes");
});

test("a failed mutation does not wedge the ones behind it", async () => {
  // The chain is kept alive across a rejection; otherwise one CLI blow-up would
  // permanently break every later deposit and reclaim.
  const serialize = createQueue(1000, fakeClock().opts);
  await assert.rejects(serialize(() => Promise.reject(new Error("cli blew up")),
                                 () => {}));
  assert.equal(await serialize(() => "still works", () => {}), "still works");
});

test("the queue is bounded in TIME, not in depth", async () => {
  // Depth is a poor proxy: these ops have budgets an order of magnitude apart,
  // so "3 ahead of me" says nothing useful about how long the wait will be.
  // Many fast mutations must all succeed.
  const clock = fakeClock();
  const serialize = createQueue(30000, clock.opts);
  const results = await Promise.all(
    Array.from({ length: 50 }, (_, i) =>
      serialize(() => i, () => assert.fail("no wall time passed"))));
  assert.deepEqual(results, Array.from({ length: 50 }, (_, i) => i));
});
