// Serialization + queue admission for the control server's wallet mutations.
//
// Extracted from control.js for the same reason amount.js, ids.js and db.js
// were: control.js pulls in `pg`, which only exists inside the sidecar image, so
// anything left in that file cannot be unit-tested from the repo. This module
// has NO dependencies, so queue.test.js runs wherever the suite runs.
//
// WHY THE QUEUE IS BOUNDED. Two concurrent deposits would race the buyer's
// sqlite store and its nonce, so mutations run one at a time. The queue used to
// be unbounded, which made the SERVER's worst case unbounded — and a caller
// cannot choose a timeout that strictly exceeds an unbounded budget. In practice
// a dashboard deposit or a 240s reclaim phase queued ahead guaranteed a client
// timeout on a request the sidecar would still go on to execute: the caller gave
// up, the CLI spent, and nothing in the ledger said so.
//
// The bound is IN TIME, not in depth. What hurts a caller is not how many
// requests are ahead of it but how long it waits, and depth is a poor proxy when
// the ops have budgets an order of magnitude apart. A request rejected while
// still queued provably did not spend, which is why the rejection is safe to
// report as `attempted: false` — the caller may retry it freely.
"use strict";

// `budgetMs` bounds how long a caller waits BEFORE its mutation starts. The
// timer is armed at ENQUEUE, not checked at dequeue: checking at dequeue bounds
// admission but not RESPONSE LATENCY, so a request queued behind a 240s reclaim
// phase would still sit there for the whole 240s and only then be refused. The
// caller's own timeout fires long before that, and it then records an outcome it
// cannot rule out (`unknown` — counts as spent, burns a cap slot, feeds the
// hard-halt breaker) for a request that in fact never ran. Answering on time is
// what makes the published per-endpoint budget true.
//
// `opts.now` / `opts.setTimer` / `opts.clearTimer` exist so the tests can drive
// the clock; production uses the real ones.
function createQueue(budgetMs, opts = {}) {
  const now = opts.now || Date.now;
  const setTimer = opts.setTimer || ((fn, ms) => {
    const t = setTimeout(fn, ms);
    if (t && typeof t.unref === "function") t.unref();
    return t;
  });
  const clearTimer = opts.clearTimer || clearTimeout;
  let chain = Promise.resolve();

  // Run `fn` after every previously-queued mutation. If the budget elapses
  // first, `onTimeout` answers the caller and `fn` NEVER RUNS — not when the
  // timer fires and not when its turn later arrives. That ordering is the whole
  // point: the refusal is a proof of non-execution, not a guess, which is why
  // the caller may safely record it as "nothing was attempted".
  return function serialize(fn, onTimeout) {
    const queuedAt = now();
    let settled = false;
    let resolveOuter;
    let rejectOuter;
    const outer = new Promise((res, rej) => { resolveOuter = res; rejectOuter = rej; });

    const timer = setTimer(() => {
      if (settled) return;
      settled = true;                       // the slot is dead; gate() skips fn
      try {
        resolveOuter(onTimeout(now() - queuedAt));
      } catch (e) {
        rejectOuter(e);
      }
    }, budgetMs);

    // Keep the chain alive across a rejection: one failed mutation must not
    // wedge every later one.
    chain = chain.then(gate, gate).catch(() => {});
    return outer;

    function gate() {
      if (settled) return undefined;        // already refused — do not execute
      settled = true;
      clearTimer(timer);
      // The chain must WAIT for fn (that is the serialization), while the
      // caller's promise settles with fn's own result.
      return Promise.resolve().then(fn).then(resolveOuter, rejectOuter);
    }
  };
}

module.exports = { createQueue };
