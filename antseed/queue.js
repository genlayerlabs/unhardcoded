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

function createQueue(budgetMs, now = Date.now) {
  let chain = Promise.resolve();

  // Run `fn` after every previously-queued mutation. If this call has been
  // waiting longer than the budget by the time its turn arrives, `onTimeout` is
  // invoked INSTEAD and `fn` never runs at all — that ordering is the whole
  // point, so the rejection is a proof of non-execution rather than a guess.
  return function serialize(fn, onTimeout) {
    const queuedAt = now();
    const next = chain.then(gate, gate);
    // Keep the chain alive across a rejection: one failed mutation must not
    // wedge every later one.
    chain = next.catch(() => {});
    return next;

    function gate() {
      const waited = now() - queuedAt;
      if (waited > budgetMs) return onTimeout(waited);
      return fn();
    }
  };
}

module.exports = { createQueue };
