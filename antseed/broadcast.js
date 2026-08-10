// Did a FAILED buyer-CLI run put a transaction on Base mainnet?
//
// Extracted from control.js for the same reason amount.js and ids.js were:
// control.js pulls in `pg`, which only exists inside the sidecar image, so
// anything left in that file cannot be unit-tested from the repo. This module
// has NO dependencies, so broadcast.test.js runs wherever the suite runs.
//
// WHAT IT DECIDES. control.js publishes `attempted` on every response, and that
// one boolean is the difference between "nothing happened, retry freely" and "a
// transaction may be on Base mainnet right now": the router's wallet keeper
// records the first as `failed` (consumes neither the daily cap nor the
// cooldown) and the second as `unknown` (consumes both). Until now every
// non-zero CLI exit was `attempted: true`, because the exit code alone cannot
// tell the two apart. This module narrows that — but only where the answer is
// PROVABLE, which is a much smaller set than it looks.
//
// WHY IT LIVES HERE AND NOT IN THE KEEPER. The keeper never sees the evidence.
// control.js hands it `(stderr || stdout).slice(0, 600)`: one stream, not both,
// truncated. `@antseed/cli` prints its transaction hash with `console.log`, i.e.
// on STDOUT, while ora writes the spinner and the failure line to STDERR — so on
// the failure path the keeper is handed the stream that cannot contain the hash
// and never sees the one that can. (Prod reclaim rows are already truncated at
// exactly 600 chars, mid-token.) A classifier reading that view would be reading
// a lossy projection of the thing it has to be sure about.
//
// ---------------------------------------------------------------------------
// WHAT CANNOT BE PROVED FROM CLI OUTPUT — read this before widening the rule.
//
// `@antseed/cli@0.1.128`'s `buyer deposit` runs SIX RPC calls inside one ora
// spinner, and TWO of them are broadcasts (an unconditional ERC-20 `approve`,
// then the deposit itself), each followed by a `wait()` receipt poll:
//
//     getTransactionCount -> estimateGas -> sendTransaction(approve)
//       -> approveTx.wait() -> estimateGas -> sendTransaction(deposit)
//       -> tx.wait()
//
// Every one of those can throw ethers' `SERVER_ERROR`. Critically, the CLI's
// deposit path is:
//
//     const tx = await this._sendBuffered(...);   // tx.hash exists here
//     const receipt = await tx.wait();            // <-- throws
//     return receipt.hash;                        // never reached
//
// `tx` is a local and the hash is never attached to the thrown error, and
// `deposit.js` prints only `err.message`. So a run that BROADCAST and then hit a
// 403 while polling for the receipt prints exactly what a run that failed before
// signing prints — no hash, same `Deposit failed: server response 403 ...` line.
// The two are byte-indistinguishable. The prod incident this module was written
// for is that shape, and it therefore STAYS `unknown`. Deciding it `failed` on
// the "no tx hash + transport-level error" heuristic would be precisely the
// optimistic mistake the `attempted` contract exists to prevent: it moves real
// USDC with the ledger recording nothing.
//
// Resolving that class needs evidence from OUTSIDE the CLI's stdio — the wallet
// nonce read before and after the run, or the escrow delta one status cycle
// later. Neither is this module's job.
// ---------------------------------------------------------------------------
//
// SO THE RULE IS AFFIRMATIVE, NOT RESIDUAL. `attempted: false` requires a
// RECOGNISED pre-RPC failure shape. An unrecognised failure stays `attempted:
// true`, which means a future @antseed/cli that changes its wording degrades
// toward the safe answer rather than the expensive one.
"use strict";

// A transaction-hash-shaped token. Deliberately GENEROUS — the `0x` is optional
// and a longer hex run still matches its 64-char prefix — because every extra
// match lands on `attempted: true`, the safe side. A missed hash is the only
// error direction that costs money.
const TX_HASH_RE = /(?:0x)?[0-9a-fA-F]{64}/;

// The CLI's own markers for the step that does the signing and broadcasting.
// Their PRESENCE proves the on-chain step started, which is all we need: from
// there on nothing in the output can rule a broadcast out. Matched case-
// insensitively against both streams combined.
const ONCHAIN_STEP_MARKERS = [
  "depositing usdc",          // ora start text, written to stderr
  "deposit failed:",          // spinner.fail() — the catch around deposit()
  "deposited ",               // spinner.succeed()
  "withdrawing",              // the /withdraw twin
  "withdraw failed:",
  "withdrew ",
];

// Phrases that only exist once a transaction exists. Belt and braces behind the
// hash and the step markers: any one of them forces `attempted: true`.
const BROADCAST_PHRASES = [
  "nonce",
  "already known",
  "replacement transaction",
  "underpriced",
  "dropped or replaced",      // the CLI's own post-broadcast throw
  "execution reverted",
  "transactionhash",
  "transaction hash",
  "receipt",
  "confirmations",
  "eth_sendrawtransaction",
  "broadcast",
];

// execFile could not START the process. Node reports these on the error object
// itself rather than in any stream, so this is structural evidence, not prose:
// a process that never spawned cannot have signed anything.
const SPAWN_ERRNOS = new Set([
  "ENOENT", "EACCES", "EPERM", "ENOTDIR", "ENOEXEC", "E2BIG",
]);

// Failures that happen before the CLI can reach an RPC at all: its own argument
// guard, and a module graph that would not load. Recognised affirmatively and
// only when STDOUT IS EMPTY — `buyer deposit` prints its `Wallet:` / `Amount:`
// preamble to stdout before the deposits client is even constructed, so any
// stdout at all means the run got past the point these signatures describe, and
// a signature appearing anyway is a contradiction we decline to resolve.
const PRE_RPC_SIGNATURES = [
  "error: amount must be a positive number",
  "cannot find module",
  "err_module_not_found",
  "module_not_found",
];

// r is control.js's `run()` result: { code, killed, stdout, stderr }.
// Returns { attempted, why } — `why` is a short phrase for the ledger, so an
// operator reading wallet_ops can see WHICH branch decided their row.
function classifyCliFailure(r) {
  const stdout = String((r && r.stdout) || "");
  const stderr = String((r && r.stderr) || "");
  const code = r && r.code;
  const combined = stdout + "\n" + stderr;
  const hay = combined.toLowerCase();

  // 1. We SIGTERMed it on the timeout. Says nothing at all about whether it had
  //    already broadcast — that is the entire reason control.js answers 504 here.
  if (r && r.killed) {
    return { attempted: true, why: "the CLI was killed on the timeout" };
  }

  // 2. The on-chain step had started. Unconditional: past this marker the output
  //    cannot rule a broadcast out (see the header — the hash is discarded on
  //    the post-broadcast failure path).
  const marker = ONCHAIN_STEP_MARKERS.find((m) => hay.includes(m));
  if (marker) {
    return { attempted: true, why: "the on-chain step had started (" + marker.trim() + ")" };
  }

  // 3. A transaction-hash-shaped token anywhere. Unconditional, whatever else
  //    the output says.
  if (TX_HASH_RE.test(combined)) {
    return { attempted: true, why: "the output carries a transaction-hash-shaped token" };
  }

  // 4. Wording that only exists once a transaction exists.
  const phrase = BROADCAST_PHRASES.find((p) => hay.includes(p));
  if (phrase) {
    return { attempted: true, why: "the output mentions " + JSON.stringify(phrase) };
  }

  // 5. Anything on stdout means the run got past its pre-flight preamble and
  //    into territory this module does not claim to understand.
  if (stdout.trim() !== "") {
    return { attempted: true, why: "the CLI produced stdout we do not recognise" };
  }

  // 6. The process never started.
  if (SPAWN_ERRNOS.has(String(code)) && stderr.trim() === "") {
    return { attempted: false, why: "execFile could not start the CLI (" + code + ")" };
  }

  // 7. A recognised pre-RPC failure, with an empty stdout confirming the run
  //    never reached its preamble.
  const sig = PRE_RPC_SIGNATURES.find((s) => hay.includes(s));
  if (sig) {
    return { attempted: false, why: "the CLI failed before any RPC call (" + JSON.stringify(sig) + ")" };
  }

  // 8. Unrecognised. Silence included: a CLI that exits non-zero saying nothing
  //    is not evidence that it did nothing. The default must never be inverted.
  return { attempted: true, why: "the failure is unrecognised, so a broadcast cannot be ruled out" };
}

module.exports = {
  classifyCliFailure,
  TX_HASH_RE,
  ONCHAIN_STEP_MARKERS,
  BROADCAST_PHRASES,
  PRE_RPC_SIGNATURES,
  SPAWN_ERRNOS,
};
