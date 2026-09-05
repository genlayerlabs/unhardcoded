// The pre-broadcast classifier. This is the piece that decides whether a failed
// `antseed buyer deposit` cost the router's daily cap a slot or nothing at all —
// and, much more importantly, the piece that must NEVER call a run `failed` when
// that run could have put a transaction on Base mainnet.
//
// The fixtures below are the REAL prod output, byte for byte, from the
// wallet_ops row written the first time the keeper was armed. Keep them that
// way: the whole point of this file is that the classifier is pinned against
// what @antseed/cli actually prints, not against what it might reasonably print.
//
// Run: node --test antseed/broadcast.test.js
"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const {
  classifyCliFailure, ONCHAIN_STEP_MARKERS, PRE_RPC_SIGNATURES, SPAWN_ERRNOS,
} = require("./broadcast.js");

// wallet_ops id=4 and id=6, pid=antseed, op=topup, amount 5, outcome `unknown`.
// ora writes to stderr, so this is the whole of it.
const PROD_403_STDERR = String.raw`- Depositing USDC into deposits contract...
✖ Deposit failed: server response 403 Forbidden (request={  }, response={  }, error=null, info={ "requestUrl": "https://base.publicnode.com", "responseBody": "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32602,\"message\":\"Archive requests require a personal token. Get one at: https://www.allnodes.com/publicnode\"},\"id\":8}\n", "responseStatus": "403 Forbidden" }, code=SERVER_ERROR, version=6.16.0)
`;

// console.log(chalk.dim(...)) — printed before the deposits client is built.
const PROD_403_STDOUT = `Wallet: 0x32485F22a04C1054a4292b152f02363e3849f93F
Amount: 5 USDC (5000000 base units)
`;

const HASH = "0x" + "ab12cd34".repeat(8);   // 0x + 64 hex

function fail(over) {
  return Object.assign({ code: 1, killed: false, stdout: "", stderr: "" }, over);
}

// ---- the incident, and why it is NOT narrowable -------------------------

test("the exact prod 403 stays ATTEMPTED — the on-chain step had started", () => {
  // The deposit step runs two broadcasts (approve, then deposit) and polls for a
  // receipt after each. @antseed/cli throws away the TransactionResponse on the
  // post-broadcast path, so a 403 while polling prints exactly this. There is no
  // hash to look for and no way to tell the two apart, so `unknown` is the only
  // answer that cannot lose money.
  const v = classifyCliFailure(fail({ stdout: PROD_403_STDOUT, stderr: PROD_403_STDERR }));
  assert.equal(v.attempted, true);
  assert.match(v.why, /on-chain step/);
});

test("the same 403 carrying a transaction hash is ATTEMPTED too", () => {
  const v = classifyCliFailure(fail({
    stdout: PROD_403_STDOUT + `Transaction: ${HASH}\n`,
    stderr: PROD_403_STDERR,
  }));
  assert.equal(v.attempted, true);
});

test("a transaction hash alone forces ATTEMPTED, whatever else the output says", () => {
  // Even stripped of every step marker and dressed up as a pre-RPC failure.
  for (const sig of PRE_RPC_SIGNATURES) {
    const v = classifyCliFailure(fail({ stderr: `${sig}\nsaw ${HASH}` }));
    assert.equal(v.attempted, true, `${sig} + a tx hash must stay attempted`);
  }
});

test("a bare 64-hex run counts as a hash — the regex over-matches on purpose", () => {
  const bare = "ab12cd34".repeat(8);
  assert.equal(classifyCliFailure(fail({ stderr: `oops ${bare}` })).attempted, true);
  // ...and so does a longer hex blob that merely contains one.
  assert.equal(classifyCliFailure(fail({ stderr: "0x" + "f".repeat(130) })).attempted, true);
});

test("every on-chain step marker forces ATTEMPTED on its own", () => {
  for (const m of ONCHAIN_STEP_MARKERS) {
    const v = classifyCliFailure(fail({ stderr: `✖ ${m} something went wrong` }));
    assert.equal(v.attempted, true, `${m} must force attempted`);
  }
});

test("wording that only exists once a transaction exists forces ATTEMPTED", () => {
  for (const s of ["nonce too low", "already known", "replacement transaction underpriced",
                   "Transaction was dropped or replaced", "execution reverted",
                   "eth_sendRawTransaction failed"]) {
    assert.equal(classifyCliFailure(fail({ stderr: s })).attempted, true, s);
  }
});

// ---- the default must not be inverted -----------------------------------

test("a killed CLI is ATTEMPTED even with an otherwise pre-RPC-looking output", () => {
  const v = classifyCliFailure(fail({ killed: true, stderr: "Cannot find module './x.js'" }));
  assert.equal(v.attempted, true);
  assert.match(v.why, /killed/);
});

test("an UNRECOGNISED non-zero exit stays ATTEMPTED", () => {
  // The load-bearing default. A failure shape this module has never seen is not
  // evidence that nothing happened — including the shapes that look reassuringly
  // transport-ish.
  for (const stderr of ["something exploded", "ECONNRESET", "socket hang up",
                        "Error: connect ETIMEDOUT 1.2.3.4:443", "panic: nope",
                        "TypeError: undefined is not a function"]) {
    const v = classifyCliFailure(fail({ stderr }));
    assert.equal(v.attempted, true, `${stderr} must stay attempted`);
    assert.match(v.why, /unrecognised/);
  }
});

test("a SILENT non-zero exit stays ATTEMPTED — silence is not evidence", () => {
  assert.equal(classifyCliFailure(fail({ code: 1 })).attempted, true);
  assert.equal(classifyCliFailure(fail({ code: 127 })).attempted, true);
});

test("stdout the module does not recognise keeps a run ATTEMPTED", () => {
  // `buyer deposit` prints its Wallet:/Amount: preamble before the deposits
  // client is constructed, so ANY stdout means the run got past the point every
  // pre-RPC signature describes. A signature appearing anyway is a contradiction
  // we decline to resolve in the optimistic direction.
  for (const sig of PRE_RPC_SIGNATURES) {
    const v = classifyCliFailure(fail({ stdout: PROD_403_STDOUT, stderr: sig }));
    assert.equal(v.attempted, true, `${sig} alongside a preamble must stay attempted`);
  }
});

// ---- what CAN be proved -------------------------------------------------

test("a CLI that never spawned is NOT attempted", () => {
  // execFile reports these on the error object, not in any stream: structural
  // evidence, not prose. A process that never started cannot have signed.
  for (const code of SPAWN_ERRNOS) {
    const v = classifyCliFailure(fail({ code }));
    assert.equal(v.attempted, false, `${code} must be provably pre-broadcast`);
    assert.match(v.why, /could not start/);
  }
});

test("a spawn errno with output on stderr is NOT trusted", () => {
  // If something wrote to stderr the process did run, whatever the errno says.
  assert.equal(
    classifyCliFailure(fail({ code: "ENOENT", stderr: "Deposit failed: boom" })).attempted,
    true);
});

test("the CLI's own pre-RPC guards are NOT attempted", () => {
  // control.js's validAmount rejects this upstream so it should be unreachable,
  // but it is the CLI's own proof that it exited before touching an RPC.
  const v = classifyCliFailure(fail({ stderr: "Error: Amount must be a positive number.\n" }));
  assert.equal(v.attempted, false);
  assert.match(v.why, /before any RPC/);
});

test("a module graph that would not load is NOT attempted", () => {
  // The realistic one: a sidecar image missing a file, which is exactly the
  // class the keeper's halt message calls "a configuration fault, not a chain one".
  for (const stderr of ["Error: Cannot find module './ids.js'",
                        "code: 'ERR_MODULE_NOT_FOUND'",
                        "Error [ERR_MODULE_NOT_FOUND]: Cannot find package"]) {
    assert.equal(classifyCliFailure(fail({ stderr })).attempted, false, stderr);
  }
});

test("every verdict carries a why, so wallet_ops can explain itself", () => {
  for (const r of [fail({ stdout: PROD_403_STDOUT, stderr: PROD_403_STDERR }),
                   fail({ killed: true }), fail({ code: "ENOENT" }),
                   fail({ stderr: "Cannot find module 'x'" }), fail({})]) {
    const v = classifyCliFailure(r);
    assert.equal(typeof v.attempted, "boolean");
    assert.ok(typeof v.why === "string" && v.why.length > 0);
  }
});

test("a missing/garbage result object is ATTEMPTED", () => {
  for (const r of [undefined, null, {}, { stdout: null, stderr: undefined }]) {
    assert.equal(classifyCliFailure(r).attempted, true);
  }
});
