// Buyer status writer (replaces the inline `node -e` + status-antseed.json).
// Reads `antseed buyer status --json` from stdin and UPSERTs the buyer's status
// (session pin + escrow + wallet) into the host store's buyer_status, keyed by
// the buyer pid. sources/antseed.py reads it (_pinned_peer + balances).
//
// Validation: a non-object / non-JSON exits non-zero so entrypoint.sh keeps the
// last good row (no write).
const { Client } = require("pg");
const { pgConfig } = require("./db.js");
const { UPSERT_BUYER_STATUS, buyerStatusRow } = require("./store.js");

const PID = process.env.ANTSEED_BUYER_PID || "antseed";

let d;
try { d = JSON.parse(require("fs").readFileSync(0, "utf8")); }
catch (e) { process.exit(2); }                            // not JSON
if (d === null || typeof d !== "object") process.exit(3); // JSON but not a status

(async () => {
  const client = new Client(pgConfig(process.env.DATABASE_URL));
  try {
    await client.connect();
    await client.query(UPSERT_BUYER_STATUS, buyerStatusRow(d, PID));
  } catch (e) {
    process.stderr.write(`write-status: ${e.message}\n`);
    process.exitCode = 4;                                 // DB error -> keep row
  } finally {
    await client.end().catch(() => {});
  }
})();
