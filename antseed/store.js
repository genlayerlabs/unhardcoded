// Shared host-store writes for the antseed sidecar (write-status.js on the poll
// loop and control.js after a wallet op both upsert the buyer status). One place
// for the buyer_status row shape + UPSERT so the two writers can't drift.
//
// Raw buyer-reported fields as columns; the deposits are kept as the strings the
// buyer reports (sources/antseed coerces on read). No interpretation here.
const UPSERT_BUYER_STATUS = `INSERT INTO buyer_status
  (pid, pinned_peer_id, deposits_available, deposits_reserved, wallet_address,
   connection_state, fetched_at)
  VALUES ($1,$2,$3,$4,$5,$6,$7)
  ON CONFLICT (pid) DO UPDATE SET
    pinned_peer_id=EXCLUDED.pinned_peer_id,
    deposits_available=EXCLUDED.deposits_available,
    deposits_reserved=EXCLUDED.deposits_reserved,
    wallet_address=EXCLUDED.wallet_address,
    connection_state=EXCLUDED.connection_state,
    fetched_at=EXCLUDED.fetched_at`;

// node-postgres's `connectionString` only understands a postgres:// URL, but the
// prod secret hands us a libpq KEYWORD/VALUE conninfo (host=… port=… dbname=…
// user=… password='…' sslmode=require) — the form psycopg uses on the router/
// ingress. Detect that and parse it into a pg config object; pass a URL through
// untouched (dev/compose). Without this the sidecar resolves host "base" and
// dies with ENOTFOUND, leaving peer_offers/buyer_status empty.
function pgConfig() {
  const dsn = process.env.DATABASE_URL || "";
  if (!dsn || dsn.includes("://")) return { connectionString: dsn };
  const cfg = {};
  const re = /(\w+)\s*=\s*'((?:[^'\\]|\\.)*)'|(\w+)\s*=\s*(\S+)/g;
  let m;
  while ((m = re.exec(dsn)) !== null) {
    const key = m[1] || m[3];
    const val = m[1] ? m[2].replace(/\\(.)/g, "$1") : m[4];
    if (key === "host") cfg.host = val;
    else if (key === "hostaddr" && !cfg.host) cfg.host = val;
    else if (key === "port") cfg.port = Number(val);
    else if (key === "dbname") cfg.database = val;
    else if (key === "user") cfg.user = val;
    else if (key === "password") cfg.password = val;
    else if (key === "sslmode") cfg.ssl = val === "disable" ? false : { rejectUnauthorized: false };
  }
  return cfg;
}

const str = (v) => (v === null || v === undefined) ? null : String(v);

function buyerStatusRow(d, pid) {
  return [pid, str(d.pinnedPeerId), str(d.depositsAvailable),
          str(d.depositsReserved), str(d.walletAddress),
          str(d.connectionState), Date.now()];
}

module.exports = { UPSERT_BUYER_STATUS, buyerStatusRow, pgConfig };
