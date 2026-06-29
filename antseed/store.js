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

const str = (v) => (v === null || v === undefined) ? null : String(v);

function buyerStatusRow(d, pid) {
  return [pid, str(d.pinnedPeerId), str(d.depositsAvailable),
          str(d.depositsReserved), str(d.walletAddress),
          str(d.connectionState), Date.now()];
}

module.exports = { UPSERT_BUYER_STATUS, buyerStatusRow };
