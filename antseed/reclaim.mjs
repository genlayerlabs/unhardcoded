// AntSeed channel reclaim — recover buyer USDC locked in idle payment channels
// (`depositsReserved`). The buyer `withdraw` CLI only frees `depositsAvailable`
// (escrow); funds reserved inside channels need the two-step on-chain close that
// AntSeed confirmed: requestClose(channelId) -> ~15 min challenge window ->
// withdraw(channelId). No CLI verb exists for it, so we drive @antseed/node's
// ChannelsClient directly, reusing the SAME config + client construction the
// vendored CLI uses for deposit/withdraw (proven in prod). Real money, on-chain:
// every action here is idempotent and guarded per-channel — a revert (e.g. the
// 15-min window not elapsed) is caught and reported, never fund loss.
//
// Usage:  node reclaim.mjs <phase>
//   list           read-only: enumerate channels + on-chain reclaimable (default)
//   request-close  fire requestClose on channels that still hold buyer funds
//   withdraw       withdraw channels whose challenge window has elapsed
//
// Emits ONE JSON object on stdout (control.js parses it). Never throws to the
// shell without a JSON envelope.
import { loadConfig } from '@antseed/cli/dist/config/loader.js';
import {
  loadCryptoContext,
  createChannelsClient,
  openChannelStore,
  formatUsdc,
} from '@antseed/cli/dist/cli/payment-utils.js';

const PHASE = process.argv[2] || 'list';
const DATA_DIR = process.env.ANTSEED_DATA_DIR || '/data';
// Same defaults the vendored CLI uses for deposit/withdraw: a missing config
// file makes loadConfig fall back to createDefaultConfig() (Base mainnet chain
// config), so payments.crypto resolves exactly as it does for those commands.
const CONFIG_PATH = process.env.ANTSEED_CONFIG || '~/.antseed/config.json';

function out(obj) { process.stdout.write(JSON.stringify(obj)); }
function big(x) { try { return BigInt(x ?? '0'); } catch { return 0n; } }

async function main() {
  if (!['list', 'request-close', 'withdraw'].includes(PHASE)) {
    out({ ok: false, error: `unknown phase '${PHASE}'` });
    process.exit(2);
    return;
  }

  const config = await loadConfig(CONFIG_PATH);
  const { wallet, address } = await loadCryptoContext(DATA_DIR);
  const client = createChannelsClient(config);

  const store = openChannelStore(DATA_DIR);
  let sessions;
  try {
    sessions = store.getAllChannelsByBuyer('buyer', address);
  } finally {
    store.close();
  }

  const channels = [];
  let reclaimableTotal = 0n;

  for (const s of sessions) {
    const id = s.sessionId;
    const rec = { id, seller: s.sellerEvmAddr, localStatus: s.status };

    let info;
    try {
      info = await client.getSession(id);
    } catch (e) {
      rec.error = 'getSession: ' + (e && e.message || e);
      channels.push(rec);
      continue;
    }

    const deposit = big(info.deposit);
    const settled = big(info.settled);
    const closeRequestedAt = big(info.closeRequestedAt);
    const remaining = deposit > settled ? deposit - settled : 0n; // buyer's reclaimable upper bound

    // Nothing left on-chain (already withdrawn / never funded) and no pending
    // close — drop it from the view entirely.
    if (deposit === 0n && closeRequestedAt === 0n) continue;

    rec.deposit = formatUsdc(deposit);
    rec.settled = formatUsdc(settled);
    rec.reclaimable = formatUsdc(remaining);
    rec.closeRequested = closeRequestedAt > 0n;
    rec.onchainStatus = Number(info.status);
    reclaimableTotal += remaining;

    if (PHASE === 'request-close') {
      if (closeRequestedAt > 0n) {
        rec.action = 'skip';
        rec.reason = 'close already requested';
      } else if (deposit === 0n) {
        rec.action = 'skip';
        rec.reason = 'no on-chain deposit';
      } else {
        try {
          rec.tx = await client.requestClose(wallet, id);
          rec.action = 'requestClose';
        } catch (e) {
          rec.action = 'error';
          rec.error = 'requestClose: ' + (e && e.message || e);
        }
      }
    } else if (PHASE === 'withdraw') {
      if (closeRequestedAt === 0n) {
        rec.action = 'skip';
        rec.reason = 'close not requested yet';
      } else {
        try {
          rec.tx = await client.withdraw(wallet, id);
          rec.action = 'withdraw';
        } catch (e) {
          // Most common revert here: the ~15-min challenge window hasn't elapsed.
          rec.action = 'error';
          rec.error = 'withdraw: ' + (e && e.message || e);
        }
      }
    }

    channels.push(rec);
  }

  out({
    ok: true,
    phase: PHASE,
    address,
    reclaimableTotal: formatUsdc(reclaimableTotal),
    count: channels.length,
    channels,
  });
}

main().catch((e) => {
  out({ ok: false, phase: PHASE, error: String(e && e.message || e) });
  process.exit(1);
});
