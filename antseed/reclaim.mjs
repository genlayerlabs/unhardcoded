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
// requestClose/withdraw are gated on the buyer's OPERATOR (an AntseedDeposits
// role): the buyer must first assign a wallet to act on its behalf, else every
// close reverts NotAuthorized() even though it owns the channel. We self-assign
// the buyer wallet as its own operator (set-operator phase) — the one-time setup
// the normal deposit/reserve flow never needs.
//
// Usage:  node reclaim.mjs <phase>
//   list           read-only: enumerate channels + on-chain reclaimable + operator (default)
//   set-operator   one-time: assign the buyer wallet as its own deposits operator
//   request-close  fire requestClose on channels that still hold buyer funds
//   withdraw       withdraw channels whose challenge window has elapsed
//
// Emits ONE JSON object on stdout (control.js parses it). Never throws to the
// shell without a JSON envelope.
import { loadConfig } from '@antseed/cli/dist/config/loader.js';
import {
  loadCryptoContext,
  createChannelsClient,
  createDepositsClient,
  requireCryptoConfig,
  openChannelStore,
  formatUsdc,
} from '@antseed/cli/dist/cli/payment-utils.js';

const ZERO_ADDR = '0x0000000000000000000000000000000000000000';
// EIP-712 for AntseedDeposits.setOperator — mirrors @antseed/node's
// makeDepositsDomain + SET_OPERATOR_TYPES (contract-level constants).
const DEPOSITS_DOMAIN_NAME = 'AntseedDeposits';
const SET_OPERATOR_TYPES = { SetOperator: [{ name: 'operator', type: 'address' }, { name: 'nonce', type: 'uint256' }] };

const PHASE = process.argv[2] || 'list';
const DATA_DIR = process.env.ANTSEED_DATA_DIR || '/data';
// Same defaults the vendored CLI uses for deposit/withdraw: a missing config
// file makes loadConfig fall back to createDefaultConfig() (Base mainnet chain
// config), so payments.crypto resolves exactly as it does for those commands.
const CONFIG_PATH = process.env.ANTSEED_CONFIG || '~/.antseed/config.json';

function out(obj) { process.stdout.write(JSON.stringify(obj)); }
function big(x) { try { return BigInt(x ?? '0'); } catch { return 0n; } }

async function main() {
  if (!['list', 'set-operator', 'request-close', 'withdraw'].includes(PHASE)) {
    out({ ok: false, error: `unknown phase '${PHASE}'` });
    process.exit(2);
    return;
  }

  const config = await loadConfig(CONFIG_PATH);
  const { wallet, address } = await loadCryptoContext(DATA_DIR);
  const client = createChannelsClient(config);
  const deposits = createDepositsClient(config);
  const crypto = requireCryptoConfig(config);

  // Current operator (the party authorized to requestClose/withdraw on the
  // buyer's behalf). Unset (zero) → reclaim is blocked until set-operator runs.
  let operator = ZERO_ADDR;
  try { operator = await deposits.getOperator(address); } catch { /* read best-effort */ }
  const operatorIsSelf = String(operator).toLowerCase() === address.toLowerCase();
  const operatorSet = String(operator).toLowerCase() !== ZERO_ADDR.toLowerCase();

  if (PHASE === 'set-operator') {
    // Assign the buyer wallet as its own operator: sign the EIP-712 auth, then
    // submit setOperator. Grants the reclaim role; moves no funds.
    if (operatorIsSelf) {
      out({ ok: true, phase: PHASE, address, operator, operatorIsSelf, action: 'skip', reason: 'already self' });
      return;
    }
    const nonce = await deposits.getOperatorNonce(address);
    const domain = { name: DEPOSITS_DOMAIN_NAME, version: '1', chainId: crypto.evmChainId, verifyingContract: crypto.depositsContractAddress };
    const buyerSig = await wallet.signTypedData(domain, SET_OPERATOR_TYPES, { operator: address, nonce });
    const tx = await deposits.setOperator(wallet, address, address, nonce, buyerSig);
    out({ ok: true, phase: PHASE, address, operator: address, operatorIsSelf: true, action: 'setOperator', tx });
    return;
  }

  const store = openChannelStore(DATA_DIR);
  let sessions;
  try {
    sessions = store.getAllChannelsByBuyer('buyer', address);
  } finally {
    store.close();
  }

  const channels = [];
  let reclaimableTotal = 0n;
  // On-chain channel status (AntseedChannels enum): 0 missing, 1 active,
  // 2 settled, 3 timeout. Only ACTIVE channels still hold reservable buyer
  // funds; settled/timeout channels already returned their remainder to
  // depositsAvailable, so their historical deposit-settled is NOT reclaimable
  // (a requestClose/withdraw there just reverts). Count them for transparency.
  const skipped = { settled: 0, timeout: 0, other: 0 };

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

    const status = Number(info.status);
    if (status !== 1) {
      if (status === 2) skipped.settled++;
      else if (status === 3) skipped.timeout++;
      else skipped.other++;
      continue;
    }

    const deposit = big(info.deposit);
    const settled = big(info.settled);
    const closeRequestedAt = big(info.closeRequestedAt);
    const remaining = deposit > settled ? deposit - settled : 0n; // buyer's reclaimable upper bound

    rec.deposit = formatUsdc(deposit);
    rec.settled = formatUsdc(settled);
    rec.reclaimable = formatUsdc(remaining);
    rec.closeRequested = closeRequestedAt > 0n;
    rec.onchainStatus = status;
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
    operator,           // current on-chain operator (zero = unset)
    operatorSet,        // any operator assigned?
    operatorIsSelf,     // is the buyer wallet its own operator? (reclaim ready)
    reclaimableTotal: formatUsdc(reclaimableTotal),
    count: channels.length,
    skipped, // settled/timeout channels: already returned funds, not reclaimable
    channels,
  });
}

main().catch((e) => {
  out({ ok: false, phase: PHASE, error: String(e && e.message || e) });
  process.exit(1);
});
