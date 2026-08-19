'use strict';

// Keep the dashboard wallet API insulated from CLI syntax drift. Since
// @antseed/cli 0.1.137, `buyer deposit` is the QR/watch flow; the legacy direct
// on-chain operation moved to `buyer deposit --onchain <amount>`. The control
// endpoint already receives funded-wallet amounts and must retain that exact
// transaction semantics.
function walletCommandArgs(verb, amount) {
  if (verb === 'deposit') return ['buyer', 'deposit', '--onchain', amount];
  if (verb === 'withdraw') return ['buyer', 'withdraw', amount];
  throw new Error(`unsupported wallet command: ${verb}`);
}

module.exports = { walletCommandArgs };
