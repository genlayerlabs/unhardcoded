#!/usr/bin/env sh
# Generate a fresh AntSeed *dev* wallet identity (a secp256k1 / EVM private key)
# plus a control token, and print the two .env lines to paste in.
#
# The sidecar derives the Base-mainnet address from the key. After adding these
# to .env and running `docker compose --profile antseed up -d`, get the address
# to fund with:  docker compose exec antseed antseed buyer balance --json
#
# ⚠️  This is a real private key. Use a DEDICATED DEV WALLET with a tiny balance,
#     never your production wallet. Never commit it.
set -eu

rand_hex() {
  # $1 = number of bytes
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex "$1"
  elif command -v node >/dev/null 2>&1; then
    node -e "console.log(require('crypto').randomBytes($1).toString('hex'))"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c "import secrets;print(secrets.token_hex($1))"
  else
    echo "need one of: openssl, node, python3" >&2
    exit 1
  fi
}

KEY=$(rand_hex 32)
TOKEN=$(rand_hex 16)

cat <<EOF
# --- AntSeed dev wallet (paste into .env) ---
ANTSEED_IDENTITY_HEX=${KEY}
ANTSEED_CONTROL_TOKEN=${TOKEN}

# Next:
#   docker compose --profile antseed up -d --build
#   docker compose exec antseed antseed buyer balance --json   # -> the address to fund
#   send a little USDC + ETH (gas) on Base mainnet to that address,
#   then Deposit it into escrow from the dashboard Catalog (wallet cell).
EOF
