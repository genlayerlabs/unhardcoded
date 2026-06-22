# Contributing

Thanks for your interest in contributing to **unhardcoded**.

## Getting set up

This repo vendors the Σ_pol core (`genlayerlabs/unhardcoded-engine`) as a git
submodule under `core/`, so clone recursively:

```bash
git clone --recursive https://github.com/genlayerlabs/unhardcoded.git
# or, after a plain clone:
git submodule update --init
```

To bring up the full local stack (router + ingress + dashboard) and make a first
routed call, follow [`SETUP.md`](./SETUP.md) — or point a coding agent at it.

## Running the tests

Install the deps once (a virtualenv keeps them off your system Python):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

**Unit tests** — boot a real host with mocked provider responses (only the
outbound HTTP to upstream providers is mocked):

```bash
python -m pytest tests -q
```

**BDD user-flow suite** (`features/`, behave) — drives the running stack end to
end the way the dashboard does and asserts the rendered data is correct,
including a real headless-browser pass (needs Chrome/Chromium installed). End-to-end
chats route to a $0 path so it's free and repeatable. Needs the stack up (see
`SETUP.md`); the flow catalogue it covers is [`user_flows.json`](./user_flows.json).

```bash
behave
```

*(Nix users: `nix-shell -p ...` with the same packages — plus `chromium
chromedriver` — works as before.)*

Real-money AntSeed scenarios are excluded by default and gated behind
`RUN_ANTSEED_SPEND=1`; the read-only `@antseed` data checks auto-skip when no
funded wallet is present.

Please keep both suites green and add coverage for new behavior. A new user-facing
flow should get a `.feature` scenario.

## Pull requests

- Branch off `main` and open a PR; `main` is protected and merges go through review.
- Keep changes focused and explain the "why" in the description.
- Match the style and structure of the surrounding code.
- **Never commit secrets.** `.env`, `.env.secrets` and `secrets/` are gitignored —
  keep it that way; use `.env.example` as the template.
- For AntSeed / on-chain testing use a **dedicated dev wallet**
  (`./scripts/gen-dev-wallet.sh`), never a production key. `ANTSEED_IDENTITY_HEX`
  is a private key — treat it like a password.

## Security

Please do not file public issues for security problems — see
[`SECURITY.md`](./SECURITY.md).
