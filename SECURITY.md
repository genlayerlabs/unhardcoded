# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Report privately via GitHub's
[private vulnerability reporting](https://github.com/genlayerlabs/unhardcoded/security/advisories/new)
(Security → Report a vulnerability). We will acknowledge your report and work
with you on a fix and coordinated disclosure.

## Scope notes

A few areas of this project handle real credentials and funds — please take
extra care:

- **Provider keys & OAuth files** are read from the environment or host mounts
  (`.env.secrets`, `CODEX_AUTH_PATH`). They are never committed and never sent to
  clients. Do not include real keys in issues or PRs.
- **AntSeed wallet** (`ANTSEED_IDENTITY_HEX`) is a private key that controls a
  funded on-chain wallet spending real USDC on Base mainnet. Treat any exposure
  as critical and rotate immediately. This provider is opt-in (off by default).
- **Codex provider** uses an unofficial ChatGPT-subscription auth path and is
  ToS-risky; see `docs/OPENAI-CODEX.md`.
