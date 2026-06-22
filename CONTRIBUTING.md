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

## Running the tests

```bash
nix-shell -p 'python3.withPackages(ps: with ps; [lupa httpx fastapi uvicorn pydantic pytest pytest-asyncio])' \
    --run 'python -m pytest tests -q'
```

The suite boots a real host with mocked provider responses — only the outbound
HTTP to upstream providers is mocked. Please keep tests green and add coverage
for new behavior.

## Pull requests

- Branch off `main` and open a PR; `main` is protected and merges go through review.
- Keep changes focused and explain the "why" in the description.
- Match the style and structure of the surrounding code.
- Never commit secrets. `.env`, `.env.secrets` and `secrets/` are gitignored —
  keep it that way. Use `.env.example` as the template.

## Security

Please do not file public issues for security problems — see
[`SECURITY.md`](./SECURITY.md).
