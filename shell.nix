# shell.nix — dev/run environment for the unhardcoded host.
#   nix-shell            # runtime deps
#   nix-shell --run 'python -m pytest tests -q'   # with test deps
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: with ps; [
      lupa httpx fastapi uvicorn pydantic
      psycopg psycopg-pool
      pytest pytest-asyncio
    ]))
  ];
}
