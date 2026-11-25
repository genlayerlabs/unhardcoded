# Route provider calls through ESLA

Set `ESLA_ROUTES` in the router process environment and restart the router. The
JSON object is keyed by catalog provider ID. Each entry identifies the agreement
and the complete caller sidecar URL, including its named route prefix.

```bash
export ESLA_ROUTES='{
  "comput3": {
    "agreement": "0x00000000000000000000000000000000000e51a1",
    "sidecar_url": "http://127.0.0.1:28473/bonded"
  }
}'
```

In the ESLA SDK's `caller -routes` file, configure the same agreement and
prefix. The sidecar owns the agreement, signing key, and provider proxy
address. Verify the two configurations agree when provisioning the route.

```json
[
  {
    "name": "bonded",
    "agreement": "agreement.json",
    "provider": "https://provider-proxy.example"
  }
]
```

Run `esla caller -routes routes.json -key client.key -db caller.db -listen
127.0.0.1:28473`. Restrict access to the listener to the router. The sidecar
receives request bodies and provider credentials and signs on the router's
behalf. The router keeps the adapter's path and query string. For example,
`/v1/chat/completions` becomes `/bonded/v1/chat/completions` at the caller
sidecar, which forwards it to the provider proxy.

An empty or unset `ESLA_ROUTES` uses direct calls. Providers absent from the map
also use their regular routes. A malformed map prevents startup. Use mapped
routes for fixed, operator-managed upstreams with OpenAI-compatible, Anthropic,
or Google adapters. Attempts that require a different transport (such as
Bedrock, Codex, decision calls, or the synchronous Lua driver), tenant-managed
credentials, or a dynamic seller endpoint fail closed for mapped providers.

A connection failure to the sidecar is a provider network error. Timeouts and
HTTP errors follow the usual provider handling. The policy can retry via the
sidecar or choose a different provider. A mapped attempt never retries
directly. Router deadlines, and disconnects that reach its chat or Responses
endpoint, stop the router's HTTP request, including while it waits for the
first response byte. The authenticated ingress may keep the router connection
open to drain a stream for usage accounting. The caller sidecar may also keep
reading an abandoned response to settle the call, bounded by its own
`-max-call` setting.

Router observations and ESLA windows measure different things at their
respective boundaries. A sidecar connection failure has a router observation
but no signed request. ESLA's default receipt classifier counts HTTP 429 and
5xx as errors. The router also records errors such as other HTTP 4xx, malformed
responses, and local deadlines. The router measures elapsed time at its end
while ESLA's slow count uses provider-signed latency.

## Local demo and integration test

Build the executable from the ESLA checkout's `sdk/go` directory.

```bash
go build -o bin/esla ./cmd/esla
```

With the router's test dependencies and a dedicated PostgreSQL test database
available, run from the router checkout.

```bash
ESLA_BIN=/absolute/path/to/esla/sdk/go/bin/esla python -m pytest tests/test_esla.py -q
```

The test follows the SDK's `sdk/go/demo/local.sh` topology. It generates
throwaway keys and SQLite stores, starts an SDK caller sidecar, provider proxy,
and fake upstream on loopback, and sends buffered, streaming, and failing calls
through the router's chat API. It compares both sides' dry-run commitments and
their call, error, and slow counts against the router's persisted attempts.
The test uses no chain or provider account. With `ESLA_BIN` unset, pytest skips
the binary-dependent test and still runs configuration, fallback, and
cancellation coverage.
