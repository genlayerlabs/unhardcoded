# Forward provider requests through a relay

For an operator-managed static provider, set its `base_url` in the router's Lua
catalog to the relay's origin and API prefix. Keep its existing `api_kind`, model
mappings, and auth settings. For example, set
`base_url = "http://127.0.0.1:8765/api/v1"` in the provider entry. Use an
address reachable from the router process.

| Adapter | Example relay `base_url` | POST path at the relay |
| --- | --- | --- |
| OpenAI-compatible | `http://127.0.0.1:8765/api/v1` | `/api/v1/chat/completions` |
| Anthropic | `http://127.0.0.1:8765/v1` | `/v1/messages` |
| Google | `http://127.0.0.1:8765/v1beta` | `/v1beta/models/<model>:generateContent` or `/v1beta/models/<model>:streamGenerateContent?alt=sse` |

Configure the relay with the upstream destination. It receives the adapter's
normal request body and credentials (for example, `Authorization`, `x-api-key`
or `x-goog-api-key`). Use a relay you trust with those credentials. The relay
should return the upstream status, response body, and content type, and flush
server-sent events as they arrive.

Connection errors follow the route's retry and fallback policy. Every retry of
the configured static provider uses the relay. Direct service requires a
separately configured route. The router cancels active chat and Responses calls
when their request deadline expires or when its HTTP client disconnects. A
proxy in front of the router must close its router request to propagate a
disconnect. The bundled ingress may drain an open stream briefly to capture
usage before closing its router request. The relay controls cancellation of its
own upstream connection.

This configuration applies to static HTTP endpoints. Marketplace offers can
supply their own seller endpoints, and adapters with fixed destinations have
their own configuration.
