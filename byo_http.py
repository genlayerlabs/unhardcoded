"""HTTPS egress for tenant-supplied buyer endpoints.

Resolve and pin a public IP on every new request. Preserve TLS hostname/SNI;
never follow redirects or use process HTTP proxies. DNS rebinding cannot turn
the validated hostname into a subsequent private-address connection.
"""
import asyncio
import ipaddress
import socket

import httpx


class PublicHTTPS(httpx.AsyncBaseTransport):
    def __init__(self):
        # Pooling by pinned IP must not reuse a TLS session across two distinct
        # hostnames that resolve to the same address.
        self.inner = httpx.AsyncHTTPTransport(retries=0, limits=httpx.Limits(max_keepalive_connections=0))

    async def handle_async_request(self, request):
        url = request.url
        if url.scheme != 'https' or url.username or url.password or url.fragment:
            raise httpx.ConnectError('Buyer gateway requires a public HTTPS origin', request=request)
        try:
            rows = await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(
                url.host, url.port or 443, type=socket.SOCK_STREAM), 5)
            addresses = [ipaddress.ip_address(row[4][0]) for row in rows]
            if not addresses or any(not ip.is_global for ip in addresses):
                raise ValueError('non-public endpoint')
        except (ValueError, OSError, TimeoutError) as exc:
            raise httpx.ConnectError('Buyer gateway is not a public HTTPS endpoint', request=request) from exc
        pinned = httpx.Request(request.method, url.copy_with(host=str(addresses[0])),
                               headers=request.headers, stream=request.stream,
                               extensions={**request.extensions, 'sni_hostname': url.host})
        return await self.inner.handle_async_request(pinned)

    async def aclose(self):
        await self.inner.aclose()


def buyer_client():
    return httpx.AsyncClient(transport=PublicHTTPS(), follow_redirects=False,
                             trust_env=False, timeout=10)


def is_byo_buyer(request, env_get):
    return bool(env_get('SAAS_TENANT_SCOPE') and env_get('ANTSEED_BYO_TOKEN')
                and request.get('provider_id') == 'antseed')
