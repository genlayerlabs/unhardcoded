"""HTTPS egress for tenant-supplied buyer endpoints.

Resolve and pin a public IP on every new request. Preserve TLS hostname/SNI;
never follow redirects or use process HTTP proxies. DNS rebinding cannot turn
the validated hostname into a subsequent private-address connection.
"""
import asyncio
import ipaddress
import socket

import httpx

# Prefixes that embed (and route to) an IPv4 address: NAT64 well-known/local-use,
# IPv4-compatible, 6to4. `is_global` judges the IPv6 wrapper, not the target.
_EMBEDDED_V4 = tuple(ipaddress.ip_network(n) for n in
                     ('64:ff9b::/96', '64:ff9b:1::/48', '::/96', '2002::/16'))


def public_address(ip):
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.version == 6 and any(ip in net for net in _EMBEDDED_V4):
        return False
    return ip.is_global


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
            if not addresses or not all(public_address(ip) for ip in addresses):
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
