"""BYO configuration and request-local views over the existing provider sources.

Never read operator buyer state or AWS entitlements for a tenant connection.
Bounded cache contains only derived discovery data, partitioned by tenant and
credential digest. Rotation/revocation changes the key immediately.
"""
import asyncio
from collections import OrderedDict
import hashlib
import json
import math
import time


_cache = OrderedDict()


def configure(catalog, env, connections):
    allowed = set()
    aws = connections.get('bedrock') or {}
    if aws.get('access_key_id') and aws.get('secret_access_key') and aws.get('region'):
        env.update(AWS_ACCESS_KEY_ID=aws['access_key_id'], AWS_SECRET_ACCESS_KEY=aws['secret_access_key'],
                   AWS_SESSION_TOKEN=aws.get('session_token', ''), BEDROCK_REGION=aws['region'])
        for pid, p in catalog.get('providers', {}).items():
            if p.get('api_kind') == 'bedrock':
                allowed.add(pid)
                # Both native and market names discover THIS account's profiles.
                p.update(discovery='marketplace', discovery_id=pid, aws_region=aws['region'])
    buyer = connections.get('antseed') or {}
    if buyer.get('gateway_url') and buyer.get('token') and 'antseed' in catalog.get('providers', {}):
        allowed.add('antseed')
        env.update(ANTSEED_BYO_URL=buyer['gateway_url'], ANTSEED_BYO_TOKEN=buyer['token'])
        p = catalog['providers']['antseed']
        p.update(base_url=buyer['gateway_url'].rstrip('/') + '/v1',
                 auth={'kind': 'bearer', 'env': 'ANTSEED_BYO_TOKEN'})
    return allowed


class BuyerSnapshot:
    """Read-only source store; no global wallet, pin or account-health fallback."""
    def __init__(self, data):
        self.data = data

    def peer_offers(self, window_ms):
        cutoff = time.time() * 1000 - window_ms
        return [r for r in self.data.get('peer_offers', [])
                if isinstance(r, dict) and isinstance(r.get('observed_at'), (int, float))
                and cutoff <= r['observed_at'] <= time.time() * 1000 + 5000]

    def buyer_status(self, pid):
        return self.data.get('buyer_status') if pid == 'antseed' else None

    def marketplace_route_health(self, *args, **kwargs):
        return {}

    def route_stats(self):
        return {}

    def tool_incapable_routes(self):
        return set()

    def provider_recent_ok(self, *args, **kwargs):
        return []


async def _buyer(host):
    from byo_http import buyer_client
    from sources.antseed import AntSeedSource, STALE_AFTER_S
    async with buyer_client() as client:
        async with client.stream('GET', host._env['ANTSEED_BYO_URL'].rstrip('/') + '/snapshot',
                                 headers={'Authorization': 'Bearer ' + host._env['ANTSEED_BYO_TOKEN']}) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > 4 * 1024 * 1024:
                    raise ValueError('Buyer snapshot too large')
            data = json.loads(body)
    if not isinstance(data, dict) or not isinstance(data.get('peer_offers'), list):
        raise ValueError('Invalid buyer snapshot')
    store = BuyerSnapshot(data)
    status = store.buyer_status('antseed') or {}
    stamp = float(status.get('fetched_at') or 0)
    credits = float(status.get('deposits_available') or 0)
    if not math.isfinite(credits) or not 0 <= time.time() * 1000 - stamp <= STALE_AFTER_S * 1000:
        credits = 0
    source = AntSeedSource(host.catalog(), store=store)
    offers = await asyncio.to_thread(source.offers_sync, 'antseed') if credits > 0 else []
    return {'antseed': offers}, max(0, credits)


async def _aws(host):
    from sources.bedrock import BedrockSource
    source = BedrockSource(host.catalog(), env_get=host._env.get)
    try:
        await source.pricing()
        return {pid: source.offers_sync(pid) for pid in source.provider_ids}, None
    finally:
        if source._client is not None:
            await source._client.aclose()


async def prepare(host):
    configs = getattr(host, '_tenant_connections', {})
    host._connection_errors = {}

    async def one(name, load, ids):
        # Default to no offers before any network I/O: never fall back to the
        # operator source after timeout, malformed data or revoked credentials.
        host._tenant_offers.update({pid: [] for pid in ids})
        digest = hashlib.sha256(json.dumps(configs[name], sort_keys=True).encode()).hexdigest()
        key = (host._tenant_id, name, digest)
        cached = _cache.get(key)
        try:
            if cached and time.monotonic() - cached[0] < (15 if name == 'antseed' else 300):
                offers, credits = cached[1]
            else:
                offers, credits = await asyncio.wait_for(load(host), timeout=35)
                _cache[key] = (time.monotonic(), (offers, credits))
                _cache.move_to_end(key)
                while len(_cache) > 128:
                    _cache.popitem(last=False)
            host._tenant_offers.update(offers)
            if credits is not None:
                host.update_metrics('__credits', 'antseed', {'free_credits_remaining_usd': credits})
        except Exception:
            # Never render exceptions containing a connection URL or credentials.
            host._connection_errors[name] = 'Could not discover models. Check your connection, permissions and funding.'

    tasks = []
    if configs.get('bedrock') and host._env.get('AWS_ACCESS_KEY_ID'):
        ids = [pid for pid, p in host.catalog()['providers'].items() if p.get('api_kind') == 'bedrock']
        tasks.append(one('bedrock', _aws, ids))
    if configs.get('antseed') and host._env.get('ANTSEED_BYO_URL'):
        tasks.append(one('antseed', _buyer, ['antseed']))
    await asyncio.gather(*tasks)
