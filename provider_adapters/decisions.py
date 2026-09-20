"""Bounded non-streaming Decisions transport through OpenRouter or AntSeed."""
import asyncio
import json
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

from decision_protocol import MAX_RESPONSE_BYTES, validate_payload, validate_response
from decision_providers.jev import _unique_object
from provider_adapters.common import err, classify_status


async def call_decisions(request, *, env_get, client=None, timeout_s=30,
                         extra_headers=None, token_providers=None):
    from provider_adapters.openai_compatible import (
        _prepare_openai_call, _acquire_peer_capacity, _peer_capacity_error)
    from byo_http import buyer_client, is_byo_buyer

    started = time.monotonic()
    latency = lambda: int((time.monotonic() - started) * 1000)
    try:
        payload = validate_payload(request.get('decision'))
    except (ValueError, TypeError):
        return err('bad_request', 0, latency(), 'Invalid decision request')
    prep, error = _prepare_openai_call(request, env_get, extra_headers or {}, timeout_s, token_providers)
    if error:
        return error
    chat_url, _, headers, timeout = prep
    provider = request.get('provider_id', '')
    offer = request.get('offer') or {}
    parsed = urlsplit(chat_url)
    if provider in {'openrouter', 'openrouter_market'} and parsed.hostname == 'openrouter.ai' and parsed.scheme == 'https':
        url = 'https://openrouter.ai/api/alpha/decisions'
    elif provider.startswith('antseed') and offer.get('peer_id'):
        url = urlunsplit(parsed._replace(path=parsed.path.removesuffix('/chat/completions') + '/systemone'))
    else:
        return err('unsupported_api_kind', 0, latency(), 'Unsupported decisions transport')
    body = {'model': offer.get('wire_model_id') or request['served_model_id'], **payload}
    # A decision has one response, no first token. The liveness budget therefore
    # bounds the complete response and must not activate the chat SSE backend.
    if request.get('first_token_timeout_ms') is not None:
        timeout = min(timeout, request['first_token_timeout_ms'] / 1000)
    slot = None

    async def send(http):
        async with http.stream('POST', url, json=body, headers=headers,
                               timeout=timeout, follow_redirects=False) as response:
            if response.status_code != 200:
                return err(classify_status(response.status_code, ''), response.status_code,
                           latency(), 'Decision provider request failed')
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > MAX_RESPONSE_BYTES:
                    return err('bad_response', 200, latency(), 'Decision response too large')
            try:
                data = validate_response(json.loads(raw, object_pairs_hook=_unique_object), payload)
            except (ValueError, TypeError, AttributeError, RecursionError):
                return err('bad_response', 200, latency(), 'Invalid decision response')
            usage = data['usage']
            return {'ok': True, 'latency_ms': latency(), 'response': {
                'decision': data, 'raw_model': data['model'],
                'tokens_in': usage.get('input_tokens'), 'tokens_out': usage.get('output_tokens', 0),
                'tokens_total': usage.get('total_tokens'), 'cost_reported': usage.get('cost'),
            }}

    try:
        async with asyncio.timeout(timeout):
            slot, gate_error = await _acquire_peer_capacity(request, timeout)
            if gate_error:
                return _peer_capacity_error(str(offer.get('peer_id') or ''),
                                           int(offer.get('max_concurrency') or 0), gate_error, started)
            if is_byo_buyer(request, env_get):
                async with buyer_client() as http:
                    return await send(http)
            if client is not None:
                return await send(client)
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as http:
                return await send(http)
    except (TimeoutError, httpx.TimeoutException):
        return err('timeout', 0, latency(), 'Decision request timed out')
    except httpx.RequestError:
        return err('network_error', 0, latency(), 'Decision provider unavailable')
    finally:
        if slot is not None:
            await slot.release()
