"""SaaS connection metadata derived from the actual dataplane catalog.

This is not a second provider registry. Credentials and wire behavior remain
owned by catalog declarations and providers.py's existing adapters/sources.
"""
import os
import re

LABELS = {'openai': 'OpenAI', 'openrouter': 'OpenRouter', 'anthropic': 'Anthropic',
          'gemini': 'Gemini', 'bedrock': 'Amazon Bedrock', 'antseed': 'AntSeed',
          'io_net': 'IO.net', 'openai_codex': 'ChatGPT / Codex', 'ollama': 'Ollama'}


def label(pid):
    base = pid.removesuffix('_market')
    return LABELS.get(base, base.replace('_', ' ').title())


def auth_env(provider):
    return provider.get('auth_env') or (provider.get('auth') or {}).get('env')


def shared_provider_ids(catalog):
    """Explicit operator opt-in, independent of tenant-supplied credentials."""
    exposed = {s.strip() for s in os.getenv('SAAS_SHARED_PROVIDERS', '').split(',') if s.strip()}
    return exposed.intersection((catalog.get('providers') or {}).keys())


def credential_names(catalog):
    return {name for p in (catalog.get('providers') or {}).values()
            if isinstance(p, dict) and (name := auth_env(p))
            and re.fullmatch(r'[A-Z][A-Z0-9_]{0,79}', name)}


def connections(host):
    catalog = host.catalog()
    exposed = shared_provider_ids(catalog)
    groups = {}
    for pid, p in (catalog.get('providers') or {}).items():
        key = auth_env(p)
        group = p.get('source') or pid.removesuffix('_market')
        mode = ('aws' if p.get('api_kind') == 'bedrock' else
                'buyer' if str(p.get('discovery_id', '')).startswith('antseed') else
                'key' if key else 'managed')
        if mode in ('aws', 'buyer'):
            key = None
        group_key = (group, key, mode)
        item = groups.setdefault(group_key, {'id': group, 'label': label(group),
            'mode': mode, 'auth_env': key, 'providers': [], 'shared_providers': [],
            'connected': False})
        item['providers'].append(pid)
        if pid in exposed:
            item['shared_providers'].append(pid)
        if pid in (host._tenant_allowed or set()):
            item['connected'] = True
    return list(groups.values())
