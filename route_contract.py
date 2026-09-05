"""Select only an owner-published preference variant, never a caller policy."""
import re


class PreferenceNotAllowed(ValueError):
    pass


def apply_contract(payload, published):
    payload, selected = dict(payload), dict(published)
    preference = payload.pop('routing_preference', None)
    selected['routing_preference'] = preference if isinstance(preference, str) else 'default'
    if preference is not None:
        variants = published.get('preferences') or {}
        if not isinstance(preference, str) or preference not in variants:
            raise PreferenceNotAllowed('This preference is not authorized by the published contract.')
        variant = variants[preference]
        if (not isinstance(variant, dict) or not isinstance(variant.get('policy_ir'), list)
                or not re.fullmatch(r'[0-9a-f]{64}', str(variant.get('policy_id', '')))):
            raise PreferenceNotAllowed('The published preference is unavailable. Republish the contract.')
        selected.update(policy_ir=variant['policy_ir'], policy_id=variant['policy_id'])
    for field in ('policy_ir', 'flow_ir', 'timeout_ms', 'first_token_timeout_ms', 'task_policies'):
        payload.pop(field, None)
    execution = published.get('execution') or {}
    payload.update({key: execution[key] for key in ('timeout_ms', 'first_token_timeout_ms') if key in execution})
    payload['policy_ir'] = selected['policy_ir']
    return payload, selected
