"""Generic + specific step definitions for the unhardcoded user-flow suite."""
import json
import requests
from behave import given, when, then
from environment import auth_headers, jpath, SENTINEL, FREE_POLICY_IR


# ---- request steps --------------------------------------------------------

def _do(context, method, path, *, auth, body=None, stream=False):
    url = context.base_url + path
    headers = {}
    if auth == "consumer":
        headers = auth_headers(context)
    elif auth == "bad":
        headers = {"Authorization": "Bearer not-a-real-token",
                   "Content-Type": "application/json"}
    elif auth == "none":
        headers = {"Content-Type": "application/json"}
    # auth == "admin" -> dashboard NO_AUTH, no headers needed
    context.resp = requests.request(
        method, url, headers=headers,
        json=body if body is not None else None,
        timeout=200, stream=stream)
    context.resp_text = context.resp.text  # NB: context.text is reserved by behave
    try:
        context.json = context.resp.json()
    except Exception:
        context.json = None


@given('the stack is healthy')
def step_healthy(context):
    r = requests.get(context.base_url + "/healthz", timeout=10)
    assert r.status_code == 200, r.status_code
    assert r.json().get("ok") is True


@given('I have a caller token')
def step_have_token(context):
    assert getattr(context, "caller_token", None), "no caller token minted"


@when('I GET "{path}" as {auth}')
def step_get(context, path, auth):
    _do(context, "GET", path, auth=auth)


# NOTE: behave puts a step's docstring (the body) in context.text BEFORE the step
# runs; we read it here, then _do() overwrites context.text with the response body.
@when('I POST "{path}" as {auth} with json')
def step_post_json(context, path, auth):
    body = json.loads(context.text)
    _do(context, "POST", path, auth=auth, body=body)


@when('I create a consumer key for "{consumer}"')
def step_create_key(context, consumer):
    _do(context, "POST", "/dashboard/api/keys", auth="admin", body={"consumer": consumer})
    assert context.resp.status_code == 200, context.resp_text[:200]
    context.created_consumer = consumer
    context.created_key = context.json["api_key"]
    context.created_prefix = context.json["sha256_prefix"]
    context.caller_token = context.created_key  # subsequent "as consumer" uses it (scenario-scoped)


@when('I revoke the created key')
def step_revoke_created(context):
    _do(context, "POST", "/dashboard/api/keys/revoke", auth="admin",
        body={"consumer": context.created_consumer, "sha256_prefix": context.created_prefix})


@when('I run the flow1 ensemble (retry on flake)')
def step_flow1(context):
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "fixtures", "flow1.json")) as f:
        body = json.load(f)
    last = None
    for _ in range(5):
        _do(context, "POST", "/v1/chat/completions", auth="consumer", body=body)
        last = context.resp
        if context.resp.status_code == 200 and isinstance(context.json, dict) \
                and context.json.get("object") == "chat.completion":
            return
    raise AssertionError(
        f"flow1 never succeeded in 5 tries; last status {last.status_code}: "
        f"{context.resp_text[:200]}")


@when('I log into the dashboard with my caller key')
def step_dash_login(context):
    _do(context, "POST", "/dashboard/login", auth="none", body={"api_key": context.caller_token})


@then('the file "{path}" exists')
def step_file_exists(context, path):
    import os
    assert os.path.exists(path), f"missing: {path}"


@when('I POST a free chat as consumer')
def step_free_chat(context):
    _do(context, "POST", "/v1/chat/completions", auth="consumer", body={
        "model": "", "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with one short sentence."}],
        "policy_ir": FREE_POLICY_IR,
    })


@when('I POST a free flow as consumer')
def step_free_flow(context):
    flow = ["flow", {
        "q": {"kind": "input"},
        "a": {"kind": "llm", "system": "Be concise.", "policy": FREE_POLICY_IR, "inputs": ["q"]},
        "b": {"kind": "llm", "system": "Refine.", "policy": FREE_POLICY_IR, "inputs": ["a"],
              "template": "Refine: $1"},
        "out": {"kind": "output", "inputs": ["b"]},
    }]
    _do(context, "POST", "/v1/chat/completions", auth="consumer", body={
        "model": "", "max_tokens": 120,
        "messages": [{"role": "user", "content": "Say hello."}],
        "flow_ir": flow,
    })


# ---- assertion steps ------------------------------------------------------

@then('the status is {code:d}')
def step_status(context, code):
    assert context.resp.status_code == code, \
        f"expected {code}, got {context.resp.status_code}: {context.resp_text[:300]}"


@then('the field "{path}" is present')
def step_present(context, path):
    v = jpath(context.json, path)
    assert v is not SENTINEL, f'"{path}" missing in {str(context.json)[:300]}'


@then('the field "{path}" is non-empty')
def step_nonempty(context, path):
    v = jpath(context.json, path)
    assert v is not SENTINEL, f'"{path}" missing'
    assert v not in (None, "", [], {}), f'"{path}" is empty: {v!r}'


@then('the field "{path}" equals "{value}"')
def step_equals_str(context, path, value):
    v = jpath(context.json, path)
    assert str(v) == value, f'"{path}" = {v!r}, expected {value!r}'


@then('the field "{path}" equals {value:d}')
def step_equals_int(context, path, value):
    v = jpath(context.json, path)
    assert v == value, f'"{path}" = {v!r}, expected {value}'


@then('the field "{path}" is a number')
def step_is_number(context, path):
    v = jpath(context.json, path)
    assert isinstance(v, (int, float)) and not isinstance(v, bool), f'"{path}" = {v!r}'


@then('the field "{path}" is at least {value:d}')
def step_at_least(context, path, value):
    v = jpath(context.json, path)
    assert isinstance(v, (int, float)), f'"{path}" not numeric: {v!r}'
    assert v >= value, f'"{path}" = {v}, expected >= {value}'


@then('the field "{path}" contains "{sub}"')
def step_contains(context, path, sub):
    v = jpath(context.json, path)
    assert v is not SENTINEL, f'"{path}" missing'
    assert sub in str(v), f'"{path}" = {v!r} does not contain {sub!r}'


@then('the array "{path}" has at least {n:d} items')
def step_array_len(context, path, n):
    v = jpath(context.json, path)
    assert isinstance(v, list), f'"{path}" is not a list: {type(v)}'
    assert len(v) >= n, f'"{path}" has {len(v)} items, expected >= {n}'


@then('the array "{path}" includes an item where "{key}" equals "{value}"')
def step_array_item(context, path, key, value):
    arr = jpath(context.json, path)
    assert isinstance(arr, list), f'"{path}" not a list'
    for it in arr:
        if str(jpath(it, key)) == value:
            context.matched_item = it
            return
    raise AssertionError(f'no item in "{path}" with {key}=={value!r}; '
                         f'saw {[jpath(i, key) for i in arr][:10]}')


@then('the matched item field "{path}" equals "{value}"')
def step_matched_equals(context, path, value):
    v = jpath(context.matched_item, path)
    assert str(v) == value, f'matched item "{path}" = {v!r}, expected {value!r}'


@then('every item in "{path}" has a "{key}"')
def step_every_has(context, path, key):
    arr = jpath(context.json, path)
    assert isinstance(arr, list), f'"{path}" not a list'
    assert arr, f'"{path}" empty'
    for it in arr:
        assert jpath(it, key) is not SENTINEL, f'item missing {key}: {str(it)[:200]}'


@then('the response text contains "{sub}"')
def step_text_contains(context, sub):
    assert sub in context.resp_text, f'response text missing {sub!r}'
