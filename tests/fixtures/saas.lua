return {
 providers = {
  openai = {discovery='static', base_url='http://provider.test', api_kind='openai_compatible', auth_env='OPENAI_API_KEY', tier='partner'},
  anthropic = {discovery='static', base_url='http://provider.test', api_kind='anthropic', auth_env='ANTHROPIC_API_KEY', tier='partner'},
  platform_only = {discovery='static', base_url='http://provider.test', api_kind='openai_compatible', tier='partner'},
 },
 models = {
  primary = {served_by={{provider='openai'}}, capabilities={context=128000,supports_tools=true,supports_json_mode=true}},
  backup = {served_by={{provider='anthropic'}}, capabilities={context=128000,supports_tools=true,supports_json_mode=true}},
  basic = {served_by={{provider='openai'}}, capabilities={context=128000,supports_tools=false,supports_json_mode=false}},
  forbidden = {served_by={{provider='platform_only'}}, capabilities={context=128000}},
 },
 profiles = {default={scorer={'zero'}}},
 policy_envelope = {'and', {'meets_req'}, {'not', {'is','disabled'}}},
}
