return {
 providers = {
  openrouter = {discovery='static',base_url='https://openrouter.ai/api/v1',api_kind='openai_compatible',auth_env='OPENROUTER_API_KEY',source='openrouter',tier='partner'},
  openrouter_market = {discovery='marketplace',discovery_id='openrouter_market',base_url='https://openrouter.ai/api/v1',api_kind='openai_compatible',auth_env='OPENROUTER_API_KEY',source='openrouter',tier='marketplace',market_price_cap={input=1000,output=1000}},
 },
 models = {
  ['curated-model'] = {served_by={{provider='openrouter',provider_model_id='vendor/curated-model'}},capabilities={context=128000,supports_tools=true,supports_json_mode=true}},
 },
 profiles={default={scorer={'zero'}}},
 policy_envelope={'and',{'meets_req'},{'not',{'is','disabled'}}},
}
