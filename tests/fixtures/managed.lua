return {
 providers = {
  antseed = {discovery='marketplace',discovery_id='antseed',base_url='http://buyer.test/v1',api_kind='openai_compatible',auth={kind='none'},tier='marketplace'},
  bedrock = {discovery='static',base_url='bedrock://us-east-1',api_kind='bedrock',aws_region='us-east-1',source='bedrock',tier='partner'},
  bedrock_market = {discovery='marketplace',discovery_id='bedrock_market',api_kind='bedrock',aws_region='us-east-1',source='bedrock',tier='partner'},
  custom_cloud = {discovery='static',base_url='http://custom.test/v1',api_kind='openai_compatible',auth_env='CUSTOM_CLOUD_SECRET',tier='partner'},
 },
 models = {
  ['shared-model'] = {served_by={{provider='bedrock',provider_model_id='aws.profile.model'},{provider='custom_cloud'}},capabilities={context=128000,supports_tools=true,supports_json_mode=true}},
 },
 profiles={default={scorer={'zero'}}},
 policy_envelope={'and',{'meets_req'},{'not',{'is','disabled'}},{'or',{'not',{'provider_eq','antseed'}},{'cmp','credits','ge',1}}},
}
