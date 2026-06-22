Feature: Consumer key lifecycle (operator issues + governs ingress tokens)
  Each scenario mints its own throwaway consumer so they stay isolated. All
  rejections happen at the ingress BEFORE any LLM call, so these are free.

  Background:
    Given the stack is healthy

  @p0 @consumer-keys
  Scenario: A freshly issued key authenticates against /v1 immediately
    When I create a consumer key for "bdd-new"
    Then the status is 200
    And the field "api_key" is non-empty
    And the field "sha256_prefix" is non-empty
    When I GET "/v1/models" as consumer
    Then the status is 200
    And the field "object" equals "list"

  @p1 @consumer-keys
  Scenario: allowed_routes restricts which routes a key may call
    When I create a consumer key for "bdd-route"
    Then the status is 200
    When I POST "/dashboard/api/consumers/bdd-route" as admin with json
      """
      {"allowed_routes":["family:does-not-exist"]}
      """
    Then the status is 200
    When I POST "/v1/chat/completions" as consumer with json
      """
      {"model":"family:gpt-5.5","messages":[{"role":"user","content":"hi"}]}
      """
    Then the status is 403
    And the field "error.code" equals "caller_route_not_allowed"

  @p1 @consumer-keys
  Scenario: rate_per_min / burst throttle a key
    When I create a consumer key for "bdd-rate"
    Then the status is 200
    When I POST "/dashboard/api/consumers/bdd-rate" as admin with json
      """
      {"allowed_routes":[],"rate_per_min":1,"burst":1}
      """
    Then the status is 200
    When I POST a free chat as consumer
    Then the status is 200
    When I POST a free chat as consumer
    Then the status is 429
    And the field "error.code" equals "caller_rate_limit"

  @p1 @consumer-keys
  Scenario: A revoked key is rejected immediately
    # Revoke drops the key's hash, so the token becomes unknown -> 401 caller_auth
    # (not 403 caller_key_revoked, which only applies while the hash still maps).
    When I create a consumer key for "bdd-revoke"
    Then the status is 200
    When I revoke the created key
    Then the status is 200
    And the field "removed_hashes" equals 1
    When I GET "/v1/models" as consumer
    Then the status is 401
    And the field "error.code" equals "caller_auth"

  @p2 @consumer-keys
  Scenario: An inactive consumer's keys are all rejected
    When I create a consumer key for "bdd-inactive"
    Then the status is 200
    When I POST "/dashboard/api/consumers/bdd-inactive" as admin with json
      """
      {"status":"inactive"}
      """
    Then the status is 200
    When I GET "/v1/models" as consumer
    Then the status is 403
    And the field "error.code" equals "caller_inactive"
