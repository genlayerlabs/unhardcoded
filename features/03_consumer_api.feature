Feature: Consumer API flows (/v1) — the calling service's surface
  As a consuming service I call /v1 with my bearer token and the router
  decides/falls-back over the operator's provider keys. All end-to-end chats
  here route to codex ($0) so the suite is free.

  Background:
    Given the stack is healthy
    And I have a caller token

  @p0 @api
  Scenario: List the routable model catalog
    When I GET "/v1/models" as consumer
    Then the status is 200
    And the field "object" equals "list"
    And the array "data" has at least 5 items
    And the array "data" includes an item where "id" equals "profile:default"

  @p0 @api
  Scenario: Chat completion runs a policy and returns a real answer + trace
    When I POST a free chat as consumer
    Then the status is 200
    And the field "object" equals "chat.completion"
    And the field "choices[0].message.content" is non-empty
    And the field "usage.total_tokens" is a number
    And the field "x_router.provider" is non-empty
    And the field "x_router.served_model_id" is non-empty
    And the field "x_router.decision_trace" is present

  @p0 @api
  Scenario: Per-call policy_ir is admitted and executed
    When I POST "/v1/chat/completions" as consumer with json
      """
      {"model":"","max_tokens":16,"messages":[{"role":"user","content":"hi"}],
       "policy_ir":["policy",
         ["and",["meets_req"],["not",["is","disabled"]],["family_eq","gpt-5.5"]],
         ["neg",["normalize",["field","price_in"]]],
         ["argmax"],["id"],["always",{"action":"next_candidate"}]]}
      """
    Then the status is 200
    And the field "x_router.policy_fingerprint" is present
    And the field "choices[0].message.content" is non-empty

  @p1 @api
  Scenario: Malformed policy_ir is rejected cleanly at admission (no spend)
    When I POST "/v1/chat/completions" as consumer with json
      """
      {"model":"","messages":[{"role":"user","content":"hi"}],
       "policy_ir":["policy","not-a-valid-term"]}
      """
    Then the status is 400
    And the field "error.type" equals "invalid_request_error"
    And the field "error.message" contains "policy_ir"

  @p0 @api
  Scenario: Sigma_flow DAG runs and returns the sink answer with a per-node trace
    When I POST a free flow as consumer
    Then the status is 200
    And the field "x_router.provider" equals "flow"
    And the field "choices[0].message.content" is non-empty
    And the array "x_router.decision_trace.flow_nodes" has at least 2 items
    And every item in "x_router.decision_trace.flow_nodes" has a "provider"
    And every item in "x_router.decision_trace.flow_nodes" has a "served_model_id"

  @p1 @api
  Scenario: Malformed flow_ir is rejected at admission
    When I POST "/v1/chat/completions" as consumer with json
      """
      {"model":"","messages":[{"role":"user","content":"hi"}],
       "flow_ir":["flow",{"out":{"kind":"output","inputs":["missing"]}}]}
      """
    Then the status is 400
    And the field "error.message" contains "flow_ir"

  @p1 @api
  Scenario: Per-key usage self-service is scoped and sanitized
    When I POST a free chat as consumer
    And I GET "/v1/usage?window=24h" as consumer
    Then the status is 200
    And the field "kind" equals "router_key_usage"
    And the field "key_sha256_prefix" is non-empty
    And the field "totals.requests" is at least 1
    And the field "consumer_settings.status" is present

  @p0 @api
  Scenario: Missing bearer token is rejected
    When I GET "/v1/models" as none
    Then the status is 401
    And the field "error.code" equals "caller_auth"

  @p0 @api
  Scenario: Unknown bearer token is rejected
    When I GET "/v1/models" as bad
    Then the status is 401
    And the field "error.code" equals "caller_auth"
