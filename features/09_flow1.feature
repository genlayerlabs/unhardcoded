Feature: Flow 1 — the GLM ∥ GPT → merge ensemble produces the expected output
  The concrete ensemble we ship in opencode: GPT-5.5 (served by Codex, $0) in
  parallel with GLM-5.2 (OpenRouter), merged by GLM-5.2. Asserts that when run it
  yields exactly the expected shape — a 3-node DAG with Codex + OpenRouter — and
  that the run shows up correctly in the dashboard Activity.

  Background:
    Given the stack is healthy
    And I have a caller token

  @p0 @flow @flow1
  Scenario: Running flow1 returns the merged answer with the expected per-node routing
    When I run the flow1 ensemble (retry on flake)
    Then the status is 200
    And the field "object" equals "chat.completion"
    And the field "x_router.provider" equals "flow"
    And the field "choices[0].message.content" is non-empty
    And the array "x_router.decision_trace.flow_nodes" has at least 3 items
    And every item in "x_router.decision_trace.flow_nodes" has a "provider"
    And every item in "x_router.decision_trace.flow_nodes" has a "served_model_id"
    And the array "x_router.decision_trace.flow_nodes" includes an item where "provider" equals "openai"
    And the array "x_router.decision_trace.flow_nodes" includes an item where "served_model_id" equals "z-ai/glm-5.2"

  @p0 @flow @flow1
  Scenario: The Codex (gpt) node really served via the subscription at $0
    When I run the flow1 ensemble (retry on flake)
    Then the status is 200
    And the array "x_router.decision_trace.flow_nodes" includes an item where "served_model_id" equals "gpt-5.5"
    And the matched item field "provider" equals "openai"
    And the matched item field "price_out" equals "0.0"

  @p1 @flow @flow1
  Scenario: The flow1 run is recorded correctly in the dashboard Activity
    When I run the flow1 ensemble (retry on flake)
    Then the status is 200
    When I GET "/dashboard/api/stats" as admin
    Then the status is 200
    And the array "recent" includes an item where "provider" equals "flow"
