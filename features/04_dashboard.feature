Feature: Dashboard data — what the operator console renders MUST be present and correct
  The dashboard is a thin renderer of /dashboard/api/*. These scenarios assert the
  backing data is complete and correct (so the frontend shows real, correct values
  in Analytics, Activity, Catalog, Config, Consumers, Provider keys). Seeded
  activity (one chat + one flow) is created in before_all.

  Background:
    Given the stack is healthy

  @p0 @dashboard
  Scenario: The dashboard HTML page loads with all its tabs and renderers
    When I GET "/dashboard" as admin
    Then the status is 200
    And the response text contains "Analytics"
    And the response text contains "Builder"
    And the response text contains "Activity"
    And the response text contains "Catalog"
    And the response text contains "Config"
    And the response text contains "renderActivity"
    And the response text contains "renderAnalytics"

  @p0 @dashboard
  Scenario: Analytics — totals, breakdowns and health are populated
    When I GET "/dashboard/api/stats" as admin
    Then the status is 200
    And the field "viewer_role" equals "admin"
    And the field "totals.requests" is at least 1
    And the field "totals.tokens_total" is a number
    And the field "totals.cost_usd" is a number
    And the field "by_provider" is non-empty
    And the field "by_status" is non-empty
    And the field "health_summary" is present
    And the array "daily_totals" has at least 1 items

  @p0 @dashboard
  Scenario: Activity — recent requests carry a full, correct per-request trace
    When I POST a free chat as consumer
    And I POST a free flow as consumer
    And I GET "/dashboard/api/stats" as admin
    Then the status is 200
    And the array "recent" has at least 2 items
    And every item in "recent" has a "status"
    And every item in "recent" has a "ts"
    And the array "recent" includes an item where "provider" equals "flow"
    And the array "recent" includes an item where "provider" equals "openai"

  @p0 @dashboard
  Scenario: Catalog (Market) — families list with prices and per-seller perf
    When I GET "/dashboard/api/market" as admin
    Then the status is 200
    And the array "families" has at least 3 items
    And every item in "families" has a "family"
    And every item in "families" has a "quality"
    And every item in "families" has a "rows"
    And the array "families" includes an item where "family" equals "gpt-5.5"

  @p0 @dashboard
  Scenario: Policies — the default profile and live providers with health
    When I GET "/dashboard/api/policies" as admin
    Then the status is 200
    And the array "profiles" includes an item where "name" equals "default"
    And the field "providers" is non-empty
    And every item in "providers" has a "health"

  @p1 @dashboard
  Scenario: Builder field vocabulary is available
    When I GET "/dashboard/api/fields" as admin
    Then the status is 200
    And the array "fields" includes an item where "name" equals "price_in"
    And the array "fields" includes an item where "name" equals "latency_ms"
    And the array "fields" includes an item where "name" equals "success_rate"

  @p1 @dashboard
  Scenario: Config — per-provider tunable knobs are present
    When I GET "/dashboard/api/config" as admin
    Then the status is 200
    And the field "knobs" is non-empty

  @p1 @dashboard
  Scenario: Consumers — the test consumer is listed with stats
    When I GET "/dashboard/api/keys" as admin
    Then the status is 200
    And the array "keys" includes an item where "consumer" equals "bdd-test"

  @p1 @dashboard
  Scenario: Provider keys — credentials snapshot is privatized but present
    When I GET "/dashboard/api/provider-keys" as admin
    Then the status is 200
    And the field "rows" is non-empty

  @p1 @dashboard
  Scenario: Codex accounts — an active account is configured
    When I GET "/dashboard/api/codex/accounts" as admin
    Then the status is 200
    And the field "accounts" is non-empty
    And the field "active" is non-empty
    And the field "activity" is present

  @p1 @dashboard
  Scenario: Builder dry-run ranking (policy preview) returns an ordering (no spend)
    When I POST "/dashboard/api/policy/preview" as admin with json
      """
      {"policy_ir":["policy",
        ["and",["meets_req"],["not",["is","disabled"]],["family_eq","gpt-5.5"]],
        ["neg",["normalize",["field","price_in"]]],
        ["argmax"],["id"],["always",{"action":"next_candidate"}]]}
      """
    Then the status is 200
    And the field "ranked" is non-empty
