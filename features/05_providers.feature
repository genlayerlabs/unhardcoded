Feature: Providers — OpenRouter, Codex, discovery and registered model traits
  Asserts the configured providers are live and the registered benchmark/modality
  fields (model_meta) are part of the field vocabulary the builder/policies use.

  Background:
    Given the stack is healthy

  @p0 @providers
  Scenario: OpenRouter and Codex providers are present with health
    When I GET "/dashboard/api/policies" as admin
    Then the status is 200
    And the array "providers" includes an item where "name" equals "openrouter"
    And the array "providers" includes an item where "name" equals "openai"
    And every item in "providers" has a "health"

  @p1 @providers
  Scenario: Codex is configured as a ChatGPT-subscription (openai_codex) provider
    When I GET "/dashboard/api/policies" as admin
    Then the status is 200
    And the array "providers" includes an item where "name" equals "openai"
    And the matched item field "api_kind" equals "openai_codex"

  @p1 @providers
  Scenario: A Codex account is active (auth wired through)
    When I GET "/dashboard/api/codex/accounts" as admin
    Then the status is 200
    And the field "accounts" is non-empty
    And the field "active" is non-empty

  @p1 @providers
  Scenario: Registered model traits (model_meta benchmarks) are in the field vocabulary
    When I GET "/dashboard/api/fields" as admin
    Then the status is 200
    And the array "fields" includes an item where "name" equals "bench_intelligence"
    And the array "fields" includes an item where "name" equals "bench_coding"

  @p1 @providers
  Scenario: The discovered catalog exposes routable families
    Given I have a caller token
    When I GET "/v1/models" as consumer
    Then the status is 200
    And the array "data" includes an item where "id" equals "family:gpt-5.5"
