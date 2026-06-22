Feature: Onboarding & setup — a new user gets a running, healthy stack
  The clone/compose steps themselves are environment-level (@manual: cannot be
  re-run inside the suite); here we assert their OUTCOME on the running stack.

  @p0 @onboarding
  Scenario: The core engine submodule is populated (recursive clone outcome)
    Then the file "core/router.lua" exists
    And the file "core/llm_policy.lua" exists

  @p0 @onboarding
  Scenario: The stack is up and healthy (compose up outcome)
    Given the stack is healthy
    When I GET "/healthz" as none
    Then the status is 200
    And the field "ok" equals "True"

  @p0 @onboarding
  Scenario: The router loaded its catalog (engine embedded + config.live.lua)
    Given I have a caller token
    When I GET "/v1/models" as consumer
    Then the status is 200
    And the array "data" has at least 5 items

  @manual @onboarding
  Scenario: Recursive clone (manual — run once on a fresh machine)
    # git clone --recursive https://github.com/genlayerlabs/unhardcoded.git
    # -> core/ submodule populated; covered by the 'submodule populated' outcome above.
    Given the stack is healthy

  @manual @onboarding
  Scenario: docker compose up --build (manual — environment setup)
    # cp .env.example .env.secrets; fill secrets; docker compose up -d --build
    # -> router + ingress healthy; covered by the 'stack up and healthy' outcome above.
    Given the stack is healthy
