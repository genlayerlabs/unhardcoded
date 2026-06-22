Feature: Authentication — dashboard sessions and the caller bearer contract

  Background:
    Given the stack is healthy

  @p0 @auth
  Scenario: DASHBOARD_NO_AUTH grants local admin to the console API
    When I GET "/dashboard/api/stats" as admin
    Then the status is 200
    And the field "viewer_role" equals "admin"

  @p0 @auth
  Scenario: A valid caller bearer token is accepted on /v1
    Given I have a caller token
    When I GET "/v1/models" as consumer
    Then the status is 200

  @p0 @auth
  Scenario: A missing caller token is rejected on /v1
    When I GET "/v1/models" as none
    Then the status is 401
    And the field "error.code" equals "caller_auth"

  @p1 @auth
  Scenario: A consumer can log into the dashboard with their API key (scoped session)
    Given I have a caller token
    When I log into the dashboard with my caller key
    Then the status is 200
    And the field "role" equals "consumer"

  @manual @auth
  Scenario: Admin password login (manual — needs DASHBOARD_PASSWORD_SHA256 set and NO_AUTH off)
    # POST /dashboard/login {password} -> sets an admin session cookie.
    # Not auto-tested: the local dev stack runs with DASHBOARD_NO_AUTH=1.
    Given the stack is healthy

  @manual @auth
  Scenario: Trusted-header SSO admin (manual — needs a reverse proxy injecting the header+secret)
    Given the stack is healthy
