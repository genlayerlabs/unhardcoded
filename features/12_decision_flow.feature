Feature: Decision-routed generation flows
  A caller declares economical and capable generation policies in a flow.
  A decision model selects one branch; the caller cannot inject an undeclared policy.

  Background:
    Given the stack is healthy
    And I have a caller token

  @p1 @flow
  Scenario: Every branch is admitted and decision routing survives normalization
    When I normalize a decision-routed flow with fallback "valid"
    Then the status is 200
    And the normalized flow retains both generation choices

  @p1 @flow
  Scenario: An unknown fallback is rejected before inference
    When I normalize a decision-routed flow with fallback "unknown"
    Then the status is 400
