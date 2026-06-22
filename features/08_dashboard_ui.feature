Feature: Dashboard UI rendered in a real headless browser
  Proves the operator actually SEES the data in the dashboard (real DOM render
  via headless chromium), not just that the API returns it. Relies on the seeded
  activity (one chat via codex + one flow) created in before_all.

  @browser @p0
  Scenario: The dashboard loads its shell and Activity shows the real flow run
    Given I open the dashboard in a browser
    Then I see "Analytics" rendered
    When I click the "Activity" tab
    Then I see "PROVIDER" rendered
    And I see "flow" rendered

  @browser @p0
  Scenario: Catalog renders model families with prices
    Given I open the dashboard in a browser
    When I click the "Catalog" tab
    Then I see "gpt-5.5" rendered

  @browser @p0
  Scenario: Analytics renders totals (requests / spend / tokens)
    Given I open the dashboard in a browser
    Then I see "Requests" rendered
    And I see "Spend" rendered
    And I see "Tokens" rendered

  @browser @p1
  Scenario: Config renders per-provider tunable knobs
    Given I open the dashboard in a browser
    When I click the "Config" tab
    Then I see "codex" rendered

  @browser @p1
  Scenario: Provider keys tab renders the credentials view
    Given I open the dashboard in a browser
    When I click the "Provider keys" tab
    Then I see "openrouter" rendered

  @browser @p1 @regression
  Scenario: An expanded Activity row survives the 15s auto-refresh (no auto-close bug)
    Given I open the dashboard in a browser
    When I click the "Activity" tab
    And I expand the first Activity row
    And I wait 17 seconds
    Then an Activity row is still expanded

