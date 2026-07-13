Feature: Agent context compaction — summary-only transform over /v1/compact
  An agent loop owns its context layout and sends only complete newly-aged
  interaction groups to /v1/compact. The router returns raw summary text and
  never receives or reconstructs authority or the retained tail. This exercises
  the transform end to end over the consumer surface. The local Compose router
  owns a profile restricted to its documented Ollama test model at zero price,
  so the summary is free and hermetic without accepting a caller policy.

  Background:
    Given the stack is healthy
    And I have a caller token

  @api @agent @compact
  Scenario: An agent summarizes caller-selected aged interactions
    Given newly-aged complete interactions sealable by the local model
    When the agent requests an aged-context summary
    Then the summary is compacted
    And the response contains only raw summary output
