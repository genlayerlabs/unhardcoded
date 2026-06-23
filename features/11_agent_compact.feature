Feature: Agent context compaction — append-only sealing over /v1/compact
  An agent loop grows its context until it must be compacted. /v1/compact seals
  the AGED middle into one cheaply-routed summary and splices it back, leaving the
  frozen system prefix and the recent turns byte-identical so everything upstream
  stays prompt-cache hot. This exercises it end to end over the consumer surface,
  routing the summary to the local Ollama model so it is free and reliable.

  Background:
    Given the stack is healthy
    And I have a caller token

  @api @agent @compact
  Scenario: An agent seals its aged context while preserving prefix and tail
    Given a long agent conversation sealable by the local model
    When the agent compacts its context keeping the last 4 turns
    Then the context is compacted
    And the system prefix is preserved
    And the last 4 turns are preserved verbatim
