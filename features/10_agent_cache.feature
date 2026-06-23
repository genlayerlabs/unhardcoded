Feature: Agent cache affinity — cache_hot session stickiness
  An agent loop re-sends a large, stable prefix every turn. To make the
  provider's prompt-cache discount actually land, the router must keep the
  conversation pinned to the peer that already holds that prefix hot. This proves
  the session -> route_cache -> cache_hot pipeline end to end over /v1: a turn
  carrying a session teaches the router which peer served it, and the next turn's
  ranking gives that peer a decisive affinity bonus. Provider-health independent:
  the assertions read the RANKING from x_router.decision_trace, which is present
  even when execution later exhausts.

  Background:
    Given the stack is healthy
    And I have a caller token

  @p0 @api @agent @cache
  Scenario: An agent's session keeps its working route cache-hot across turns
    When an agent establishes session "agent-cache-1" with a free turn
    Then the agent's turn routed to a concrete peer
    When the agent re-ranks its turn with the same session as "hot"
    And the agent re-ranks the same turn with no session as "cold"
    Then the agent's route scores higher in "hot" than in "cold"

  @p1 @api @agent @cache
  Scenario: A brand-new session gets no phantom affinity
    When an agent establishes session "agent-cache-2" with a free turn
    And the agent re-ranks the same turn with unknown session "never-seen-zzz" as "fresh"
    And the agent re-ranks the same turn with no session as "none"
    Then the rankings "fresh" and "none" are identical
