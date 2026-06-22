Feature: AntSeed marketplace — wallet, escrow and on-chain money
  Split into two: @antseed READ-ONLY data checks (free — verify the dashboard
  shows the real wallet/escrow correctly; needs the antseed sidecar up + funded,
  so excluded from the default run), and @manual on-chain EXECUTION (real USDC on
  Base mainnet — deposit/withdraw/spend, run by hand).

  Run the read-only ones (with the sidecar up + funded):  behave --tags=antseed

  # ---- READ-ONLY: the money DATA the dashboard renders is real and correct ----

  @antseed @money
  Scenario: The dashboard Catalog shows the AntSeed wallet with real escrow data
    Given the stack is healthy
    When I GET "/dashboard/api/market" as admin
    Then the status is 200
    And the field "wallet.provider" equals "antseed"
    And the field "wallet.address" contains "0x"
    And the field "wallet.deposits_available" is a number
    And the field "wallet.deposits_reserved" is present
    And the field "wallet.connection" equals "connected"

  @antseed @money
  Scenario: AntSeed appears in the catalog as a marketplace provider
    Given the stack is healthy
    When I GET "/dashboard/api/policies" as admin
    Then the status is 200
    And the array "providers" includes an item where "name" equals "antseed"
    And the matched item field "tier" equals "marketplace"

  @antseed @spend @money
  Scenario: AntSeed serves a request (routes to a peer) — proves the marketplace works
    # NB: REAL MONEY. Costs a few cents of escrow + reserves ~1 USDC in a channel
    # (returns on settle). Gated behind RUN_ANTSEED_SPEND=1 so it never runs by
    # accident:  RUN_ANTSEED_SPEND=1 behave --tags=spend
    Given the stack is healthy
    And I have a caller token
    When I POST "/v1/chat/completions" as consumer with json
      """
      {"model":"","max_tokens":16,"messages":[{"role":"user","content":"Reply: pong"}],
       "policy_ir":["policy",
         ["and",["meets_req"],["not",["is","disabled"]],["family_eq","glm-5.2"]],
         ["neg",["normalize",["field","price_in"]]],
         ["argmax"],["id"],["always",{"action":"next_candidate"}]]}
      """
    Then the status is 200
    And the field "x_router.provider" equals "antseed"
    And the field "x_router.served_model_id" equals "glm-5.2"
    And the field "x_router.cost_usd" is a number

  # ---- MANUAL: real on-chain transactions (spend / move funds) ----

  @manual @money
  Scenario: Set up a local AntSeed dev wallet (manual — done once per machine)
    # The full local user flow (see .env.example + scripts/gen-dev-wallet.sh):
    #   1. ./scripts/gen-dev-wallet.sh        -> prints ANTSEED_IDENTITY_HEX +
    #      ANTSEED_CONTROL_TOKEN (a fresh secp256k1/EVM key — a DEV wallet, never prod)
    #   2. paste both into .env
    #   3. docker compose --profile antseed up -d --build
    #   4. docker compose exec antseed antseed buyer balance --json   # -> the address
    #   5. fund that address with a little USDC + ETH (gas) on Base mainnet
    #   6. Deposit into escrow from the dashboard Catalog (wallet cell)
    # The OUTCOME (wallet connected + escrow visible) is verified by the @antseed
    # read-only scenarios above.
    Given the stack is healthy

  @manual @money
  Scenario: Deposit USDC wallet -> escrow via the dashboard (real on-chain tx)
    # POST /dashboard/api/wallet/deposit {amount} -> /x/wallet/deposit -> sidecar
    # control :8379 -> antseed buyer deposit. Verified live: walletUSDC drops,
    # depositsAvailable rises, and the Catalog wallet cell shows it.
    Given the stack is healthy

  @manual @money
  Scenario: Withdraw escrow -> wallet (real on-chain tx)
    # POST /dashboard/api/wallet/withdraw {amount}. NOTE: the AntSeed deposits
    # contract LOCKS funds — an immediate withdraw after deposit reverts (custom
    # error 0xea8e4eb5). Funds are safe in escrow; withdrawable after the lock or
    # spendable by routing calls to antseed.
    Given the stack is healthy
