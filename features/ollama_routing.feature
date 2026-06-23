Feature: Ollama Provider Routing
  As a user
  I want to route requests to Ollama models
  So that I can use local or cloud models through the router

  Background:
    Given the stack is healthy
    And I have a caller token
    And a running router with Ollama provider configured

  @ollama
  Scenario: Route to local Ollama model
    Given Ollama is running locally with model "qwen2.5:0.5b"
    When I send a chat completion request with model "qwen2.5:0.5b"
    Then the request is routed to provider "ollama"
    And the response comes from Ollama
    And the cost is zero

  @ollama @cloud
  Scenario: Route to cloud Ollama model
    Given OLLAMA_CLOUD=1 and OLLAMA_API_KEY is set
    And the cloud model "gpt-oss:120b" is available
    When I send a chat completion request with model "gpt-oss:120b"
    Then the request is routed to provider "ollama"
    And the Authorization header is set for Ollama Cloud
    And the endpoint is "https://ollama.com"

  @ollama @cloud
  Scenario: Fallback from cloud to local
    Given OLLAMA_CLOUD=1 and OLLAMA_API_KEY is set
    And Ollama cloud is unavailable
    And Ollama is running locally with model "qwen2.5:0.5b"
    When I send a chat completion request with model "qwen2.5:0.5b"
    Then the request succeeds from local Ollama

  @ollama @cloud
  Scenario: Error when cloud required but no API key
    Given OLLAMA_CLOUD=1 and OLLAMA_API_KEY is not set
    When I send a chat completion request
    Then the response is an auth error
    And the error mentions "OLLAMA_API_KEY"
