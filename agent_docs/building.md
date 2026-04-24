# Building

This document covers build prerequisites, module metadata, and common build commands.

## Requirements
- Go 1.22+ (`go.mod:3`)
- Module path: `github.com/timwhitez/agent-sdk-golang` (`go.mod:1`)
- External dependency: `github.com/bmatcuk/doublestar/v4` (`go.mod:5`)

## Quick Build Commands
- Full workspace build: `go build ./...`
- Core runtime modules:
  - `go build ./sdk/agent`
  - `go build ./sdk/agent/compaction`
  - `go build ./sdk/tools`
  - `go build ./sdk/tools/sandbox`
  - `go build ./sdk/llm/openai`
  - `go build ./sdk/llm/anthropic`
  - `go build ./sdk/tokens`

## Module Hygiene
- Download dependencies: `go mod download`
- Validate checksums: `go mod verify`
- Normalize module graph: `go mod tidy`

## Optional Runtime Environment Knobs
- `BU_AGENT_SDK_CALCULATE_COST`: enables token-cost calculation path (`sdk/tokens/cost.go:80`)
- `XDG_CACHE_HOME`: overrides cache base directory for pricing cache (`sdk/tokens/cost.go:88`)
- Pricing data source is fetched and cached with a 24-hour TTL (`sdk/tokens/cost.go:19`, `sdk/tokens/cost.go:161`)

## Build Notes
- Repository is plain Go modules (no codegen bootstrap step required).
- Provider integrations are compile-time isolated by package (`sdk/llm/openai/`, `sdk/llm/anthropic/`).
