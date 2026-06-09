# Testing

This document summarizes recommended test commands and current coverage focus.

## Core Commands
- Run all tests: `go test ./...`
- Verbose run: `go test -v ./...`
- Race detection: `go test -race ./...`

## Focused Commands by Area
- Agent loop: `go test ./sdk/agent`
- Compaction service: `go test ./sdk/agent/compaction`
- Tools + args/schema/deps: `go test ./sdk/tools`
- Sandbox tools: `go test ./sdk/tools/sandbox`
- OpenAI provider: `go test ./sdk/llm/openai`
- Anthropic provider: `go test ./sdk/llm/anthropic`

## Coverage Map (Representative)
- `sdk/agent/agent_test.go` - max-token auto-continue metadata emission, overflow-triggered compaction checks, async compaction apply-on-next-turn behavior, structured compaction telemetry, compaction system-message deduplication, tool-call delta merge behavior, and truncation metadata/path persistence (`sdk/agent/agent_test.go:180`, `sdk/agent/agent_test.go:402`, `sdk/agent/agent_test.go:271`, `sdk/agent/agent_test.go:210`, `sdk/agent/agent_test.go:421`)
- `sdk/agent/agent_compaction_local_test.go` - automatic Tier 1 local snip
  compaction runs without invoking the summary model and applies through the
  same pending-compaction boundary.
- `sdk/agent/agent_stream_response_test.go` - stream response metadata is preserved into `Completion.ResponseID` and propagated into `UsageEvent` / `FinalResponseEvent` for stream consumers (`sdk/agent/agent_stream_response_test.go:31`, `sdk/agent/agent_stream_response_test.go:56`)
- `sdk/agent/agent_steering_test.go` - steering channel lifecycle under cancellation (`sdk/agent/agent_steering_test.go:31`)
- `sdk/agent/agent_max_iterations_test.go` - max-iteration error/final event contract (`sdk/agent/agent_max_iterations_test.go:23`)
- `sdk/agent/agent_streaming_error_test.go` - partial output persistence and streamed error metadata propagation (`sdk/agent/agent_streaming_error_test.go:36`, `sdk/agent/agent_streaming_error_test.go:94`)
- `sdk/agent/agent_tool_resolve_test.go` - alias resolution, normalized collisions, unknown-tool fallback (`sdk/agent/agent_tool_resolve_test.go:117`, `sdk/agent/agent_tool_resolve_test.go:143`, `sdk/agent/agent_tool_resolve_test.go:229`)
- `sdk/agent/agent_retry_loop_guard_test.go` - retry/backoff behavior and repeated-tool loop guard warnings, doom-loop aborts, reminder injection, and synthetic skipped tool results that preserve contiguous assistant tool-call/tool-result history (`sdk/agent/agent_retry_loop_guard_test.go`)
- `sdk/agent/agent_ephemeral_test.go` - ephemeral retention behavior across turns (`sdk/agent/agent_ephemeral_test.go:56`)
- `sdk/agent/agent_todo_prompt_test.go` - hidden todo reminder injection when work remains (`sdk/agent/agent_todo_prompt_test.go:30`)
- `sdk/agent/agent_compaction_error_test.go` - compaction-path provider failure logging, retry-once behavior, below-threshold short-circuiting, and disabled-compactor cache behavior (`sdk/agent/agent_compaction_error_test.go:61`, `sdk/agent/agent_compaction_error_test.go:93`, `sdk/agent/agent_compaction_error_test.go:127`, `sdk/agent/agent_compaction_error_test.go:146`)
- `sdk/agent/compaction/service_test.go` - summary prefix/tagging, overflow-limit checks (`context_window - reserve_output_tokens`), recent-user retention, strict last-match summary extraction (`<summary>`/`<compaction_summary>`), timeout-bounded compaction, structured summary telemetry, rune-aware tool-context threshold gating, model-aware summary prompts, protected-tool snapshot prioritization, and tool-call/tool-result pair repair after destroyed tool outputs are filtered (`sdk/agent/compaction/service_test.go:54`, `sdk/agent/compaction/service_test.go:147`, `sdk/agent/compaction/service_test.go:300`, `sdk/agent/compaction/service_test.go:372`)
- `sdk/agent/compaction/ledger_test.go` - compaction ledger schema validation,
  replacement hash checks, duplicate replacement rejection, stable message-key
  normalization, and `LedgerStore` interface compile coverage.
- `sdk/agent/compaction/local_reduce_test.go` - tool-result snip replacements,
  stable ledger reuse, protected-zone and protected-tool skips, artifact-write
  failure warnings, no user-message rewriting, and provider-valid history.
- `sdk/agent/compaction/prune_test.go` - prune watermark behavior, monotonic
  tool-result replacement upgrades, assistant-text compaction, assistant
  tool-call preservation, and user-message preservation.
- `sdk/agent/compaction/incremental_summary_test.go` - ledger-backed summary
  metadata, second-pass previous-summary-plus-delta prompt construction, and
  summary extraction failure atomicity.
- `sdk/agent/compaction/models_test.go` - compaction prompt contract checks plus summary-prompt resolver behavior for model-aware and fallback paths (`sdk/agent/compaction/models_test.go:8`, `sdk/agent/compaction/models_test.go:21`, `sdk/agent/compaction/models_test.go:46`, `sdk/agent/compaction/models_test.go:60`)
- `sdk/tools/args_normalize_test.go` - arg normalization/repair pipeline, metadata tagging, and tool-specific offset/line alias handling (`sdk/tools/args_normalize_test.go:12`, `sdk/tools/args_normalize_test.go:154`, `sdk/tools/args_normalize_test.go:277`)
- `sdk/tools/schema_test.go` - schema alias repair and tool execute decode behavior (`sdk/tools/schema_test.go:74`, `sdk/tools/schema_test.go:113`)
- `sdk/tools/deps_test.go` - dependency container concurrency memoization and non-caching of errors (`sdk/tools/deps_test.go:12`, `sdk/tools/deps_test.go:72`)
- `sdk/tools/sandbox/sandbox_test.go` - path safety, allowlist behavior, confirmer gating, edit/apply_patch/read/webfetch/glob/grep guardrails, webfetch read-error surfacing, `ls`/`grep` malformed-glob rejection, glob stat-failure warning/metadata surfacing, and grep scan-failure diagnostics for open/read/seek/walk paths (`sdk/tools/sandbox/sandbox_test.go:58`, `sdk/tools/sandbox/sandbox_test.go:89`, `sdk/tools/sandbox/sandbox_test.go:189`, `sdk/tools/sandbox/sandbox_test.go:353`, `sdk/tools/sandbox/sandbox_test.go:429`, `sdk/tools/sandbox/sandbox_test.go:468`, `sdk/tools/sandbox/sandbox_test.go:868`, `sdk/tools/sandbox/sandbox_test.go:980`)
- `sdk/llm/openai/chat_test.go` - chat downgrade logic, endpoint version-path detection, stream parse errors, retryable error classification (`sdk/llm/openai/chat_test.go:39`, `sdk/llm/openai/chat_test.go:200`, `sdk/llm/openai/chat_test.go:174`)
- `sdk/llm/openai/responses_stream_test.go` - responses streaming error event behavior (`sdk/llm/openai/responses_stream_test.go:13`)
- `sdk/llm/anthropic/client_test.go` - usage/response-id mapping, downgrade retries, stream error behavior, retryable error classification, jitter entropy bounds, and tool-ID normalization warning payloads (`sdk/llm/anthropic/client_test.go:47`, `sdk/llm/anthropic/client_test.go:84`, `sdk/llm/anthropic/client_test.go:146`, `sdk/llm/anthropic/client_test.go:654`, `sdk/llm/anthropic/client_test.go:670`)
- `sdk/tokens/cost_test.go` - initialization concurrency, cached-token clamping, warning surfacing for pricing-init/cache-read/cache-stat/cache-parse/cost-calc failures, cache-write warning behavior, and alias/family pricing lookup fallback (`sdk/tokens/cost_test.go:20`, `sdk/tokens/cost_test.go:65`, `sdk/tokens/cost_test.go:133`, `sdk/tokens/cost_test.go:223`, `sdk/tokens/cost_test.go:258`, `sdk/tokens/cost_test.go:316`)

## Practical Test Strategy
- Run area-focused tests first when editing a subsystem.
- Run `go test ./...` before finalizing cross-cutting changes.
- Use `-race` for changes involving tool dependency container concurrency (`sdk/tools/deps.go:198`).
