# Testing

This document summarizes recommended test commands and current coverage focus.

## Core Commands
- Run all tests: `go test ./...`
- Verbose run: `go test -v ./...`
- Race detection: `go test -race ./...`

## Focused Commands by Area
- Agent loop: `go test ./sdk/agent`
- Accounting schema/projectors: `go test ./sdk/accounting`
- Canonical artifact contract: `go test ./sdk/artifact`
- Compaction service: `go test ./sdk/agent/compaction`
- Message origin classifier: `go test ./sdk/agent/messageorigin`
- Tools + args/schema/deps: `go test ./sdk/tools`
- Sandbox tools: `go test ./sdk/tools/sandbox`
- Process runner and canonical raw streams: `go test ./sdk/tools/execrunner`
- OpenAI provider: `go test ./sdk/llm/openai`
- Anthropic provider: `go test ./sdk/llm/anthropic`

## Coverage Map (Representative)
- `sdk/agent/agent_test.go` - max-token auto-continue metadata emission, overflow-triggered compaction checks, async compaction apply-on-next-turn behavior, structured compaction telemetry, compaction system-message deduplication, tool-call delta merge behavior, and truncation metadata/path persistence (`sdk/agent/agent_test.go:180`, `sdk/agent/agent_test.go:402`, `sdk/agent/agent_test.go:271`, `sdk/agent/agent_test.go:210`, `sdk/agent/agent_test.go:421`)
- `sdk/accounting/projector_contract_test.go` and
  `sdk/agent/agent_accounting_contract_test.go` cover allowlisted bounded
  projection, independent scan/return disposition, secret/raw/path exclusion,
  provider-usage unknown handling, compaction summary/path exclusion,
  estimator failure, source-event adjacency, and monotonic sequence.
- `sdk/agent/agent_artifact_boundary_test.go` - oversized canonical-envelope
  re-encoding, byte/token budgets with CJK/emoji/no-newline fixtures, exact
  sink-byte recovery, provider/body metadata identity, sink/capability failure
  diagnostics without fabricated refs, invalid owner/hash/byte/recovery
  manifest rejection, ordered raw-source to logical-result lineage for bounded
  derived results, malformed/duplicate/owner-mismatched/ephemeral source
  rejection, sink lineage-preservation enforcement, dynamic owner resolution
  across session switches, provider failure, concurrent whole-owner snapshots, existing-envelope owner
  revalidation above and below the result budget, and durable host-object survival across `ClearHistory`/
  `ReplaceHistory`.
- `sdk/artifact/contract_test.go` - schema/owner/lineage/measurement/retention
  validation, explicit resolver registration, fixed recovery/continuation field
  preservation, codec byte/token budgets, exact full-versus-truncated-prefix
  view state, UTF-8 preview reduction, and clone
  isolation.
- `sdk/tools/execrunner/runner_canonical_test.go` - separate canonical
  stdout/stderr ownership, 150 KiB tail recovery, exact stream byte/hash
  manifests, no legacy combined duplicate in canonical mode, stream-aware
  progress chunks, UTF-8-safe single-line CJK/emoji previews, and fail-closed
  short-write diagnostics.
- `sdk/agent/agent_compaction_local_test.go` - automatic Tier 1 local snip
  compaction runs without invoking the summary model and applies through the
  same pending-compaction boundary.
- `sdk/agent/agent_compaction_canonical_test.go` - Agent construction and
  compaction-config replacement retain one canonical owner/sink/resolver/
  capability binding, while the default envelope codec alone leaves legacy
  embedders in path-writer compatibility mode.
- `sdk/agent/agent_stream_response_test.go` - stream response metadata is preserved into `Completion.ResponseID` and propagated into `UsageEvent` / `FinalResponseEvent` for stream consumers (`sdk/agent/agent_stream_response_test.go:31`, `sdk/agent/agent_stream_response_test.go:56`)
- `sdk/agent/agent_steering_test.go` and
  `sdk/agent/agent_steering_interrupt_test.go` cover steering channel ownership,
  streaming interruption, active-tool stage cancellation without root-query
  cancellation, partial history, continuation after steering, and the delayed
  host-acknowledgement race where steering is already in history before the
  current provider stage is canceled.
- `sdk/agent/agent_max_iterations_test.go` covers max-iteration error/final
  events plus require-done recovery: an empty/text-only post-tool stop forces
  the next auto request to `tool_choice=required` with `DisableThinking` set (so
  the forced choice stays legal under Anthropic extended thinking), and keeps
  `DisableThinking` active across an ordinary recovery tool plus its follow-up
  auto request until the done tool completes, while
	  providers that keep answering in text are bounded by the safety valve, which
	  returns the model's latest post-tool response with terminal status
	  `partial` and reason `require_done_safety`. Anthropic
	  request-shaping for this (`DisableThinking` suppresses manual/adaptive
	  thinking; other forced `tool_choice` conflicts return an actionable error)
	  is covered by `sdk/llm/anthropic/client_test.go`
	  `TestBuildRequestToolChoiceUnderThinking`. Adaptive
	  `thinking.type="adaptive"` plus `output_config.effort`, including downgrade
	  retry behavior, is covered by the adjacent adaptive-thinking tests.
- `sdk/agent/agent_streaming_error_test.go` - partial output persistence and streamed error metadata propagation (`sdk/agent/agent_streaming_error_test.go:36`, `sdk/agent/agent_streaming_error_test.go:94`)
- `sdk/agent/agent_tool_resolve_test.go` - alias resolution, normalized collisions, unknown-tool fallback (`sdk/agent/agent_tool_resolve_test.go:117`, `sdk/agent/agent_tool_resolve_test.go:143`, `sdk/agent/agent_tool_resolve_test.go:229`)
- `sdk/agent/agent_retry_loop_guard_test.go` - retry/backoff behavior and repeated-tool loop guard warnings, non-fatal loop-guard retreat (formerly doom-loop aborts), reminder injection, and synthetic skipped tool results that preserve contiguous assistant tool-call/tool-result history (`sdk/agent/agent_retry_loop_guard_test.go`)
- `sdk/agent/agent_evidence_progress_test.go` - read alias normalization,
  separate line/absolute-byte range units, byte-subrange coverage across
  `read`/`read_file`, repeat suppression, mixed-batch continuity, conservative
  invalidation after arbitrary successful non-evidence tools, target-state
  changes, and one post-compaction revalidation.
- `sdk/agent/agent_message_origin_test.go` plus the continuation, idle,
  loop-guard, require-done, early-stop, and evidence-progress tests verify that
  framework-authored user-role history carries stable `sdk_internal_*` names.
- `sdk/agent/agent_usage_test.go` - zero-prompt provider usage falls back to effective estimates, retains raw provider values, emits one warning per query, and does not invent usage when the provider omits the usage object.
- `sdk/agent/agent_ephemeral_test.go` - ephemeral retention behavior across turns (`sdk/agent/agent_ephemeral_test.go:56`)
- `sdk/agent/agent_todo_prompt_test.go` - hidden todo reminder injection when work remains (`sdk/agent/agent_todo_prompt_test.go:30`)
- `sdk/agent/agent_compaction_error_test.go` - compaction-path provider failure
  logging, retry-once behavior, below-threshold short-circuiting,
  disabled-compactor cache behavior, and the Tier 4 guarantee that a failed
  overflow compaction stops before the next provider request.
- `sdk/agent/compaction/service_test.go` - summary prefix/tagging,
  overflow-limit checks (`context_window - reserve_output_tokens`), real-user
  retention that excludes named and narrowly detected legacy internal messages,
  similar legitimate user-text preservation, strict last-match summary
  extraction, timeout-bounded compaction, telemetry, tool-context gating,
  provider-valid tool topology repair, first/latest real-user anchors,
  newest-24 key-event selection, unverified assistant-claim labeling, and
  UNKNOWN/warning behavior for host checkpoint provider failure. Watermark
  anchors cover the shared usable prompt window, exhausted-budget behavior, and
  the 70%/80%/85%/100% decision matrix. Quality-gate anchors cover system/data
  role separation, untrusted boundaries, required-section rejection,
  credential-like task material acceptance, and history/ledger atomicity on
  rejected summaries. The same suite drives real framed `Compact` requests to
  verify latest-user and verified-checkpoint coverage plus malformed-frame
  rejection.
- `sdk/agent/compaction/ledger_test.go` - compaction ledger schema validation,
  replacement hash checks, duplicate replacement rejection, stable message-key
  normalization, canonical-manifest durability and provider-stub checks,
  canonical/legacy slot mutual exclusion, and `LedgerStore` interface compile
  coverage.
- `sdk/agent/compaction/local_reduce_test.go` - tool-result snip replacements,
  stable ledger reuse, generated-marker parsing, repeated-pass fixed-point
  idempotency, same-text no-op handling, complete-current-turn/open-tool-block/
  recent-token protected zones, protected-tool skips, artifact-write warnings,
  latest-real-user microcompact protection, and provider-valid history.
- `sdk/agent/compaction/local_reduce_canonical_test.go` - full resolver
  validation for canonical envelopes and plain-source writes, immutable
  manifest identity, owner/hash/byte/retention/sink failure preservation,
  legacy in-place migration only while exact source bytes remain, snip-to-prune
  ref reuse, mismatched prune-parent rejection, and ledger/checkpoint round-trip
  recovery of the same object.
- Compaction truncation anchors
  (`TestCompactionTruncationPreservesValidUTF8`,
  `TestSummaryDeltaTruncationPreservesChinesePath`,
  `TestCompactionMaterialUsesTokenBudget`, `TestTruncationMarkerIsExplicit`,
  `TestASCIIAndCJKBudgetsRemainBounded`, and
  `TestAssistantPreviewUsesSharedTokenTruncator`) verify UTF-8 validity,
  injected-estimator budgets, explicit omission markers, and preserved exact
  identifiers across full and incremental summary material.
- `sdk/agent/messageorigin/origin_test.go` - stable constructor names, reserved
  origin recognition, destroyed-user exclusion, and preservation of unknown
  named real users.
- `sdk/agent/compaction/prune_test.go` - prune watermark behavior, monotonic
  tool-result replacement upgrades, fixed-point tool/assistant idempotency,
  assistant-text compaction, assistant tool-call preservation, and user-message
  preservation.
- `sdk/agent/compaction/pipeline_test.go` - ordered tier execution and merged
  telemetry, including the contract that provider trigger usage stays in
  `Result.Usage` while `OriginalTokens`/`NewTokens` share one estimator and
  declare `TokenCountSourceEstimate`, plus deferred local ledger mutation when a
  runtime checkpoint writer is configured and transaction retention when a
  failed summary falls back to successful local reduction.
- `sdk/agent/compaction/incremental_summary_test.go` - ledger-backed summary
  metadata, hash/coverage mismatch full rebuild, stable covered-end checkpoint
  identity, source-snapshot truthfulness, second-pass previous-summary-plus-
  delta prompt construction, and summary extraction failure atomicity.
- `sdk/agent/compaction/runtime_checkpoint_test.go` and
  `sdk/agent/agent_compaction_checkpoint_test.go` cover deterministic checkpoint
  IDs, tamper rejection, persist-before-apply ordering, manual failure
  atomicity, deferred summary-ledger commit, ledger rollback after checkpoint
  failure, visible ledger-commit/rollback failures, overflow local-fallback
  commit, stale-ledger refresh after a successful full rebuild, and retryable
  automatic checkpoint failure.
- `sdk/agent/compaction/models_test.go` - compaction prompt contract checks,
  including exact parity between the prompt's canonical `##` section headings
  and the validator, plus summary-prompt resolver behavior for model-aware and
  fallback paths.
- `sdk/agent/compaction/validation.go` is covered by
  `TestCompactionQualityGateRejectsMissingRequiredSections`,
  `TestCompactionQualityGateAllowsCredentialLikeSecurityMaterial`, and
  `TestRejectedSummaryDoesNotMutateHistoryOrLedger`, plus framed-material
  fact-coverage and malformed-frame integration tests in `service_test.go`.
- `sdk/agent/compaction/checkpoint.go` is covered through service and Goode
  adapter tests: it applies per-type entry limits and a final token bound to the
  portable host checkpoint schema without importing repository packages.
- `sdk/tools/args_normalize_test.go` - arg normalization/repair pipeline, metadata tagging, and tool-specific offset/line alias handling (`sdk/tools/args_normalize_test.go:12`, `sdk/tools/args_normalize_test.go:154`, `sdk/tools/args_normalize_test.go:277`)
- `sdk/tools/schema_test.go` - schema alias repair and tool execute decode behavior (`sdk/tools/schema_test.go:74`, `sdk/tools/schema_test.go:113`)
- `sdk/tools/deps_test.go` - dependency container concurrency memoization and non-caching of errors (`sdk/tools/deps_test.go:12`, `sdk/tools/deps_test.go:72`)
- `sdk/tools/sandbox/sandbox_test.go` - path safety, allowlist behavior, confirmer gating, edit/apply_patch/read/webfetch/glob/grep guardrails, webfetch read-error surfacing, `ls`/`grep` malformed-glob rejection, glob stat-failure warning/metadata surfacing, and grep scan-failure diagnostics for open/read/seek/walk paths (`sdk/tools/sandbox/sandbox_test.go:58`, `sdk/tools/sandbox/sandbox_test.go:89`, `sdk/tools/sandbox/sandbox_test.go:189`, `sdk/tools/sandbox/sandbox_test.go:353`, `sdk/tools/sandbox/sandbox_test.go:429`, `sdk/tools/sandbox/sandbox_test.go:468`, `sdk/tools/sandbox/sandbox_test.go:868`, `sdk/tools/sandbox/sandbox_test.go:980`)
- `sdk/llm/usage_test.go`, `sdk/llm/openai/chat_test.go`, `sdk/llm/openai/responses_parse_test.go`, and `sdk/llm/anthropic/client_test.go` cover the versioned prompt-token contract, legacy quality labeling, cached/image breakdowns without double counting, and Anthropic streaming/non-streaming parity.
- `sdk/llm/openai/responses_stream_test.go` - responses streaming error event behavior (`sdk/llm/openai/responses_stream_test.go:13`)
- `sdk/llm/openai/responses_terminal_test.go` - buffered and streaming Responses terminal-state, usage/ID ordering, refusal visibility, and typed-error coverage
- `sdk/llm/anthropic/client_test.go` - usage/response-id mapping, downgrade retries, stream error behavior, retryable error classification, jitter entropy bounds, and tool-ID normalization warning payloads (`sdk/llm/anthropic/client_test.go:47`, `sdk/llm/anthropic/client_test.go:84`, `sdk/llm/anthropic/client_test.go:146`, `sdk/llm/anthropic/client_test.go:654`, `sdk/llm/anthropic/client_test.go:670`)
- `sdk/llm/anthropic/client_agent_history_test.go` uses a strict two-request
  HTTP/SSE fixture to prove that streamed thinking text plus
  `signature_delta` is retained in assistant history and replayed before the
  prior `tool_use` block on the next Anthropic request. Adjacent client tests
  cover the same signed block in non-streaming responses and raw stream events.
- `sdk/tokens/cost_test.go` - initialization concurrency, cached-token clamping, warning surfacing for pricing-init/cache-read/cache-stat/cache-parse/cost-calc failures, cache-write warning behavior, and alias/family pricing lookup fallback (`sdk/tokens/cost_test.go:20`, `sdk/tokens/cost_test.go:65`, `sdk/tokens/cost_test.go:133`, `sdk/tokens/cost_test.go:223`, `sdk/tokens/cost_test.go:258`, `sdk/tokens/cost_test.go:316`)

## Practical Test Strategy
- Run area-focused tests first when editing a subsystem.
- Run `go test ./...` before finalizing cross-cutting changes.
- Use `-race` for changes involving tool dependency container concurrency (`sdk/tools/deps.go:198`).
