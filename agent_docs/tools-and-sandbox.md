# Tools and Sandbox

This document explains the tool subsystem and how it interacts with the agent loop,
provider-facing schemas, metadata propagation, and sandbox safety gates.

## Tool Core Model
- `Tool` is the runtime unit: name, description, schema, handler, visibility (`Hidden`), and retention (`EphemeralKeep`) (`sdk/tools/tool.go:15`)
- `Definition()` converts internal tools to strict provider tool definitions (`sdk/tools/tool.go:31`)
- `Execute()` is the single execution path used by the agent (`sdk/tools/tool.go:40`)
- Typed helper `Func` builds a `Tool` from typed args + handler and enforces strict decode (`sdk/tools/tool.go:701`)
- `SchemaFor` derives conservative JSON schema from Go structs (`sdk/tools/schema.go:8`)
- `SerializeResult` converts handler outputs into `llm.Content` blocks (`sdk/tools/result.go:10`)

## Argument Normalization and Repair Pipeline
1. **Raw normalization entry**
   - `NormalizeToolArgs` preprocesses tool args before decode.
   - First-object extraction now requires syntactically valid JSON (balanced braces alone are not enough).
   - References: `sdk/tools/args_normalize.go:18`, `sdk/tools/args_normalize.go:111`

2. **Loose repair and string-wrapping**
   - Supports malformed JSON-like inputs and plain-string inputs via schema-driven single-string detection (with legacy built-in mappings as fallback).
   - Loose-object repair is now schema-aware: repaired payloads are accepted only when their shape/types remain compatible with the tool schema.
   - References: `sdk/tools/args_normalize.go:64`, `sdk/tools/tool.go:398`

3. **Strict decode and second-chance schema repair**
   - `Tool.Execute` retries when decode fails due to unknown/misaligned keys.
   - Schema-key repair now recurses through nested objects, strips unknown fields on retry, and applies tool-specific alias mappings for ambiguous keys (for example `line` -> `offset` only for read-style tools).
   - References: `sdk/tools/tool.go:58`, `sdk/tools/tool.go:67`, `sdk/tools/tool.go:96`, `sdk/tools/tool.go:105`, `sdk/tools/tool.go:363`

4. **Empty-output fallback**
   - If a tool handler returns empty content with `nil` error, execution now surfaces a warning message and attaches metadata (`tool_warning`) instead of silently returning blank output.
   - Reference: `sdk/tools/tool.go:77`

5. **Repair metadata propagation**
   - Repair information is carried forward for telemetry/UI correlation.
   - References: `sdk/tools/tool.go:40`, `sdk/agent/agent.go:382`, `sdk/agent/agent.go:445`

## Name Resolution and Alias Handling
- Tool name normalization strips stacked prefixes (for example `tools.function.read` -> `read`) and symbol noise (`sdk/tools/tool_name.go:20`)
- Agent builds normalized lookup tables and alias candidates (`sdk/agent/tool_resolve.go:10`, `sdk/agent/tool_resolve.go:30`)
- Runtime lookup is exact name first, then normalized fallback (`sdk/agent/tool_resolve.go:107`)

## Agent <-> Tool Interaction Flow
- Before provider calls, agent exports non-hidden tool definitions (`sdk/agent/agent.go:232`)
- For each tool call, agent normalizes args, enriches context with call ID and metadata store, then executes (`sdk/agent/agent.go:382`, `sdk/agent/agent.go:396`)
- Tool output + metadata snapshot is emitted as `ToolResultEvent` and appended as a tool-role message (`sdk/agent/agent.go:408`, `sdk/agent/agent.go:445`)
- Oversized tool output is truncated before history append/event emission using configurable `MaxToolResultBytes` (default 50KB) with `...[truncated]` suffix; metadata includes both SDK keys (`result_truncated`, `result_original_bytes`, `result_max_bytes`, `result_output_path`) and compatibility aliases (`truncated`, `originalSize`, `outputPath`). The full untruncated payload is persisted to a temp file when truncation happens (`sdk/agent/agent.go:31`, `sdk/agent/agent.go:461`, `sdk/agent/agent.go:485`)

## Tool Result Metadata Contract
- `WithToolCallID` injects `tool_call_id` into context (`sdk/tools/deps.go:27`)
- `WithToolResultMetadata` enables per-call metadata storage (`sdk/tools/deps.go:56`)
- Metadata merges and snapshots via `UpsertToolResultMetadata` and `TakeToolResultMetadataSnapshot` (`sdk/tools/deps.go:98`, `sdk/tools/deps.go:136`)
- `Container` offers dependency provisioning with memoization and inflight de-duplication for tool handlers (`sdk/tools/deps.go:156`, `sdk/tools/deps.go:198`)

## Ephemeral Output Retention
- Tools can request bounded short-term retention using `EphemeralKeep` (`sdk/tools/tool.go:18`)
- Sandbox default toolset applies `EphemeralKeep(1)` to reduce stale context growth (`sdk/tools/sandbox/sandbox.go:335`)
- Agent prunes old ephemeral tool messages before each model call (`sdk/agent/agent.go:229`, `sdk/agent/agent.go:790`)

## Sandbox Architecture
- Sandbox resolves paths inside allowed roots and blocks escape patterns (`sdk/tools/sandbox/sandbox.go:75`, `sdk/tools/sandbox/sandbox.go:116`)
- `Confirmer` is the explicit trust boundary for risky operations (`sdk/tools/sandbox/sandbox.go:306`)
- Default toolset includes filesystem, shell, search, web fetch, patch/edit, todo, and completion tools (`sdk/tools/sandbox/sandbox.go:283`)
- Read streams line windows from an open file handle (offset/limit) and applies binary detection with BOM + extension-aware heuristics (`sdk/tools/sandbox/sandbox.go:628`, `sdk/tools/sandbox/sandbox.go:742`)
- `ls` ignore globs and `grep` `glob` filters are validated once per request; malformed patterns now fail with explicit tool-scoped errors instead of silently changing match behavior (`sdk/tools/sandbox/sandbox.go:508`, `sdk/tools/sandbox/sandbox.go:1905`)
- `glob` now surfaces matched-path `stat` failures as deterministic warnings and tool-result metadata (`has_errors`, `skipped_paths`, `skipped_path_samples`) so partial/no-match results do not masquerade as complete success (`sdk/tools/sandbox/sandbox.go:1762`, `sdk/tools/sandbox/sandbox.go:1792`)
- Grep surfaces permission-denied subtree skips and non-permission scan failures (open/read/seek/walk) as warnings instead of silently dropping paths (`sdk/tools/sandbox/sandbox.go:1854`, `sdk/tools/sandbox/sandbox.go:2052`, `sdk/tools/sandbox/sandbox.go:2064`)

## Confirmation-Gated Operations
- Network fetch confirmation (`sdk/tools/sandbox/sandbox.go:377`, `sdk/tools/sandbox/sandbox.go:411`)
- `webfetch` now reports response-body read failures with explicit partial-body diagnostics instead of silently returning incomplete content (`sdk/tools/sandbox/sandbox.go:446`)
- Shell command confirmation and timeout-guarded execution (`sdk/tools/sandbox/sandbox.go:558`, `sdk/tools/sandbox/sandbox.go:572`)
- Write/edit/multiedit/apply_patch confirmations with diff context (`sdk/tools/sandbox/sandbox.go:887`, `sdk/tools/sandbox/sandbox.go:938`, `sdk/tools/sandbox/sandbox.go:1006`, `sdk/tools/sandbox/sandbox.go:1083`)

## Safety and Fallback Behaviors
- Hidden `invalid` tool is auto-injected by `Agent.New` and used as fallback for unknown calls. `Tool.Hidden` tools stay in the execution registry but are excluded from model-visible tool definitions.
- `TaskCompleteError` is the structured termination signal used by done-style tools (`sdk/tools/task_complete.go:7`, `sdk/agent/agent.go:353`)
