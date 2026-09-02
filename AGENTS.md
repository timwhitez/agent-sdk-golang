# AGENTS.md

This is the entry point for agent collaboration in this repository.
It follows `AGENTSMD-guideline.md`: stateless context loading, concise instructions,
and progressive disclosure through `agent_docs/`.

## WHAT (Project Overview)

### Tech Stack
- Go 1.22 (`go.mod:3`)
- `github.com/bmatcuk/doublestar/v4` for glob matching (`go.mod:5`)

### Core Abstractions
- Agent orchestration and runtime state: `Config`, `Agent` (`sdk/agent/agent.go:20`, `sdk/agent/agent.go:39`)
- Provider abstraction: `ChatModel`, `StreamingChatModel`, `InvokeRequest`, `StreamEvent` (`sdk/llm/model.go:8`, `sdk/llm/model.go:17`, `sdk/llm/model.go:82`, `sdk/llm/model.go:24`)
- Message and tool-call model: `ToolChoice`, `ToolDefinition`, `ToolCall`, `Content`, `Message`, `Completion` (`sdk/llm/types.go:17`, `sdk/llm/types.go:25`, `sdk/llm/types.go:37`, `sdk/llm/types.go:68`, `sdk/llm/types.go:100`, `sdk/llm/types.go:133`)
- Request-local prompt-cache intent and concrete-client capability types:
  `CachePlan`, `CacheDirective`, `CacheTarget`, `PromptCacheCapabilities`, and
  `PromptCacheCapabilityProvider` (`sdk/llm/cache_plan.go`). These types are not
  yet attached to `InvokeRequest` or consumed by Provider serializers.
- Tool system primitives: `Tool`, `Func`, `SchemaFor`, `SerializeResult` (`sdk/tools/tool.go:15`, `sdk/tools/tool.go:701`, `sdk/tools/schema.go:8`, `sdk/tools/result.go:10`)
- Tool dependency context and metadata: `Container`, `WithToolCallID`, `WithToolResultMetadata` (`sdk/tools/deps.go:156`, `sdk/tools/deps.go:27`, `sdk/tools/deps.go:56`)
- Compaction service: `compaction.Service` (`sdk/agent/compaction/service.go:10`)

### Runtime Capabilities
- Boundary-aware steering before LLM calls and after each tool call, plus a root-context cancellation gate before every provider admission (`sdk/agent/agent.go`, `sdk/agent/agent_cancel_boundary_test.go`)
- Stream delta aggregation into a unified `Completion`, with response metadata propagated into `Completion.ResponseID` and surfaced on `UsageEvent` / `AutoContinueEvent` / `FinalResponseEvent` for downstream UI/aggregators (`sdk/agent/agent.go:255`, `sdk/agent/agent.go:285`, `sdk/agent/agent.go:348`, `sdk/agent/agent.go:646`)
- Versioned prompt-usage normalization (`total_input_v1`) keeps cache/image counters as breakdowns, preserves raw provider zeroes, and falls back to one SDK history estimate plus a deduplicated warning when compatible gateways omit prompt tokens (`sdk/llm/usage.go`, `sdk/agent/agent.go`).
- Versioned accounting projection emits one bounded, allowlisted
  `AccountingEvent` after each delivered tool-result, usage, and compaction
  event. The SDK payload contains surface-neutral measurements/dispositions but
  no session, path, raw result, command, or arbitrary metadata; hosts add
  identity, persistence, rollup, and reporting (`sdk/accounting`,
  `sdk/agent/events.go`).
- `QueryStreamEnveloped` preserves typed Event payloads while adding a
  versioned, per-query `QueryID`, pre-delivery monotonic sequence, stable
  kind/origin, and observational timestamp. Legacy `QueryStream` uses the same
  direct delivery/backpressure path without metadata; sequence gaps expose
  dropped events (`sdk/agent/event_envelope.go`, `sdk/agent/agent.go`).
- The sequential Tool Loop mirrors accepted/running/terminal transitions into
  a query-local, observe-only `toolBlockState`. Mismatches use `Warningf`; the
  shadow does not construct history, events, Tool Results, or Provider payloads
  (`sdk/agent/tool_block_state.go`, `sdk/agent/agent.go`).
- Auto-continue on `max_tokens` (including partial tool-call merge) now emits metadata-only `AutoContinueEvent` instead of text markers (`sdk/agent/agent.go:259`, `sdk/agent/agent.go:275`, `sdk/agent/agent.go:298`, `sdk/agent/events.go:113`)
- Tool resolution via exact + normalized + alias matching (`sdk/agent/tool_resolve.go:107`, `sdk/agent/tool_resolve.go:30`)
- Argument normalization with conservative loose-object repair + schema-key retry, including schema-derived single-string wrapping for custom tools and tool-specific alias handling for ambiguous keys like `line`/`offset` (`sdk/tools/args_normalize.go:21`, `sdk/tools/args_normalize.go:247`, `sdk/tools/tool.go:363`)
- Empty tool results are surfaced as a warning content block (instead of silent no-op) (`sdk/tools/tool.go:77`)
- Configurable byte/token tool-result budgets recognize canonical
  `sdk/artifact` envelopes, preserve fixed recovery fields while shrinking only
  preview text, and can persist oversized plain results through a host-injected
  dynamic owner provider (or static owner), sink, and registered resolver
  capability. A plain result carrying the reserved ordered
  `artifact_manifests` source metadata is persisted even below the ordinary
  size threshold, with `tool_serialize_v1` lineage to those validated source
  refs; the sink must preserve that lineage. The provider is resolved once per
  canonical validation/write so session-switching Agents do not reuse stale
  owners. Missing or failed host capability
  returns a bounded `complete=false` / `recoverable=false` diagnostic without a
  fabricated ref; the older temp dump remains metadata-only legacy
  compatibility (`sdk/agent/tool_result_artifact.go`, `sdk/agent/agent.go`).
- `execrunner` can stream stdout/stderr into separate host-owned canonical raw
  objects through `artifact.StreamSink`; each committed manifest is checked
  against collector bytes/hash/owner/recovery. Canonical mode keeps one bounded
  combined UTF-8 preview and disables the anonymous combined temp duplicate;
  begin/write/commit/manifest failures stay non-fatal to the process result and
  surface structured stage/action diagnostics
  (`sdk/tools/execrunner/canonical_stream.go`).
- Ephemeral tool-output pruning before each model invocation (`sdk/tools/tool.go:18`, `sdk/agent/agent.go:229`, `sdk/agent/agent.go:790`)
- Per-query evidence-progress recovery for deterministic `read`/`grep`/`list`
  families suppresses already-covered evidence without caching side-effecting
  tools. `read` and `read_file` share coverage while line and absolute-byte
  ranges use separate units; mixed tool-call batches remain intact, and
  compaction or target-state changes allow revalidation
  (`sdk/agent/evidence_progress.go`, `sdk/agent/agent.go`).
- Context compaction now uses cached enable/disable fast-path checks, async background compaction with next-turn atomic apply, hard overflow detection (`context_window - reserve_output_tokens`) alongside ratio thresholds, idempotent artifact-backed local reducers, comparable same-estimator before/after telemetry with provider usage kept separately, deduplicated system-message preservation (including compacted payloads), timeout/retry safeguards, model-aware summary prompts, strict summary extraction from the last `<summary>`/`<compaction_summary>` block, and rune-length-gated tool-context snapshots. When the Agent has a canonical host binding, local reducers resolve and revalidate complete owner-bound objects, persist plain source bytes once, store cloned manifests in the ledger, and reuse the same ref across prune/checkpoint cycles; embedders with only `ToolArtifactWriter` retain legacy path behavior (`sdk/agent/agent.go`, `sdk/agent/compaction/local_reduce.go`, `sdk/agent/compaction/canonical_artifact.go`, `sdk/agent/compaction/pipeline.go`, `sdk/agent/compaction/service.go`)
- Provider compatibility downgrades for OpenAI/Anthropic variants; OpenAI Chat emits `parallel_tool_calls` only when enabled, uses stricter numeric `/api/vN` endpoint detection, and Anthropic logs original+normalized tool IDs when ID sanitization occurs (`sdk/llm/openai/chat.go:119`, `sdk/llm/openai/chat.go:445`, `sdk/llm/openai/chat.go:783`, `sdk/llm/openai/responses.go:126`, `sdk/llm/anthropic/client.go:120`, `sdk/llm/anthropic/client.go:521`)
- Streaming guardrails preserve Anthropic whitespace thinking deltas, buffer malformed SSE splits, include provider/model/url context on OpenAI decode errors, and use crypto-randomized Anthropic backoff jitter (`sdk/llm/anthropic/client.go:724`, `sdk/llm/anthropic/client.go:757`, `sdk/llm/anthropic/client.go:220`, `sdk/llm/openai/chat.go:296`)
- Sandbox safety with explicit confirmation gates (`sdk/tools/sandbox/sandbox.go:116`, `sdk/tools/sandbox/sandbox.go:306`); webfetch surfaces mid-stream response-body read failures (`sdk/tools/sandbox/sandbox.go:482`), `ls`/`grep` now reject malformed glob filters with explicit tool-scoped errors, glob surfaces matched-path `stat` failures via warnings/metadata, and grep reports non-permission open/read/seek/walk scan failures as warnings instead of silently skipping files (`sdk/tools/sandbox/sandbox.go:508`, `sdk/tools/sandbox/sandbox.go:1762`, `sdk/tools/sandbox/sandbox.go:1905`, `sdk/tools/sandbox/sandbox.go:1914`, `sdk/tools/sandbox/sandbox.go:1940`, `sdk/tools/sandbox/sandbox.go:2100`)
- Optional token-cost helper with 24h pricing cache, warning logs for pricing init/cache read/cache parse/cost-calc failures (with non-fatal usage-only fallback), and broader model pricing lookup (aliases, case-insensitive names, and latest-family fallback) (`sdk/tokens/cost.go:82`, `sdk/tokens/cost.go:108`, `sdk/tokens/cost.go:126`, `sdk/tokens/cost.go:173`, `sdk/tokens/cost.go:332`)

### Project Structure (Core)
```
sdk/agent/           Agent loop, events, steering, tool execution, compaction integration
sdk/accounting/      Runtime-neutral bounded accounting schema and projectors
sdk/llm/             Provider interfaces, types, and provider clients
sdk/tools/           Tool definitions, arg normalization, schema helpers, sandbox tools
sdk/tokens/          Optional token pricing and cost utilities
agent_docs/          Detailed architecture/provider/tool/build/test documentation
```

## WHY (Design Intent)
- Keep provider integrations swappable through a unified LLM interface (`sdk/llm/model.go:8`)
- Make tool execution robust against imperfect model arguments (`sdk/tools/args_normalize.go:18`, `sdk/tools/tool.go:39`)
- Apply steering only at safe control boundaries (never mid-provider call) (`sdk/agent/agent.go:226`, `sdk/agent/agent.go:448`)
- Stay within context limits using threshold-based compaction and summaries with timeout/retry safeguards (`sdk/agent/compaction/service.go:90`, `sdk/agent/agent.go:754`)
- Enforce least-privilege filesystem and action confirmation in sandbox tools (`sdk/tools/sandbox/sandbox.go:75`, `sdk/tools/sandbox/sandbox.go:306`)

## HOW (Workflow)

### Configuration
- Configure behavior via `agent.Config` (LLM, tools, tool choice, compaction, history, dependency context) (`sdk/agent/agent.go:20`)
- `Agent.New` injects hidden `invalid` fallback tool when absent (`sdk/agent/agent.go:68`, `sdk/tools/invalid.go:15`)

### Execution APIs
- `Query(ctx, text)` for synchronous usage (`sdk/agent/agent.go:159`)
- `QueryStream(ctx, input)` and `QueryStreamWithSteering(ctx, input, steeringCh)` for event streaming (`sdk/agent/agent.go:181`, `sdk/agent/agent.go:197`)
- History and maintenance APIs: `Messages`, `ReplaceHistory`, `ClearHistory`, `NotifyTodoCompletion`, `CompactNow` (`sdk/agent/agent.go:120`, `sdk/agent/agent.go:153`, `sdk/agent/agent.go:137`, `sdk/agent/agent.go:144`, `sdk/agent/agent.go:874`)

### Validation Commands
- Build and environment setup: `agent_docs/building.md`
- Test strategy and focused commands: `agent_docs/testing.md`

## Deep Docs (Progressive Disclosure Index)
- Architecture and component interaction flow: `agent_docs/architecture.md`
- Tool system, args repair, metadata, and sandbox gates: `agent_docs/tools-and-sandbox.md`
- LLM providers, streaming normalization, and compatibility downgrades: `agent_docs/providers.md`
- Build requirements and environment knobs: `agent_docs/building.md`
- Test coverage map and recommended test commands: `agent_docs/testing.md`

## Ralph Loop 检查清单与执行指引

- 目的：将代码改进与验证流程标准化，避免“静默失败”，并通过 Roadmap 检查项逐条执行与核验。
- 清单链接：`../docs/IMPROVEMENT_ROADMAP.md`（参见“检查清单（Ralph Loop）”章节，含 `CHECK-000` 至 `CHECK-011`）。
- 执行建议：
  - 每轮先验证上一个已完成检查项（代码路径 + 相关测试），再处理当前单项，避免回归在后续轮次扩散。
  - 本仓库以 `go test` + `../scripts/validate.sh` 为最小闭环；若改动影响 Goode 运行模式，需同步校验 Headless/ACP 行为。
  - 对接 Goode 非交互输出约定时，保持结构化输出语义：Headless `--json` 为 JSONL，ACP 为 JSON-RPC；不要把纯文本诊断混入 stdout 数据流。
  - 出现写入/读取/网络/解析失败时不得静默，必须输出可操作错误或告警，并保留可恢复路径。
  - 若发现与当前任务无关的问题，按 `NEW-xxx` 模板登记到 Roadmap（`Location`/`Description`/`Fix Strategy`/`Verification`），本轮不顺手修复。

> 注：以上约定为最小文档改动，更多细节以 Roadmap 检查清单为准。
