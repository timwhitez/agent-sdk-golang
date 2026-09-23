# Architecture

This is a navigation and ownership guide, not a release checklist.
For exact signatures/defaults, inspect the referenced package and its tests.

## Ownership

| Component | Owns | Does not own |
|---|---|---|
| [sdk/agent](../sdk/agent) | Query admission, request Frames, invocation/recovery, tool lifecycle, event delivery | Host sessions, confirmation UI, application persistence |
| [sdk/llm](../sdk/llm) | Provider request/response serialization, streaming types, usage normalization | Host policy or tool side effects |
| [sdk/tools](../sdk/tools) | Tool invocation/argument handling, dependency context, sandbox primitives | A global concurrent scheduler |
| [sdk/agent/compaction](../sdk/agent/compaction) | Reduction/summary pipeline, ledger/checkpoint contracts | Goode wiring or host session revisions |
| [sdk/artifact](../sdk/artifact), [sdk/accounting](../sdk/accounting) | Portable artifact and measurement contracts | Host storage layout or authority |

Goode supplies application-specific adapters; the SDK does not import Goode.

## Query lifecycle

The Driver admits one active Query/manual publication operation, materializes
a request, invokes the captured model, finalizes any continuation, validates the
Tool Block, executes tools, and emits results through the existing output path.
Recovery and compaction re-enter at explicit boundaries rather than modifying a
captured Provider request in place.

Key implementation: [agent.go](../sdk/agent/agent.go),
[execution_frame.go](../sdk/agent/execution_frame.go),
[tool_block_state.go](../sdk/agent/tool_block_state.go), and
[sequential_block.go](../sdk/agent/sequential_block.go).

Native and opt-in host child blocks share the sequential executor and the same
block-state implementation. The executor invokes complete PreparedCalls and owns
start/return/terminal/commit/publication order. Adapters retain policy and output
projection; child scoped records are not automatically written to parent history.
A synchronous child scope rejects concurrent, nested or expired admission and
waits for accepted children before its parent completes. No lock is held across
a Handler. Calls are Exclusive unless the adapter sets `Parallel`: then the
host's `Plan` may declare a call Concurrent (with opaque Resource keys), and
consecutive Concurrent calls without shared keys run in one bounded wave
(`MaxWorkers`, hard-capped at `MaxBlockWorkers`). Workers only execute the
owned PreparedCall; the owner still admits, projects, commits and publishes in
model order, so out-of-order completion keeps one terminal per call. A missing,
false or panicking plan is Exclusive. The native Agent loop sets no `Parallel`
today; this is not a concurrent effect classifier or a promise about arbitrary
host goroutines.

- A Frame owns a cloned logical request and resolver definitions. It calls an
  explicit `llm.FrameModelBinder` once before invocation; all SDK retries reuse
  that configuration binding. Unknown/unsupported wrappers retain legacy handles.
  Handler closure state and transport handles are not frozen. Provider
  serialization may still transform the outgoing copy.
- Partial continuation calls are not accepted Tool Blocks. Finalized calls use
  the finalizing Frame's dispatch context; their content can have multiple sources.
- Provider Call IDs may repeat across blocks. The sequential block authority uses
  ordinal/state transitions, one history writer, and one-time publication claims.
  A claimed/persisted result is not proof that its event was delivered.
- Started-but-uncertain tool effects remain indeterminate. Unstarted calls are
  distinct. An uncooperative host handler cannot be made bounded merely by naming
  it safe; tool names/schema text are not effect or idempotence authority.

## Events and correlation

[EventEnvelope](../sdk/agent/event_envelope.go) adds Query-wide sequence, origin,
kind and observational time around the existing typed payload. The original
eventOutput owns cancellation/backpressure; scopes must not copy it or create
another event sequence. Critical delivery/drop accounting remains separate
from terminal history state.

Explicit producer sites add optional FrameID/InvokeAttempt. IDs contain no
Prompt/tool/result fingerprint. Attempts count Agent calls to the captured
Invoke/InvokeStream, not inner wrapper calls or HTTP retries. Retried requests
reuse a Frame; new logical iterations get new IDs. Retained partial usage keeps
its completed invocation association even when cancellation occurs in backoff.

Accepted native tool dispatch adds optional `ToolBlockID`, one-based
`ToolCallOrdinal`, and `ToolBlockCallCount` on explicitly associated Step,
ToolCall, ToolResult, and tool-result Accounting envelopes. The count includes
accepted tails closed only in history; this adds no events for those tails.
Zero/empty remains unknown. Provider ToolCallID reuse across completed blocks
does not reuse this block identity.

The driver creates one Frame per logical iteration and accepts at most one
final tool block in it, after continuation/admission checks. The block ID is
that finalizing `FrameID + "/tool-block"`; retries remain within the Frame and
unfinished continuations advance without acquiring a block identity. No new
sequence, allocator, planner, or history writer is introduced. This identifies
the accepted native dispatch block, not all argument provenance, delivery,
external exactly-once effects, or full child/delegation lineage. Suppressed
calls may still have this identity without executing a handler.

Unannotated host/compaction contexts remain absent. Correlation identifies
execution/finalizing context, not complete lineage or failure causation.
Additional public fields require keyed Go literals and compatible strict JSON
decoders; unavailable fields are omitted. Legacy typed payloads remain supported.

## History and compaction publication

Use checked history mutation APIs when publication matters. During an active
Query, changes must preserve non-System message identity and protected Tool
topology; rejected mutations do not apply. Independent manual compaction rejects
all competing history replacement until its publication ends.

The canonical compaction pipeline is shared by private automatic paths and
public manual/preflight entry points. Runtime-use barriers keep configuration
replacement from changing an operation's service underneath it; that barrier
is not a numeric Session/runtime revision.

[CommitCompactionHistory](../sdk/agent/compaction_publication.go) accepts the exact
source snapshot used to compute a candidate. It rejects admission, pending work
or stale content before persistence, owns candidate data across callbacks, and
publishes after acknowledgement before finalizing the deferred ledger.
A successful acknowledgement followed by cancellation is not an automatic rollback.

The checkpoint-only compatibility API does not own a later history replacement.
Content equality is not Session revision binding; external writers and prior
host Trim/snapshot work remain outside the SDK lease. No-writer publication is
memory-only, not a durable checkpoint. Live recovery dumps follow existing TTL/GC,
not history-reset deletion.

Exact pre-I/O rejection sentinels and wrapped persistence failures carry different
evidence. Keep indeterminate/unknown outcomes visible instead of treating an older
rejection as proof that a later write was safely unexecuted.

## Artifacts, cache and diagnostics

Canonical artifact references are owner-bound and validated before reuse;
integrity digests and ledger/checkpoint coverage checks protect data. Preserve
the existing codec, resolver, limits and failure paths when changing projection.
See [tools/sandbox](tools-and-sandbox.md) and the artifact package.

CachePlan intent is request-local and cloned with the request. Its presence
alone does not establish operational provider caching/capability policy; inspect
[cache_plan.go](../sdk/llm/cache_plan.go) and provider serializers before extending it.

Diagnostics and accounting should describe bounded outcomes without copying
secrets, full prompts/source/results or hidden reasoning. A Frame or model name
alone is not evidence of model failure. [Testing guidance](testing.md) maps the
relevant contract suites; historical receipts do not prove current behavior.

### RequireDone thinking-control provenance

The optional Envelope `RequestControlRelation=require_done_disable_thinking`
and `RequestControlSourceFrameID` identify only the successful reminder producer
that enabled the logical request's DisableThinking recovery mode. The source is
captured after the reminder enters history and copied into each actual Frame;
retries reuse it. Ordinary work can reset forced tool choice while retaining this
thinking mode. Steering and done/safety resets clear future control state, without
relabeling events from an already-created Frame. A new Query starts without it.
Absent fields are unreported, not proof of a complete request/content lineage.
Compaction, steering and multi-source continuation content retain separate scope.
