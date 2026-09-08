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
[tool_block_state.go](../sdk/agent/tool_block_state.go).

- A Frame owns a cloned logical request and resolver definitions. Model/handler
  references are handles, not immutable closure state or a dynamic wrapper's
  concrete model snapshot. Provider serialization may transform the outgoing copy.
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
