package agent

import (
	"encoding/json"

	sdkaccounting "github.com/timwhitez/agent-sdk-golang/sdk/accounting"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// Event is a marker interface for streamed agent execution events.
type Event interface{ isEvent() }

type TextEvent struct{ Content string }

func (TextEvent) isEvent() {}

// TextDeltaEvent is emitted when the underlying model supports true streaming.
// Delta should be appended to the current assistant output buffer.
type TextDeltaEvent struct{ Delta string }

func (TextDeltaEvent) isEvent() {}

type ThinkingEvent struct{ Content string }

func (ThinkingEvent) isEvent() {}

type ThinkingDeltaEvent struct{ Delta string }

func (ThinkingDeltaEvent) isEvent() {}

// ErrorEvent is emitted when the agent hits a fatal error (e.g. provider API error).
// The stream will end after emitting this event.
type ErrorEvent struct {
	Provider     string
	StatusCode   int
	Message      string
	RetryAfterMS int64
	Kind         string // "rate_limit"|"provider"|"network"|"timeout"|"canceled"|"auth"|"permission"|"invalid_request"|"decode"|"max_iterations"|"loop_guard"|"doom_loop"|"unknown"
	// StallRecoveries records how many automatic stream-idle recoveries were
	// applied earlier in the same turn before this terminal error surfaced.
	StallRecoveries int
}

func (ErrorEvent) isEvent() {}

// WarnEvent is emitted for non-fatal runtime warnings where the agent can continue.
type WarnEvent struct {
	Message  string
	Kind     string // "loop_guard"|"continuation"|"continuation_limit"|"early_stop"|"runtime"
	Metadata map[string]any
}

func (WarnEvent) isEvent() {}

type HiddenUserMessageEvent struct{ Content string }

func (HiddenUserMessageEvent) isEvent() {}

type StepStartEvent struct {
	StepID     string
	Title      string
	StepNumber int
}

func (StepStartEvent) isEvent() {}

type StepCompleteEvent struct {
	StepID     string
	Status     string // "completed"|"error"
	DurationMS int64
}

func (StepCompleteEvent) isEvent() {}

type ToolCallEvent struct {
	Tool        string
	Args        map[string]any
	ArgsJSON    json.RawMessage
	ArgsMeta    map[string]any
	ToolCallID  string
	DisplayName string
}

func (ToolCallEvent) isEvent() {}

type ToolResultEvent struct {
	Tool             string
	Result           string
	ToolCallID       string
	IsError          bool
	ScreenshotBase64 string
	Metadata         map[string]any
}

func (ToolResultEvent) isEvent() {}

type FinalResponseEvent struct {
	Content    string
	ResponseID string
	// Status is "complete" for normal terminal answers and "partial" when the
	// agent accepts a bounded fallback answer without satisfying the normal
	// completion gate. Reason identifies the fallback (for example
	// "require_done_safety"). Empty status from older producers means complete.
	Status string
	Reason string
	// StallRecoveries records how many automatic stream-idle recoveries were
	// applied earlier in the same turn before the final response completed.
	StallRecoveries int
	// DroppedEvents records how many events this turn were discarded because
	// the consumer could not keep up with the event channel. A non-zero value
	// means the delivered event stream (and any audit log derived from it) is
	// an incomplete view of the turn, even though history stayed consistent.
	DroppedEvents uint64
	// DroppedCriticalEvents counts the subset of DroppedEvents whose loss makes
	// the delivered stream contradict the history the agent actually mutated
	// (tool results, step lifecycle, injected messages, compaction, usage and
	// accounting). A non-zero value means the stream is not merely incomplete
	// but inconsistent, so any audit log or ledger derived from it must be
	// treated as unreliable for this turn.
	DroppedCriticalEvents uint64
}

func (FinalResponseEvent) isEvent() {}

// UsageEvent is emitted after each provider invocation.
// It reflects the usage for the most recent LLM call (prompt ~= current context size).
type UsageEvent struct {
	Usage      llm.Usage
	ResponseID string
}

func (UsageEvent) isEvent() {}

// CompactionEvent is emitted when the agent applies a compacted conversation
// history. Result carries the structured compaction telemetry. TriggerUsage is
// retained as the legacy usage snapshot from the LLM call that caused automatic
// compaction.
type CompactionEvent struct {
	Result       compaction.Result
	TriggerUsage *llm.Usage
}

func (CompactionEvent) isEvent() {}

// AccountingEvent carries a bounded, surface-neutral semantic projection for
// the immediately preceding tool-result, usage, or compaction event. Runtime
// hosts add session/run/surface identity and persistence policy.
type AccountingEvent struct {
	Payload         sdkaccounting.Payload
	CorrelationKind string
	ToolCallID      string
	ResponseID      string
	Sequence        uint64
	DurationMS      int64
}

func (AccountingEvent) isEvent() {}

// SteeringReceivedEvent is emitted when a steering message from the user
// has been incorporated into the conversation history mid-turn.
// This enables real-time steering: users can send feedback while the agent
// is working, and the agent will adjust its approach immediately.
type SteeringReceivedEvent struct {
	Content string
}

func (SteeringReceivedEvent) isEvent() {}

// AutoContinueEvent is emitted when the agent inserts an internal continuation
// prompt after a max-token truncation. It is metadata-only and should not be
// rendered as assistant text.
type AutoContinueEvent struct {
	Reason     string
	ResponseID string
}

func (AutoContinueEvent) isEvent() {}
