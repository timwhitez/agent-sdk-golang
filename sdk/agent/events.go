package agent

import (
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

// ErrorEvent is emitted when the agent hits a fatal error (e.g. provider API error).
// The stream will end after emitting this event.
type ErrorEvent struct {
	Provider   string
	StatusCode int
	Message    string
	Kind       string // "rate_limit"|"provider"|"network"|"unknown"
}

func (ErrorEvent) isEvent() {}

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
}

func (ToolResultEvent) isEvent() {}

type FinalResponseEvent struct{ Content string }

func (FinalResponseEvent) isEvent() {}

// UsageEvent is emitted after each provider invocation.
// It reflects the usage for the most recent LLM call (prompt ~= current context size).
type UsageEvent struct{ Usage llm.Usage }

func (UsageEvent) isEvent() {}

// CompactionEvent is emitted when the agent automatically compacts the conversation history.
// TriggerUsage is the usage from the LLM call that caused compaction.
type CompactionEvent struct {
	Result       compaction.Result
	TriggerUsage *llm.Usage
}

func (CompactionEvent) isEvent() {}
