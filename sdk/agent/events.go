package agent

// Event is a marker interface for streamed agent execution events.
type Event interface{ isEvent() }

type TextEvent struct{ Content string }
func (TextEvent) isEvent() {}

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
	StepID      string
	Title       string
	StepNumber  int
}
func (StepStartEvent) isEvent() {}

type StepCompleteEvent struct {
	StepID      string
	Status      string // "completed"|"error"
	DurationMS  int64
}
func (StepCompleteEvent) isEvent() {}

type ToolCallEvent struct {
	Tool       string
	Args       map[string]any
	ToolCallID string
	DisplayName string
}
func (ToolCallEvent) isEvent() {}

type ToolResultEvent struct {
	Tool       string
	Result     string
	ToolCallID string
	IsError    bool
	ScreenshotBase64 string
}
func (ToolResultEvent) isEvent() {}

type FinalResponseEvent struct{ Content string }
func (FinalResponseEvent) isEvent() {}
