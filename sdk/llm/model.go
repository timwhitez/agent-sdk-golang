package llm

import "context"

// ChatModel is the provider-agnostic interface used by the Agent.
// Implementations must support tool calling when tools are provided.
type ChatModel interface {
	Provider() string
	Model() string

	Invoke(ctx context.Context, req InvokeRequest) (*Completion, error)
}

// StreamingChatModel is an optional extension interface.
// When implemented, callers can receive partial output tokens via InvokeStream.
type StreamingChatModel interface {
	ChatModel
	InvokeStream(ctx context.Context, req InvokeRequest) (<-chan StreamEvent, error)
}

// StreamEvent represents one event emitted from an InvokeStream call.
// Implementations should be resilient to partial / incremental data.
type StreamEvent interface{ isStreamEvent() }

// StreamTextDeltaEvent represents a text delta for the assistant output.
type StreamTextDeltaEvent struct{ Delta string }

func (StreamTextDeltaEvent) isStreamEvent() {}

// StreamThinkingDeltaEvent represents a thinking delta (provider-specific).
// Callers may choose to ignore it for UI purposes.
type StreamThinkingDeltaEvent struct{ Delta string }

func (StreamThinkingDeltaEvent) isStreamEvent() {}

// StreamToolCallDeltaEvent represents an incremental tool call.
// NameDelta/ArgumentsDelta may be partial chunks.
type StreamToolCallDeltaEvent struct {
	Index          int
	ID             string
	NameDelta      string
	ArgumentsDelta string
}

func (StreamToolCallDeltaEvent) isStreamEvent() {}

// StreamUsageEvent is emitted when usage is available (usually near the end).
type StreamUsageEvent struct{ Usage Usage }

func (StreamUsageEvent) isStreamEvent() {}

// StreamDoneEvent marks successful stream completion.
type StreamDoneEvent struct{}

func (StreamDoneEvent) isStreamEvent() {}

// StreamErrorEvent marks a fatal streaming error.
type StreamErrorEvent struct{ Err error }

func (StreamErrorEvent) isStreamEvent() {}

type InvokeRequest struct {
	Messages   []Message
	Tools      []ToolDefinition
	ToolChoice ToolChoice

	// Provider-specific knobs. Keep these minimal; wire more via concrete provider configs.
	Temperature *float64
}
