package llm

import (
	"context"
	"time"
)

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
// Callers may choose to ignore it for UI purposes. Providers that require
// signed thinking-block replay (Anthropic) also set Index/BlockType and emit
// SignatureDelta or Data so the agent can preserve the exact structured block
// in assistant history.
type StreamThinkingDeltaEvent struct {
	Delta          string
	Index          int
	BlockType      string
	SignatureDelta string
	Data           string
}

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
type StreamDoneEvent struct {
	StopReason string
}

func (StreamDoneEvent) isStreamEvent() {}

// StreamResponseEvent carries response-level metadata (e.g., response_id).
// It is emitted when providers surface a completed response object.
type StreamResponseEvent struct {
	ResponseID string
}

func (StreamResponseEvent) isStreamEvent() {}

// StreamRetryEvent reports a non-fatal provider retry that is about to wait.
// It lets interactive clients surface backoff progress without treating the
// transient failure as a terminal stream error.
type StreamRetryEvent struct {
	Provider   string
	StatusCode int
	Message    string
	RetryAfter time.Duration
	Attempt    int
	MaxRetries int
}

func (StreamRetryEvent) isStreamEvent() {}

// StreamErrorEvent marks a fatal streaming error.
// Provider/status/message metadata is optional and used when Err is nil.
type StreamErrorEvent struct {
	Err        error
	Provider   string
	StatusCode int
	Message    string
	RetryAfter time.Duration
}

func (StreamErrorEvent) isStreamEvent() {}

type InvokeRequest struct {
	Messages   []Message
	Tools      []ToolDefinition
	ToolChoice ToolChoice

	// Provider-specific knobs. Keep these minimal; wire more via concrete provider configs.
	Temperature *float64

	// DisableThinking asks the provider to suppress extended thinking / reasoning
	// for this single call, even when the model is otherwise configured with a
	// thinking budget. Providers that do not support extended thinking ignore it.
	// The agent loop uses this for the require-done recovery invocation: some
	// providers (Anthropic) forbid a forced tool_choice while extended thinking is
	// enabled, so the one call that must force a tool has to run without thinking.
	DisableThinking bool

	// Responses options (OpenAI Responses API). Ignored by providers that don't use it.
	Responses *ResponsesOptions
}
