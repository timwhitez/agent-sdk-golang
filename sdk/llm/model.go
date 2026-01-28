package llm

import "context"

// ChatModel is the provider-agnostic interface used by the Agent.
// Implementations must support tool calling when tools are provided.
type ChatModel interface {
	Provider() string
	Model() string

	Invoke(ctx context.Context, req InvokeRequest) (*Completion, error)
}

type InvokeRequest struct {
	Messages   []Message
	Tools      []ToolDefinition
	ToolChoice ToolChoice

	// Provider-specific knobs. Keep these minimal; wire more via concrete provider configs.
	Temperature *float64
}
