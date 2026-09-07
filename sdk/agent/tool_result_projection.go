package agent

import "github.com/timwhitez/agent-sdk-golang/sdk/llm"

// toolResultProjection is local to one result, not a second terminal authority.
// History commit and event publication keep their existing separate boundaries.
// The loop guard intentionally has a shorter event view than its history view.
type toolResultProjection struct {
	history  llm.Message
	visible  string
	original string
	metadata map[string]any
}

func projectToolResult(history llm.Message, metadata map[string]any, original string) toolResultProjection {
	return toolResultProjection{history: history, visible: history.Content.PlainText(), original: original, metadata: metadata}
}

func (p toolResultProjection) event() ToolResultEvent {
	return ToolResultEvent{Tool: p.history.ToolName, ToolCallID: p.history.ToolCallID, IsError: p.history.IsError, Result: p.visible, Metadata: p.metadata}
}
