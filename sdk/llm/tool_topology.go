package llm

import (
	"fmt"
	"strings"
)

// OpenToolCallBlockStart conservatively locates the first unfinished tool block,
// or returns -1. This is shared history protection, not a full ordering/ID validator.
func OpenToolCallBlockStart(messages []Message) int {
	openStart := -1
	pending := map[string]int{}
	for i, msg := range messages {
		if msg.Role == RoleAssistant && len(msg.ToolCalls) > 0 {
			if openStart < 0 {
				openStart = i
			}
			for callIndex, call := range msg.ToolCalls {
				id := strings.TrimSpace(call.ID)
				if id == "" {
					id = fmt.Sprintf("__missing_tool_call_id_%d_%d", i, callIndex)
				}
				pending[id]++
			}
			continue
		}
		if msg.Role != RoleTool || len(pending) == 0 {
			continue
		}
		id := strings.TrimSpace(msg.ToolCallID)
		if count := pending[id]; count > 1 {
			pending[id] = count - 1
		} else {
			delete(pending, id)
		}
		if len(pending) == 0 {
			openStart = -1
		}
	}
	if len(pending) == 0 {
		return -1
	}
	return openStart
}
