package openai

import (
	"fmt"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func validateOpenAIToolHistory(messages []llm.Message, provider string) error {
	for i := 0; i < len(messages); i++ {
		m := messages[i]
		if m.Role == llm.RoleTool {
			return fmt.Errorf("%s: invalid tool history: tool message at index %d has no preceding assistant tool call", provider, i)
		}
		if m.Role != llm.RoleAssistant || len(m.ToolCalls) == 0 {
			continue
		}
		expected := make(map[string]bool, len(m.ToolCalls))
		for _, call := range m.ToolCalls {
			id := strings.TrimSpace(call.ID)
			if id == "" {
				return fmt.Errorf("%s: invalid tool history: assistant tool call at index %d has empty id", provider, i)
			}
			expected[id] = false
		}
		j := i + 1
		for j < len(messages) && messages[j].Role == llm.RoleTool {
			id := strings.TrimSpace(messages[j].ToolCallID)
			if id == "" {
				return fmt.Errorf("%s: invalid tool history: tool message at index %d has empty tool_call_id", provider, j)
			}
			if _, ok := expected[id]; !ok {
				return fmt.Errorf("%s: invalid tool history: tool message at index %d references unknown tool_call_id %q", provider, j, id)
			}
			expected[id] = true
			j++
		}
		for id, seen := range expected {
			if !seen {
				return fmt.Errorf("%s: invalid tool history: assistant tool call %q at index %d is missing a contiguous tool result", provider, id, i)
			}
		}
		i = j - 1
	}
	return nil
}
