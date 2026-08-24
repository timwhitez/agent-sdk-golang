package openai

import (
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func historyToolCall(id, name string) llm.ToolCall {
	return llm.ToolCall{
		ID:   id,
		Type: "function",
		Function: llm.FunctionCall{
			Name:      name,
			Arguments: `{}`,
		},
	}
}

func historyToolResult(id string) llm.Message {
	return llm.Message{
		Role:       llm.RoleTool,
		ToolCallID: id,
		Content:    llm.TextContent("ok"),
	}
}

func TestValidateOpenAIToolHistoryRejectsDuplicateAssistantIDs(t *testing.T) {
	messages := []llm.Message{
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{
				historyToolCall("call_dup", "first"),
				historyToolCall("call_dup", "second"),
			},
		},
		historyToolResult("call_dup"),
	}

	err := validateOpenAIToolHistory(messages, "openai")
	if err == nil || !strings.Contains(err.Error(), "repeats id") {
		t.Fatalf("validateOpenAIToolHistory() error = %v, want duplicate assistant id error", err)
	}
}

func TestValidateOpenAIToolHistoryRejectsDuplicateResults(t *testing.T) {
	messages := []llm.Message{
		{
			Role:      llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{historyToolCall("call_1", "first")},
		},
		historyToolResult("call_1"),
		historyToolResult("call_1"),
	}

	err := validateOpenAIToolHistory(messages, "openai")
	if err == nil || !strings.Contains(err.Error(), "repeats tool_call_id") {
		t.Fatalf("validateOpenAIToolHistory() error = %v, want duplicate result error", err)
	}
}

func TestValidateOpenAIToolHistoryAcceptsOneResultPerParallelCall(t *testing.T) {
	messages := []llm.Message{
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{
				historyToolCall("call_1", "first"),
				historyToolCall("call_2", "second"),
			},
		},
		historyToolResult("call_2"),
		historyToolResult("call_1"),
	}

	if err := validateOpenAIToolHistory(messages, "openai"); err != nil {
		t.Fatalf("validateOpenAIToolHistory() error = %v, want nil", err)
	}
}
