package anthropic

import (
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func anthropicHistoryCall(id, name string) llm.ToolCall {
	return llm.ToolCall{
		ID:   id,
		Type: "function",
		Function: llm.FunctionCall{
			Name:      name,
			Arguments: `{}`,
		},
	}
}

func anthropicHistoryResult(id string) llm.Message {
	return llm.Message{
		Role:       llm.RoleTool,
		ToolCallID: id,
		Content:    llm.TextContent("ok"),
	}
}

func TestValidateAnthropicToolHistoryRejectsNormalizedIDCollision(t *testing.T) {
	messages := []llm.Message{
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{
				anthropicHistoryCall("call/a", "first"),
				anthropicHistoryCall("call:a", "second"),
			},
		},
		anthropicHistoryResult("call/a"),
		anthropicHistoryResult("call:a"),
	}

	err := validateAnthropicToolHistory(messages)
	if err == nil || !strings.Contains(err.Error(), "both normalize to") {
		t.Fatalf("validateAnthropicToolHistory() error = %v, want normalization collision", err)
	}
}

func TestValidateAnthropicToolHistoryRejectsDuplicateOriginalID(t *testing.T) {
	messages := []llm.Message{
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{
				anthropicHistoryCall("call_dup", "first"),
				anthropicHistoryCall("call_dup", "second"),
			},
		},
		anthropicHistoryResult("call_dup"),
	}

	err := validateAnthropicToolHistory(messages)
	if err == nil || !strings.Contains(err.Error(), "repeats id") {
		t.Fatalf("validateAnthropicToolHistory() error = %v, want duplicate call id", err)
	}
}

func TestValidateAnthropicToolHistoryRejectsDuplicateResult(t *testing.T) {
	messages := []llm.Message{
		{
			Role:      llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{anthropicHistoryCall("call_1", "first")},
		},
		anthropicHistoryResult("call_1"),
		anthropicHistoryResult("call_1"),
	}

	err := validateAnthropicToolHistory(messages)
	if err == nil || !strings.Contains(err.Error(), "repeats tool_use id") {
		t.Fatalf("validateAnthropicToolHistory() error = %v, want duplicate result", err)
	}
}

func TestSerializeAnthropicToolHistoryKeepsMatchingNormalizedWireID(t *testing.T) {
	messages := []llm.Message{
		{
			Role:      llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{anthropicHistoryCall("call/a", "first")},
		},
		anthropicHistoryResult("call/a"),
	}

	_, wire, err := serializeMessagesWithWarning(messages, nil)
	if err != nil {
		t.Fatalf("serializeMessagesWithWarning() error = %v", err)
	}
	if len(wire) != 2 || len(wire[0].Content) != 1 || len(wire[1].Content) != 1 {
		t.Fatalf("unexpected serialized history: %#v", wire)
	}
	if got := wire[0].Content[0].ID; got != "call_a" {
		t.Fatalf("tool_use id = %q, want call_a", got)
	}
	if got := wire[1].Content[0].ToolUseID; got != "call_a" {
		t.Fatalf("tool_result id = %q, want call_a", got)
	}
}
