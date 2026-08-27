package compaction

import (
	"encoding/json"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestPrepareForSummaryClearsProviderStateWhenToolCallsAreRemoved(t *testing.T) {
	content, err := llm.WithProviderState(llm.TextContent("partial assistant text"), []llm.ProviderState{{
		Provider: "openai-responses",
		Kind:     "response.output_item.v1",
		Data:     json.RawMessage(`{"type":"function_call","call_id":"call_1"}`),
	}})
	if err != nil {
		t.Fatal(err)
	}
	messages := []llm.Message{{
		Role:      llm.RoleAssistant,
		Content:   content,
		ToolCalls: []llm.ToolCall{{ID: "call_1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}},
	}}
	prepared := prepareForSummary(messages)
	if len(prepared) != 1 {
		t.Fatalf("summary preparation returned %d messages", len(prepared))
	}
	state, err := llm.ProviderStateFromContent(prepared[0].Content)
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared[0].ToolCalls) != 0 || len(state) != 0 {
		t.Fatalf("summary preparation retained stale provider state: %#v", prepared)
	}
}
