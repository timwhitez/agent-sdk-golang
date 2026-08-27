package compaction

import (
	"encoding/json"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestPrepareForSummaryClearsProviderStateWhenToolCallsAreRemoved(t *testing.T) {
	messages := []llm.Message{{
		Role:      llm.RoleAssistant,
		Content:   llm.TextContent("partial assistant text"),
		ToolCalls: []llm.ToolCall{{ID: "call_1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}},
		ProviderState: []llm.ProviderState{{
			Provider: "openai-responses",
			Kind:     "response.output_item.v1",
			Data:     json.RawMessage(`{"type":"function_call","call_id":"call_1"}`),
		}},
	}}
	prepared := prepareForSummary(messages)
	if len(prepared) != 1 || len(prepared[0].ToolCalls) != 0 || len(prepared[0].ProviderState) != 0 {
		t.Fatalf("summary preparation retained stale provider state: %#v", prepared)
	}
}
