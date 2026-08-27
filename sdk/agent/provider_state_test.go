package agent

import (
	"encoding/json"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestToolPairRepairClearsStaleProviderState(t *testing.T) {
	call := llm.ToolCall{ID: "call_1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}
	state := []llm.ProviderState{{Provider: "openai-responses", Kind: "response.output_item.v1", Data: json.RawMessage(`{"type":"function_call","call_id":"call_1"}`)}}
	broken := []llm.Message{{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{call}, ProviderState: state}}
	repaired, changed, _ := repairToolCallPairsDetailed(broken)
	if !changed || len(repaired) != 1 || len(repaired[0].ToolCalls) != 0 || len(repaired[0].ProviderState) != 0 {
		t.Fatalf("repair retained stale provider state: %#v", repaired)
	}

	complete := append(append([]llm.Message(nil), broken...), llm.NewToolMessage("call_1", "read", llm.TextContent("ok"), false))
	repaired, changed, _ = repairToolCallPairsDetailed(complete)
	if changed || len(repaired) != 2 || len(repaired[0].ProviderState) != 1 {
		t.Fatalf("valid pair lost provider state: changed=%t messages=%#v", changed, repaired)
	}
}

func TestToolContinuationCleanupClearsStaleProviderState(t *testing.T) {
	call := llm.ToolCall{ID: "call_1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{"path":`}}
	messages := []llm.Message{{
		Role:      llm.RoleAssistant,
		ToolCalls: []llm.ToolCall{call},
		ProviderState: []llm.ProviderState{{
			Provider: "openai-responses",
			Kind:     "response.output_item.v1",
			Data:     json.RawMessage(`{"type":"function_call","call_id":"call_1"}`),
		}},
	}}
	continuation := newToolCallContinuation(2)
	continuation.addPartial(0, []llm.ToolCall{call})
	continuation.clearPartialToolCalls(messages, 1)
	if len(messages[0].ToolCalls) != 0 || len(messages[0].ProviderState) != 0 {
		t.Fatalf("continuation cleanup retained stale state: %#v", messages[0])
	}
}
