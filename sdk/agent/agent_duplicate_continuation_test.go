package agent

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type duplicateAfterContinuationModel struct {
	calls int
}

func (m *duplicateAfterContinuationModel) Provider() string { return "stub" }
func (m *duplicateAfterContinuationModel) Model() string    { return "stub" }

func (m *duplicateAfterContinuationModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	switch m.calls {
	case 1:
		return &llm.Completion{
			ToolCalls:  []llm.ToolCall{{ID: "partial", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{"value":`}}},
			StopReason: "max_tokens",
		}, nil
	case 2:
		return &llm.Completion{
			ToolCalls: []llm.ToolCall{
				{ID: "duplicate", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{}`}},
				{ID: " duplicate ", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{}`}},
			},
			StopReason: "tool_calls",
		}, nil
	default:
		return &llm.Completion{Content: llm.TextContent("recovered"), StopReason: "stop"}, nil
	}
}

func TestDuplicateToolCallIDsAfterContinuationRemovePartialHistory(t *testing.T) {
	var executions atomic.Int32
	model := &duplicateAfterContinuationModel{}
	agent, err := New(Config{
		LLM: model,
		Tools: []tools.Tool{{
			Name: "mutate",
			Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				executions.Add(1)
				return llm.TextContent("applied"), nil
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}

	events := collectEvents(agent.QueryStream(context.Background(), llm.TextContent("continue then run")))
	if got := executions.Load(); got != 0 {
		t.Fatalf("handler executions = %d, want 0", got)
	}
	foundError := false
	for _, event := range events {
		if event, ok := event.(ErrorEvent); ok && event.Kind == "invalid_tool_call_block" {
			foundError = true
		}
	}
	if !foundError {
		t.Fatalf("missing invalid_tool_call_block error: %#v", events)
	}
	for _, message := range agent.Messages() {
		if message.Role == llm.RoleAssistant && len(message.ToolCalls) > 0 {
			t.Fatalf("partial assistant tool block remained in history: %#v", message)
		}
	}

	response, err := agent.Query(context.Background(), "recover")
	if err != nil || response != "recovered" {
		t.Fatalf("next query response=%q err=%v", response, err)
	}
}
