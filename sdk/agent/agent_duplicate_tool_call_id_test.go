package agent

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type duplicateToolCallIDModel struct {
	calls int
}

func (m *duplicateToolCallIDModel) Provider() string { return "stub" }
func (m *duplicateToolCallIDModel) Model() string    { return "stub" }

func (m *duplicateToolCallIDModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls > 1 {
		return &llm.Completion{Content: llm.TextContent("recovered"), StopReason: "stop"}, nil
	}
	return &llm.Completion{
		ToolCalls: []llm.ToolCall{
			{ID: "sensitive-duplicate-id", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{}`}},
			{ID: " sensitive-duplicate-id ", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{}`}},
		},
		Usage:      &llm.Usage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3},
		StopReason: "tool_calls",
	}, nil
}

func TestDuplicateToolCallIDsFailBeforeAnyHandler(t *testing.T) {
	var executions atomic.Int32
	model := &duplicateToolCallIDModel{}
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

	events := collectEvents(agent.QueryStream(context.Background(), llm.TextContent("run both")))
	if got := executions.Load(); got != 0 {
		t.Fatalf("handler executions = %d, want 0", got)
	}
	var foundError, foundUsage bool
	for _, event := range events {
		switch event := event.(type) {
		case ErrorEvent:
			if event.Kind == "invalid_tool_call_block" {
				foundError = true
				if strings.Contains(event.Message, "sensitive-duplicate-id") {
					t.Fatalf("error leaked provider tool-call ID: %q", event.Message)
				}
			}
		case UsageEvent:
			foundUsage = true
		case ToolCallEvent, ToolResultEvent:
			t.Fatalf("malformed block emitted tool lifecycle event: %#v", event)
		}
	}
	if !foundError || !foundUsage {
		t.Fatalf("events missing fail-closed error or billed usage: %#v", events)
	}
	for _, message := range agent.Messages() {
		if message.Role == llm.RoleAssistant && len(message.ToolCalls) > 0 {
			t.Fatalf("malformed assistant tool block entered history: %#v", message)
		}
	}

	response, err := agent.Query(context.Background(), "recover")
	if err != nil || response != "recovered" {
		t.Fatalf("next query response=%q err=%v", response, err)
	}
}
