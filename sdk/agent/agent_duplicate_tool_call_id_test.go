package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type duplicateToolCallIDModel struct {
	calls int
}

type streamingSyntheticIDCollisionModel struct {
	calls int
}

func (m *streamingSyntheticIDCollisionModel) Provider() string { return "stub" }
func (m *streamingSyntheticIDCollisionModel) Model() string    { return "stub" }
func (m *streamingSyntheticIDCollisionModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("non-streaming invoke should not be called")
}

func (m *streamingSyntheticIDCollisionModel) InvokeStream(context.Context, llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.calls++
	events := make(chan llm.StreamEvent, 5)
	if m.calls == 1 {
		events <- llm.StreamToolCallDeltaEvent{Index: 0, NameDelta: "mutate", ArgumentsDelta: `{}`}
		events <- llm.StreamToolCallDeltaEvent{Index: 1, ID: "call_0", NameDelta: "mutate", ArgumentsDelta: `{}`}
		events <- llm.StreamDoneEvent{StopReason: "tool_calls"}
	} else {
		events <- llm.StreamTextDeltaEvent{Delta: "recovered"}
		events <- llm.StreamDoneEvent{StopReason: "stop"}
	}
	close(events)
	return events, nil
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

func TestStreamingMissingIDDoesNotCollideWithExistingSyntheticPrefix(t *testing.T) {
	var executions atomic.Int32
	model := &streamingSyntheticIDCollisionModel{}
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

	response, err := agent.Query(context.Background(), "run both")
	if err != nil || response != "recovered" {
		t.Fatalf("response=%q err=%v", response, err)
	}
	if got := executions.Load(); got != 2 {
		t.Fatalf("handler executions = %d, want 2", got)
	}
	messages := agent.Messages()
	for i, message := range messages {
		if message.Role != llm.RoleAssistant || len(message.ToolCalls) != 2 {
			continue
		}
		if message.ToolCalls[0].ID == message.ToolCalls[1].ID {
			t.Fatalf("assistant message %d has colliding IDs: %#v", i, message.ToolCalls)
		}
		if message.ToolCalls[1].ID != "call_0" {
			t.Fatalf("explicit ID changed: %#v", message.ToolCalls)
		}
		return
	}
	t.Fatal("missing assistant tool block")
}
