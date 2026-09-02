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

type repeatedSyntheticIDContinuationModel struct {
	calls int
}

type rotatingContinuationIDModel struct {
	calls    int
	contents []llm.Content
}

func (m *rotatingContinuationIDModel) Provider() string { return "stub" }
func (m *rotatingContinuationIDModel) Model() string    { return "stub" }
func (m *rotatingContinuationIDModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	switch m.calls {
	case 1:
		return &llm.Completion{Content: m.contents[0], ToolCalls: []llm.ToolCall{{ID: "old-partial", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{"value":`}}}, StopReason: "max_tokens"}, nil
	case 2:
		return &llm.Completion{Content: m.contents[1], ToolCalls: []llm.ToolCall{{ID: "new-partial", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{"other":`}}}, StopReason: "tool_calls"}, nil
	case 3:
		return &llm.Completion{ToolCalls: []llm.ToolCall{
			{ID: "duplicate", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{}`}},
			{ID: "duplicate", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{}`}},
		}, StopReason: "tool_calls"}, nil
	default:
		return &llm.Completion{Content: llm.TextContent("recovered"), StopReason: "stop"}, nil
	}
}

func (m *repeatedSyntheticIDContinuationModel) Provider() string { return "stub" }
func (m *repeatedSyntheticIDContinuationModel) Model() string    { return "stub" }
func (m *repeatedSyntheticIDContinuationModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	switch m.calls {
	case 1:
		return &llm.Completion{ToolCalls: []llm.ToolCall{{Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{}`}}}, StopReason: "tool_calls"}, nil
	case 2:
		return &llm.Completion{Content: llm.TextContent("first done"), StopReason: "stop"}, nil
	case 3:
		return &llm.Completion{ToolCalls: []llm.ToolCall{{Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{"value":`}}}, StopReason: "max_tokens"}, nil
	case 4:
		return &llm.Completion{ToolCalls: []llm.ToolCall{
			{ID: "duplicate", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{}`}},
			{ID: "duplicate", Type: "function", Function: llm.FunctionCall{Name: "mutate", Arguments: `{}`}},
		}, StopReason: "tool_calls"}, nil
	default:
		return &llm.Completion{Content: llm.TextContent("recovered"), StopReason: "stop"}, nil
	}
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
		LLM:      model,
		Warningf: failOnToolBlockShadowWarning(t),
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

func TestDuplicateAfterContinuationKeepsOlderSyntheticIDBlock(t *testing.T) {
	var executions atomic.Int32
	model := &repeatedSyntheticIDContinuationModel{}
	agent, err := New(Config{
		LLM:      model,
		Warningf: failOnToolBlockShadowWarning(t),
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

	if response, err := agent.Query(context.Background(), "first"); err != nil || response != "first done" {
		t.Fatalf("first response=%q err=%v", response, err)
	}
	events := collectEvents(agent.QueryStream(context.Background(), llm.TextContent("start partial")))
	if got := executions.Load(); got != 1 {
		t.Fatalf("handler executions = %d, want only the completed first call", got)
	}
	foundDuplicateError := false
	for _, event := range events {
		if event, ok := event.(ErrorEvent); ok && event.Kind == "invalid_tool_call_block" {
			foundDuplicateError = true
		}
	}
	if !foundDuplicateError {
		t.Fatalf("missing duplicate rejection: %#v", events)
	}

	messages := agent.Messages()
	completedCalls, completedResults := 0, 0
	for _, message := range messages {
		if message.Role == llm.RoleAssistant {
			for _, call := range message.ToolCalls {
				if call.ID == "call_0" {
					completedCalls++
				}
			}
		}
		if message.Role == llm.RoleTool && message.ToolCallID == "call_0" {
			completedResults++
		}
	}
	if completedCalls != 1 || completedResults != 1 {
		t.Fatalf("older completed block was altered: calls=%d results=%d messages=%#v", completedCalls, completedResults, messages)
	}
	if _, changed, unexpected := repairToolCallPairsDetailed(messages); changed || unexpected {
		t.Fatalf("history still needs pairing repair: changed=%t unexpected=%t", changed, unexpected)
	}

	events = collectEvents(agent.QueryStream(context.Background(), llm.TextContent("recover")))
	response := ""
	for _, event := range events {
		if event, ok := event.(WarnEvent); ok && event.Kind == "tool_pairing_repaired" {
			t.Fatalf("recovery required outbound pairing repair: %#v", events)
		}
		if event, ok := event.(FinalResponseEvent); ok {
			response = event.Content
		}
	}
	if response != "recovered" {
		t.Fatalf("recovery response=%q events=%#v", response, events)
	}
}

func TestDuplicateAfterRotatingContinuationIDsClearsWholeEpisode(t *testing.T) {
	model := &rotatingContinuationIDModel{contents: []llm.Content{
		mustProviderStateContent(t, llm.Content{}, []llm.ProviderState{{Provider: "openai-responses", Kind: "response.output_item.v1", Data: json.RawMessage(`{"type":"function_call","call_id":"old-partial"}`)}}),
		mustProviderStateContent(t, llm.Content{}, []llm.ProviderState{{Provider: "openai-responses", Kind: "response.output_item.v1", Data: json.RawMessage(`{"type":"function_call","call_id":"new-partial"}`)}}),
	}}
	agent, err := New(Config{LLM: model, Warningf: failOnToolBlockShadowWarning(t)})
	if err != nil {
		t.Fatal(err)
	}

	events := collectEvents(agent.QueryStream(context.Background(), llm.TextContent("start partial")))
	foundDuplicateError := false
	for _, event := range events {
		if event, ok := event.(ErrorEvent); ok && event.Kind == "invalid_tool_call_block" {
			foundDuplicateError = true
		}
	}
	if !foundDuplicateError {
		t.Fatalf("missing duplicate rejection: %#v", events)
	}
	messages := agent.Messages()
	for i, message := range messages {
		if message.Role == llm.RoleAssistant && (len(message.ToolCalls) != 0 || providerStateCount(t, message.Content) != 0) {
			t.Fatalf("partial block %d survived rotating-ID cleanup: %#v", i, message)
		}
	}
	if _, changed, unexpected := repairToolCallPairsDetailed(messages); changed || unexpected {
		t.Fatalf("history still needs pairing repair: changed=%t unexpected=%t messages=%#v", changed, unexpected, messages)
	}
	if response, err := agent.Query(context.Background(), "recover"); err != nil || response != "recovered" {
		t.Fatalf("recovery response=%q err=%v", response, err)
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
