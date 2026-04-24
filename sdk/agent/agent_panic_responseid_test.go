package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type doneToolCallModel struct{}

func (m *doneToolCallModel) Provider() string { return "stub" }
func (m *doneToolCallModel) Model() string    { return "stub" }

func (m *doneToolCallModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{
		ToolCalls: []llm.ToolCall{{
			ID:   "done_1",
			Type: "function",
			Function: llm.FunctionCall{
				Name:      "done",
				Arguments: `{"message":"finished"}`,
			},
		}},
		StopReason: "tool_calls",
		ResponseID: "resp_done_123",
	}, nil
}

func TestDoneToolFinalResponseCarriesResponseID(t *testing.T) {
	ag, err := New(Config{LLM: &doneToolCallModel{}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	finalResponseID := ""
	for _, ev := range events {
		if f, ok := ev.(FinalResponseEvent); ok {
			finalResponseID = f.ResponseID
		}
	}
	if finalResponseID != "resp_done_123" {
		t.Fatalf("expected final response id resp_done_123, got %q", finalResponseID)
	}
}

func TestToolPanicRecoveredIntoErrorResult(t *testing.T) {
	model := &stubModel{toolName: "explode", toolArgs: `{}`}
	panicTool := tools.Tool{
		Name:   "explode",
		Schema: tools.SchemaFor[struct{}](),
		Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
			panic("boom")
		},
	}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{panicTool}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	toolResult, ok := findToolResult(events, "explode")
	if !ok {
		t.Fatalf("expected tool result for recovered panic")
	}
	if !toolResult.IsError {
		t.Fatalf("expected recovered panic to surface as error tool result")
	}
	if !strings.Contains(toolResult.Result, `tool "explode" panicked: boom`) {
		t.Fatalf("unexpected tool result: %q", toolResult.Result)
	}
	if toolResult.Metadata == nil || toolResult.Metadata["panic"] != true {
		t.Fatalf("expected panic metadata, got %#v", toolResult.Metadata)
	}

	finalText := ""
	for _, ev := range events {
		if f, ok := ev.(FinalResponseEvent); ok {
			finalText = f.Content
		}
	}
	if strings.TrimSpace(finalText) == "" {
		t.Fatalf("expected agent to continue to a final response after recovered panic")
	}
}
