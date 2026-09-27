package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type numericRepairModel struct{ calls int }

func (*numericRepairModel) Provider() string { return "fixture" }
func (*numericRepairModel) Model() string    { return "numeric-repair" }
func (m *numericRepairModel) Invoke(_ context.Context, request llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls == 1 {
		return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "numeric-call", Type: "function", Function: llm.FunctionCall{Name: "lookup", Arguments: `{"id":9007199254740993,"extra":true}`}}}}, nil
	}
	for _, message := range request.Messages {
		if message.Role == llm.RoleTool && message.ToolCallID == "numeric-call" {
			if !strings.Contains(message.Content.PlainText(), "9007199254740993") {
				return nil, fmt.Errorf("tool result omitted exact id: %s", message.Content.PlainText())
			}
			return &llm.Completion{Content: llm.TextContent("verified")}, nil
		}
	}
	return nil, fmt.Errorf("missing tool result")
}

func TestAgentRepairPreservesNumericToolTarget(t *testing.T) {
	model := &numericRepairModel{}
	called := 0
	tool := tools.Func("lookup", "fixture", func(_ context.Context, args struct {
		ID int64 `json:"id"`
	}, _ *tools.Container) (any, error) { called++; return fmt.Sprint(args.ID), nil })
	agent, err := New(Config{LLM: model, Tools: []tools.Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	answer, err := agent.Query(context.Background(), "look up exact id")
	if err != nil || answer != "verified" || called != 1 || model.calls != 2 {
		t.Fatalf("answer=%q err=%v called=%d model calls=%d", answer, err, called, model.calls)
	}
	var original, terminal int
	for _, message := range agent.Messages() {
		if message.Role == llm.RoleAssistant && len(message.ToolCalls) == 1 && message.ToolCalls[0].ID == "numeric-call" {
			original++
			if message.ToolCalls[0].Function.Arguments != `{"id":9007199254740993,"extra":true}` {
				t.Fatalf("original call changed: %s", message.ToolCalls[0].Function.Arguments)
			}
		}
		if message.Role == llm.RoleTool && message.ToolCallID == "numeric-call" {
			terminal++
		}
	}
	if original != 1 || terminal != 1 {
		t.Fatalf("original=%d terminal=%d", original, terminal)
	}
}
