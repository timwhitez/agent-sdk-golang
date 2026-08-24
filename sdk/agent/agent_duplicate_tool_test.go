package agent

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type duplicateToolModel struct {
	calls atomic.Int32
}

func (m *duplicateToolModel) Provider() string { return "stub" }
func (m *duplicateToolModel) Model() string    { return "stub" }
func (m *duplicateToolModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls.Add(1)
	return &llm.Completion{Content: llm.TextContent("unexpected")}, nil
}

func TestAgentNewRejectsDuplicateExactToolNames(t *testing.T) {
	t.Parallel()

	model := &duplicateToolModel{}
	_, err := New(Config{
		LLM: model,
		Tools: []tools.Tool{
			{Name: "lookup", Description: "query lookup"},
			{Name: "lookup", Description: "id lookup"},
		},
	})
	if err == nil {
		t.Fatal("expected duplicate tool name to be rejected")
	}
	message := err.Error()
	for _, want := range []string{`duplicate tool name "lookup"`, "positions 1 and 2"} {
		if !strings.Contains(message, want) {
			t.Fatalf("error %q does not contain %q", message, want)
		}
	}
	if got := model.calls.Load(); got != 0 {
		t.Fatalf("provider invoked %d time(s) after constructor rejection", got)
	}
}
