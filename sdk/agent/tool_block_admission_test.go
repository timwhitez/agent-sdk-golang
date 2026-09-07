package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// Inject corruption after the early provider-ID check, through a synchronous
// clock callback. Constructor rejection must also discard this unaccepted block.
func TestToolBlockConstructorFailureDiscardsUnacceptedHistory(t *testing.T) {
	providers, handled := 0, 0
	completion := &llm.Completion{Content: llm.TextContent("visible before failure"), ToolCalls: []llm.ToolCall{
		{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: "{}"}},
		{ID: "b", Function: llm.FunctionCall{Name: "work", Arguments: "{}"}},
	}}
	model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
		providers++
		if providers == 1 {
			return completion, nil
		}
		return &llm.Completion{Content: llm.TextContent("next")}, nil
	}}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		handled++
		return llm.Content{}, nil
	}}}})
	if err != nil {
		t.Fatal(err)
	}
	injected := false
	ag.eventClock = func() time.Time {
		if providers == 1 && !injected {
			completion.ToolCalls[1].ID = "a"
			injected = true
		}
		return time.Unix(0, 0)
	}
	errorsSeen := 0
	for envelope := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("run")) {
		switch e := envelope.Event.(type) {
		case ErrorEvent:
			errorsSeen++
			if e.Kind != "invalid_tool_call_block" || envelope.Origin != EventOriginSDKDriver {
				t.Error("wrong admission failure")
			}
		case ToolCallEvent, ToolResultEvent, StepStartEvent, StepCompleteEvent:
			t.Errorf("unaccepted block emitted %T", e)
		}
	}
	if !injected || providers != 1 || handled != 0 || errorsSeen != 1 {
		t.Fatal("constructor failure did not stop admission")
	}
	visible := false
	for _, m := range ag.Messages() {
		if len(m.ToolCalls) != 0 || m.Role == llm.RoleTool {
			t.Fatal("constructor rejection retained invalid unaccepted topology")
		}
		visible = visible || strings.Contains(m.Content.PlainText(), "visible before failure")
	}
	if !visible {
		t.Fatal("discard lost visible text")
	}
	for e := range ag.QueryStream(context.Background(), llm.TextContent("next")) {
		if w, ok := e.(WarnEvent); ok && w.Kind == "tool_pairing_repaired" {
			t.Fatal("next query needed repair")
		}
	}
	if providers != 2 {
		t.Fatal("next query failed")
	}
}
