package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// issue153RecordingModel issues one ambiguous tool call, then records the
// follow-up request so the test can inspect what the model sees.
type issue153RecordingModel struct {
	calls    int
	requests []llm.InvokeRequest
	call     llm.ToolCall
}

func (m *issue153RecordingModel) Provider() string { return "fixture" }
func (m *issue153RecordingModel) Model() string    { return "issue153" }
func (m *issue153RecordingModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	m.requests = append(m.requests, req)
	if m.calls == 1 {
		return &llm.Completion{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{m.call}}, nil
	}
	return &llm.Completion{StopReason: "stop", Content: llm.TextContent("done")}, nil
}

// An ambiguous alias repair is rejected by the real Func adapter: the business
// handler never runs, the accepted call is closed exactly once with an error
// result, provider arguments and CallID are preserved, and the next request
// carries the correctable error without pairing repair.
func TestIssue153AgentClosesAmbiguousCallOnce(t *testing.T) {
	type args struct {
		Content string `json:"content"`
	}
	business := 0
	tool := tools.Func[args]("write_note", "fixture", func(context.Context, args, *tools.Container) (any, error) {
		business++
		return "ok", nil
	})
	const raw = `{"text":"A","body":"B"}`
	model := &issue153RecordingModel{call: llm.ToolCall{ID: "call_ambiguous", Type: "function", Function: llm.FunctionCall{Name: "write_note", Arguments: raw}}}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{tool}, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	var states []toolCallState
	ag.toolBlockStateObserved = func(block *toolBlockState) { states = append(states, block.calls...) }
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))

	if business != 0 {
		t.Fatalf("business handler ran %d times for ambiguous arguments", business)
	}
	results := 0
	for _, event := range events {
		if result, ok := event.(ToolResultEvent); ok {
			results++
			if !result.IsError || !strings.Contains(result.Result, "ambiguous tool arguments") || result.ToolCallID != "call_ambiguous" {
				t.Fatalf("result=%+v", result)
			}
			if kind, _ := result.Metadata["args_repair_kind"].(string); strings.Contains(kind, "schema_key") {
				t.Fatalf("rejected repair reported schema_key success: %v", result.Metadata)
			}
		}
	}
	if results != 1 {
		t.Fatalf("tool results=%d want 1", results)
	}
	// Execute returned (rejecting before business); knowledge stays the real
	// outcome_observed rather than a fabricated not_started.
	if len(states) != 1 || states[0].executionKnowledge != toolExecutionOutcomeObserved || states[0].terminalCount != 1 {
		t.Fatalf("terminal states=%+v", states)
	}
	historyResults := 0
	for _, message := range ag.Messages() {
		if len(message.ToolCalls) > 0 && (message.ToolCalls[0].Function.Arguments != raw || message.ToolCalls[0].ID != "call_ambiguous") {
			t.Fatalf("history changed the provider call: %+v", message.ToolCalls[0])
		}
		if message.Role == llm.RoleTool {
			historyResults++
		}
	}
	if historyResults != 1 {
		t.Fatalf("history tool results=%d want 1", historyResults)
	}
	assertContiguousToolResults(t, ag.Messages())

	if model.calls != 2 || len(model.requests) != 2 {
		t.Fatalf("provider calls=%d", model.calls)
	}
	sawError := false
	for _, message := range model.requests[1].Messages {
		if message.Role == llm.RoleTool && message.ToolCallID == "call_ambiguous" && message.IsError &&
			strings.Contains(message.Content.PlainText(), "exact schema field names") {
			sawError = true
		}
	}
	if !sawError {
		t.Fatal("next request did not carry the correctable ambiguity error")
	}
}
