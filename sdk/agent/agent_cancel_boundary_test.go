package agent

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type contextIgnoringCancelBoundaryModel struct {
	calls atomic.Int32
}

type cancelBoundaryScriptModel struct {
	calls     atomic.Int32
	toolCalls []llm.ToolCall
}

type cancelOnInvokeBoundaryModel struct {
	cancel context.CancelFunc
	calls  atomic.Int32
}

func (m *cancelOnInvokeBoundaryModel) Provider() string { return "fake" }
func (m *cancelOnInvokeBoundaryModel) Model() string    { return "cancel-on-invoke" }
func (m *cancelOnInvokeBoundaryModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls.Add(1)
	m.cancel()
	return &llm.Completion{Content: llm.TextContent("must not become final")}, nil
}

func (m *cancelBoundaryScriptModel) Provider() string { return "fake" }
func (m *cancelBoundaryScriptModel) Model() string    { return "cancel-boundary-script" }
func (m *cancelBoundaryScriptModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	if m.calls.Add(1) == 1 {
		return &llm.Completion{StopReason: "tool_calls", ToolCalls: m.toolCalls}, nil
	}
	return &llm.Completion{Content: llm.TextContent("unexpected provider continuation")}, nil
}

type cancelBoundaryTerminals struct {
	canceled int
	maxIter  int
	finals   int
}

func collectCancelBoundaryTerminals(ch <-chan Event) cancelBoundaryTerminals {
	var got cancelBoundaryTerminals
	for event := range ch {
		switch event := event.(type) {
		case ErrorEvent:
			switch event.Kind {
			case "canceled":
				got.canceled++
			case "max_iterations":
				got.maxIter++
			}
		case FinalResponseEvent:
			got.finals++
		}
	}
	return got
}

func cancelBoundaryCall(id, name string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: `{}`}}
}

func (m *contextIgnoringCancelBoundaryModel) Provider() string { return "fake" }
func (m *contextIgnoringCancelBoundaryModel) Model() string    { return "cancel-boundary" }
func (m *contextIgnoringCancelBoundaryModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	call := m.calls.Add(1)
	if call == 1 {
		return &llm.Completion{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{{
			ID:   "cancel-turn-1",
			Type: "function",
			Function: llm.FunctionCall{
				Name:      "cancel_turn",
				Arguments: `{}`,
			},
		}}}, nil
	}
	return &llm.Completion{Content: llm.TextContent("provider must not be invoked")}, nil
}

func TestToolStageCancellationStopsBeforeNextProviderInvocation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	model := &contextIgnoringCancelBoundaryModel{}
	cancelTool := tools.Func[struct{}]("cancel_turn", "cancel the active turn", func(context.Context, struct{}, *tools.Container) (any, error) {
		cancel()
		return nil, errors.New("host authority became indeterminate")
	})
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{cancelTool}, MaxIterations: 4})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var canceled *ErrorEvent
	for event := range ag.QueryStream(ctx, llm.TextContent("run")) {
		if eventErr, ok := event.(ErrorEvent); ok && eventErr.Kind == "canceled" {
			copy := eventErr
			canceled = &copy
		}
	}
	if got := model.calls.Load(); got != 1 {
		t.Fatalf("provider calls = %d, want 1", got)
	}
	if canceled == nil || !strings.Contains(strings.ToLower(canceled.Message), "cancel") {
		t.Fatalf("cancellation event = %#v", canceled)
	}
}

func TestCancellationPrecedesTaskCompleteTerminal(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	model := &cancelBoundaryScriptModel{toolCalls: []llm.ToolCall{cancelBoundaryCall("done-1", "done")}}
	doneTool := tools.Func[struct{}]("done", "done", func(context.Context, struct{}, *tools.Container) (any, error) {
		cancel()
		return nil, &tools.TaskCompleteError{Message: "must not complete after cancellation"}
	})
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{doneTool}, MaxIterations: 4})
	if err != nil {
		t.Fatal(err)
	}
	got := collectCancelBoundaryTerminals(ag.QueryStream(ctx, llm.TextContent("run")))
	if model.calls.Load() != 1 || got.canceled != 1 || got.maxIter != 0 || got.finals != 0 {
		t.Fatalf("calls=%d canceled=%d max_iterations=%d finals=%d; want 1/1/0/0", model.calls.Load(), got.canceled, got.maxIter, got.finals)
	}
}

func TestRootCancellationSkipsUnstartedSiblingTools(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	model := &cancelBoundaryScriptModel{toolCalls: []llm.ToolCall{
		cancelBoundaryCall("cancel-1", "cancel_turn"),
		cancelBoundaryCall("mutate-2", "mutate_after_cancel"),
	}}
	var secondRuns atomic.Int32
	cancelTool := tools.Func[struct{}]("cancel_turn", "cancel", func(context.Context, struct{}, *tools.Container) (any, error) {
		cancel()
		return nil, errors.New("authority indeterminate")
	})
	secondTool := tools.Func[struct{}]("mutate_after_cancel", "must not run", func(context.Context, struct{}, *tools.Container) (any, error) {
		secondRuns.Add(1)
		return "mutated", nil
	})
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{cancelTool, secondTool}, MaxIterations: 4})
	if err != nil {
		t.Fatal(err)
	}
	got := collectCancelBoundaryTerminals(ag.QueryStream(ctx, llm.TextContent("run")))
	if secondRuns.Load() != 0 || model.calls.Load() != 1 || got.canceled != 1 || got.finals != 0 {
		t.Fatalf("second_runs=%d provider_calls=%d canceled=%d finals=%d; want 0/1/1/0", secondRuns.Load(), model.calls.Load(), got.canceled, got.finals)
	}
	foundSkipped := false
	for _, message := range ag.Messages() {
		if message.Role == llm.RoleTool && message.ToolCallID == "mutate-2" && message.IsError {
			foundSkipped = true
			text := strings.ToLower(message.Content.PlainText())
			if !strings.Contains(text, "cancel") {
				t.Fatalf("canceled sibling result does not identify cancellation: %q", text)
			}
			if strings.Contains(text, "task was completed") {
				t.Fatalf("canceled sibling result falsely identifies task completion: %q", text)
			}
		}
	}
	if !foundSkipped {
		t.Fatalf("canceled sibling tool topology was not closed: %#v", ag.Messages())
	}
}

func TestCancellationDuringPreProviderSteeringDrainPreventsAdmission(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	model := &cancelBoundaryScriptModel{}
	steering := make(chan SteeringMsg, 3)
	steering <- SteeringMsg{Content: "first steering"}
	steering <- SteeringMsg{Content: "second steering"}
	steering <- SteeringMsg{Content: "third steering"}
	ag, err := New(Config{LLM: model, MaxIterations: 1, EventBufferSize: 1})
	if err != nil {
		t.Fatal(err)
	}
	var got cancelBoundaryTerminals
	canceledAtBoundary := false
	for event := range ag.QueryStreamWithSteering(ctx, llm.TextContent("run"), steering) {
		switch event := event.(type) {
		case SteeringReceivedEvent:
			if !canceledAtBoundary {
				cancel()
				canceledAtBoundary = true
			}
		case ErrorEvent:
			if event.Kind == "canceled" {
				got.canceled++
			}
			if event.Kind == "max_iterations" {
				got.maxIter++
			}
		case FinalResponseEvent:
			got.finals++
		}
	}
	if !canceledAtBoundary {
		t.Fatal("did not observe steering boundary")
	}
	if model.calls.Load() != 0 || got.canceled != 1 || got.maxIter != 0 || got.finals != 0 {
		t.Fatalf("calls=%d canceled=%d max_iterations=%d finals=%d; want 0/1/0/0", model.calls.Load(), got.canceled, got.maxIter, got.finals)
	}
}

func TestContextIgnoringProviderCompletionCannotOverrideRootCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	model := &cancelOnInvokeBoundaryModel{cancel: cancel}
	ag, err := New(Config{LLM: model, MaxIterations: 1})
	if err != nil {
		t.Fatal(err)
	}
	got := collectCancelBoundaryTerminals(ag.QueryStream(ctx, llm.TextContent("run")))
	if model.calls.Load() != 1 || got.canceled != 1 || got.maxIter != 0 || got.finals != 0 {
		t.Fatalf("calls=%d canceled=%d max_iterations=%d finals=%d; want 1/1/0/0", model.calls.Load(), got.canceled, got.maxIter, got.finals)
	}
}
