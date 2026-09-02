package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdkaccounting "github.com/timwhitez/agent-sdk-golang/sdk/accounting"
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

type cancelBeforeToolStartModel struct{}

type runningCancelOutcomeModel struct {
	calls atomic.Int32
	mu    sync.Mutex
	reqs  []llm.InvokeRequest
}

func (m *runningCancelOutcomeModel) Provider() string { return "fake" }
func (m *runningCancelOutcomeModel) Model() string    { return "running-cancel-outcome" }
func (m *runningCancelOutcomeModel) Invoke(_ context.Context, request llm.InvokeRequest) (*llm.Completion, error) {
	cloned, err := llm.CloneInvokeRequest(request)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	m.reqs = append(m.reqs, cloned)
	m.mu.Unlock()
	if m.calls.Add(1) == 1 {
		return &llm.Completion{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{
			cancelBoundaryCall("running-1", "running"),
			cancelBoundaryCall("tail-2", "tail"),
		}}, nil
	}
	return &llm.Completion{Content: llm.TextContent("next turn")}, nil
}

func (m *runningCancelOutcomeModel) requests() []llm.InvokeRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llm.InvokeRequest(nil), m.reqs...)
}

func (cancelBeforeToolStartModel) Provider() string { return "fake" }
func (cancelBeforeToolStartModel) Model() string    { return "cancel-before-tool-start" }
func (cancelBeforeToolStartModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{
		Thinking:   "fill-buffer",
		Content:    llm.TextContent("force-drop"),
		StopReason: "tool_calls",
		ToolCalls:  []llm.ToolCall{cancelBoundaryCall("never-start", "mutate")},
	}, nil
}

type cancelAsTaskComplete struct {
	cancel context.CancelFunc
}

func (e cancelAsTaskComplete) Error() string { return "task complete" }
func (e cancelAsTaskComplete) As(target any) bool {
	taskComplete, ok := target.(**tools.TaskCompleteError)
	if !ok {
		return false
	}
	e.cancel()
	*taskComplete = &tools.TaskCompleteError{Message: "done"}
	return true
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
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{cancelTool, secondTool}, MaxIterations: 4, Warningf: failOnToolBlockShadowWarning(t)})
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

func TestRunningToolCancellationOutcomeCharacterization(t *testing.T) {
	for _, test := range []struct {
		name    string
		toolErr error
	}{
		{name: "handler success"},
		{name: "handler error", toolErr: errors.New("handler failed after side effect")},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			model := &runningCancelOutcomeModel{}
			var sideEffects, tailRuns atomic.Int32
			running := tools.Func[struct{}]("running", "running", func(context.Context, struct{}, *tools.Container) (any, error) {
				sideEffects.Add(1)
				cancel()
				return "handler result must be overwritten", test.toolErr
			})
			tail := tools.Func[struct{}]("tail", "tail", func(context.Context, struct{}, *tools.Container) (any, error) {
				tailRuns.Add(1)
				return "must not run", nil
			})
			ag, err := New(Config{LLM: model, Tools: []tools.Tool{running, tail}, MaxIterations: 4, Warningf: failOnToolBlockShadowWarning(t)})
			if err != nil {
				t.Fatal(err)
			}
			events := collectEvents(ag.QueryStream(ctx, llm.TextContent("run")))
			if sideEffects.Load() != 1 || tailRuns.Load() != 0 || model.calls.Load() != 1 {
				t.Fatalf("side_effects=%d tail_runs=%d provider_calls=%d want 1/0/1", sideEffects.Load(), tailRuns.Load(), model.calls.Load())
			}

			const runningResult = "Tool execution canceled before turn continuation: context canceled"
			const tailResult = "[ERROR] Tool call skipped because the active turn was canceled before this call ran."
			history := ag.Messages()
			assertContiguousToolResults(t, history)
			assertCancellationToolHistory(t, history, runningResult, tailResult)
			if _, changed, unexpected := repairToolCallPairsDetailed(history); changed || unexpected {
				t.Fatalf("closed canceled block required repair: changed=%t unexpected=%t", changed, unexpected)
			}
			assertRunningCancellationEvents(t, events, runningResult)

			next := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("next")))
			if model.calls.Load() != 2 || finalText(next) != "next turn" {
				t.Fatalf("next provider calls=%d final=%q", model.calls.Load(), finalText(next))
			}
			for _, event := range next {
				if warning, ok := event.(WarnEvent); ok && warning.Kind == "tool_pairing_repaired" {
					t.Fatalf("next admission repaired already-closed block: %#v", warning)
				}
			}
			requests := model.requests()
			if len(requests) != 2 {
				t.Fatalf("provider requests=%d want 2", len(requests))
			}
			assertContiguousToolResults(t, requests[1].Messages)
		})
	}
}

func assertCancellationToolHistory(t *testing.T, history []llm.Message, runningResult, tailResult string) {
	t.Helper()
	for i, message := range history {
		if message.Role != llm.RoleAssistant || len(message.ToolCalls) != 2 || message.ToolCalls[0].ID != "running-1" {
			continue
		}
		if i+2 >= len(history) {
			t.Fatalf("canceled tool block truncated: %#v", history)
		}
		running, tail := history[i+1], history[i+2]
		if running.Role != llm.RoleTool || running.ToolCallID != "running-1" || !running.IsError || running.Content.PlainText() != runningResult {
			t.Fatalf("running cancellation result=%#v", running)
		}
		if tail.Role != llm.RoleTool || tail.ToolCallID != "tail-2" || !tail.IsError || tail.Content.PlainText() != tailResult {
			t.Fatalf("unstarted tail result=%#v", tail)
		}
		return
	}
	t.Fatalf("missing canceled tool block: %#v", history)
}

func assertRunningCancellationEvents(t *testing.T, events []Event, runningResult string) {
	t.Helper()
	var order []string
	for _, event := range events {
		switch event := event.(type) {
		case StepStartEvent:
			if event.StepID == "tail-2" {
				t.Fatalf("unstarted tail emitted StepStart: %#v", event)
			}
			if event.StepID == "running-1" {
				order = append(order, "step_start")
			}
		case ToolCallEvent:
			if event.ToolCallID == "tail-2" {
				t.Fatalf("unstarted tail emitted ToolCall: %#v", event)
			}
			if event.ToolCallID == "running-1" {
				order = append(order, "tool_call")
			}
		case ToolResultEvent:
			if event.ToolCallID == "tail-2" {
				t.Fatalf("unstarted tail emitted ToolResult: %#v", event)
			}
			if event.ToolCallID == "running-1" {
				if !event.IsError || event.Result != runningResult {
					t.Fatalf("running ToolResult=%#v", event)
				}
				order = append(order, "tool_result")
			}
		case AccountingEvent:
			if event.ToolCallID == "tail-2" {
				t.Fatalf("unstarted tail emitted Accounting: %#v", event)
			}
			if event.ToolCallID == "running-1" {
				if event.CorrelationKind != "tool_call" || event.Payload.EventKind != sdkaccounting.EventKindToolResult || event.Payload.Status != sdkaccounting.StatusError {
					t.Fatalf("running Accounting=%#v", event)
				}
				if err := event.Payload.Validate(); err != nil {
					t.Fatalf("running Accounting payload: %v", err)
				}
				order = append(order, "accounting")
			}
		case StepCompleteEvent:
			if event.StepID == "tail-2" {
				t.Fatalf("unstarted tail emitted StepComplete: %#v", event)
			}
			if event.StepID == "running-1" {
				order = append(order, "step_complete")
			}
		case ErrorEvent:
			if event.Kind == "canceled" {
				order = append(order, "canceled")
			}
		case FinalResponseEvent:
			t.Fatalf("canceled turn emitted Final: %#v", event)
		}
	}
	want := []string{"step_start", "tool_call", "tool_result", "accounting", "step_complete", "canceled"}
	if strings.Join(order, ",") != strings.Join(want, ",") {
		t.Fatalf("running cancellation event order=%v want %v", order, want)
	}
}

func finalText(events []Event) string {
	for _, event := range events {
		if final, ok := event.(FinalResponseEvent); ok {
			return final.Content
		}
	}
	return ""
}

func BenchmarkToolBlockShadowLifecycle(b *testing.B) {
	ids := []string{"a", "b", "c", "d", "e", "f", "g", "h"}
	calls := make([]llm.ToolCall, len(ids))
	for i, id := range ids {
		calls[i] = cancelBoundaryCall(id, "tool")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		state := newToolBlockState(calls)
		for ordinal := range calls {
			state.markRunning(ordinal)
			state.markTerminal(ordinal, toolCallRunning, "handler_return")
		}
		if err := state.validateClosed(); err != nil {
			b.Fatal(err)
		}
	}
}

func TestRootCancellationBeforeFirstToolClosesAcceptedBlock(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var runs atomic.Int32
	dropped := make(chan struct{})
	var sawDrop atomic.Bool
	failShadow := failOnToolBlockShadowWarning(t)
	ag, err := New(Config{
		LLM:               cancelBeforeToolStartModel{},
		Tools:             []tools.Tool{tools.Func[struct{}]("mutate", "must not run", func(context.Context, struct{}, *tools.Container) (any, error) { runs.Add(1); return "mutated", nil })},
		EventBufferSize:   1,
		EventSendTimeout:  time.Millisecond,
		EventDropLogEvery: 1,
		Warningf: func(format string, args ...any) {
			failShadow(format, args...)
			if strings.Contains(format, "dropping agent event") && sawDrop.CompareAndSwap(false, true) {
				cancel()
				close(dropped)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	events := ag.QueryStream(ctx, llm.TextContent("run"))
	select {
	case <-dropped:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for controlled event drop")
	}
	got := collectCancelBoundaryTerminals(events)
	if runs.Load() != 0 || got.canceled != 1 || got.finals != 0 {
		t.Fatalf("runs=%d canceled=%d finals=%d; want 0/1/0", runs.Load(), got.canceled, got.finals)
	}
	assertCanceledToolResult(t, ag.Messages(), "never-start")
}

func TestRootCancellationAfterTaskCompleteClosesUnstartedTail(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	model := &cancelBoundaryScriptModel{toolCalls: []llm.ToolCall{
		cancelBoundaryCall("done-1", "done"),
		cancelBoundaryCall("tail-2", "tail"),
	}}
	var tailRuns atomic.Int32
	done := tools.Func[struct{}]("done", "done", func(context.Context, struct{}, *tools.Container) (any, error) {
		return nil, cancelAsTaskComplete{cancel: cancel}
	})
	tail := tools.Func[struct{}]("tail", "must not run", func(context.Context, struct{}, *tools.Container) (any, error) {
		tailRuns.Add(1)
		return "mutated", nil
	})
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{done, tail}, Warningf: failOnToolBlockShadowWarning(t)})
	if err != nil {
		t.Fatal(err)
	}
	got := collectCancelBoundaryTerminals(ag.QueryStream(ctx, llm.TextContent("run")))
	if tailRuns.Load() != 0 || got.canceled != 1 || got.finals != 0 {
		t.Fatalf("tail_runs=%d canceled=%d finals=%d; want 0/1/0", tailRuns.Load(), got.canceled, got.finals)
	}
	assertCanceledToolResult(t, ag.Messages(), "tail-2")
}

func assertCanceledToolResult(t *testing.T, messages []llm.Message, callID string) {
	t.Helper()
	for _, message := range messages {
		if message.Role == llm.RoleTool && message.ToolCallID == callID && message.IsError && strings.Contains(strings.ToLower(message.Content.PlainText()), "cancel") {
			return
		}
	}
	t.Fatalf("missing cancellation Tool Result for %q: %#v", callID, messages)
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
