package agent

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestToolBlockStateTracksExecutedAndUnstartedClosure(t *testing.T) {
	block := newToolBlockState([]llm.ToolCall{{ID: "run"}, {ID: "skip"}})
	block.markRunning(0)
	block.markAttemptReturned(0, false)
	block.markTerminal(0, toolCallRunning, "handler_return")
	block.markTerminal(1, toolCallAccepted, "turn_end")
	if err := block.validateClosed(); err != nil {
		t.Fatal(err)
	}
}

func TestToolBlockStateExecutionKnowledgeTransitions(t *testing.T) {
	block := newToolBlockState([]llm.ToolCall{{ID: "observed"}, {ID: "indeterminate"}, {ID: "unstarted"}})
	if got := block.calls[0].executionKnowledge; got != toolExecutionNotStarted {
		t.Fatalf("accepted execution=%q want %q", got, toolExecutionNotStarted)
	}
	block.markRunning(0)
	if got := block.calls[0].executionKnowledge; got != toolExecutionAttemptStarted {
		t.Fatalf("running execution=%q want %q", got, toolExecutionAttemptStarted)
	}
	block.markAttemptReturned(0, false)
	block.markTerminal(0, toolCallRunning, "handler_return")
	block.markRunning(1)
	block.markAttemptReturned(1, true)
	block.markTerminal(1, toolCallRunning, "handler_return")
	block.markTerminal(2, toolCallAccepted, "root_cancel_after_handler")

	if got := block.calls[0].executionKnowledge; got != toolExecutionOutcomeObserved {
		t.Fatalf("observed execution=%q want %q", got, toolExecutionOutcomeObserved)
	}
	if got := block.calls[1].executionKnowledge; got != toolExecutionIndeterminate {
		t.Fatalf("canceled running execution=%q want %q", got, toolExecutionIndeterminate)
	}
	if got := block.calls[2].executionKnowledge; got != toolExecutionNotStarted {
		t.Fatalf("unstarted execution=%q want %q", got, toolExecutionNotStarted)
	}
	if err := block.validateClosed(); err != nil {
		t.Fatal(err)
	}
}

func TestToolBlockStateObservesRunningCancellationAsIndeterminate(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	model := &runningCancelOutcomeModel{}
	running := tools.Func[struct{}]("running", "running", func(context.Context, struct{}, *tools.Container) (any, error) {
		cancel()
		return "side effect may have happened", nil
	})
	tail := tools.Func[struct{}]("tail", "tail", func(context.Context, struct{}, *tools.Container) (any, error) {
		return "must not run", nil
	})
	agent, err := New(Config{LLM: model, Tools: []tools.Tool{running, tail}, MaxIterations: 4, Warningf: failOnToolBlockShadowWarning(t)})
	if err != nil {
		t.Fatal(err)
	}
	var observed []toolExecutionKnowledge
	agent.toolBlockStateObserved = func(block *toolBlockState) {
		for _, call := range block.calls {
			observed = append(observed, call.executionKnowledge)
		}
	}
	collectEvents(agent.QueryStream(ctx, llm.TextContent("run")))

	want := []toolExecutionKnowledge{toolExecutionIndeterminate, toolExecutionNotStarted}
	if !slices.Equal(observed, want) {
		t.Fatalf("execution knowledge=%v want %v", observed, want)
	}
}

func TestToolBlockStateReportsInvariantViolations(t *testing.T) {
	tests := []struct {
		name string
		run  func(*toolBlockState)
		want string
	}{
		{name: "missing terminal", run: func(*toolBlockState) {}, want: "remains phase=\"accepted\""},
		{name: "duplicate terminal", run: func(block *toolBlockState) {
			block.markTerminal(0, toolCallAccepted, "first")
			block.markTerminal(0, toolCallTerminal, "second")
		}, want: "2 terminal transitions"},
		{name: "wrong phase", run: func(block *toolBlockState) {
			block.markTerminal(0, toolCallRunning, "handler_return")
		}, want: "want \"running\""},
		{name: "running without returned outcome", run: func(block *toolBlockState) {
			block.markRunning(0)
			block.markTerminal(0, toolCallRunning, "handler_return")
		}, want: "want terminal execution knowledge"},
		{name: "out of range", run: func(block *toolBlockState) {
			block.markRunning(1)
			block.markTerminal(0, toolCallAccepted, "turn_end")
		}, want: "outside block length"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			block := newToolBlockState([]llm.ToolCall{{ID: "call-1"}})
			test.run(block)
			err := block.validateClosed()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error=%v want substring %q", err, test.want)
			}
		})
	}
}

func TestToolBlockStateRejectsAmbiguousAcceptedIDs(t *testing.T) {
	block := newToolBlockState([]llm.ToolCall{{ID: ""}, {ID: "same"}, {ID: "same"}})
	block.markTerminalRange(0, toolCallAccepted, "closed")
	err := block.validateClosed()
	if err == nil || !strings.Contains(err.Error(), "empty id") || !strings.Contains(err.Error(), "duplicates the id") || strings.Contains(err.Error(), "same") {
		t.Fatalf("error=%v", err)
	}
}

func failOnToolBlockShadowWarning(t *testing.T) func(string, ...any) {
	t.Helper()
	return func(format string, args ...any) {
		message := fmt.Sprintf(format, args...)
		if strings.Contains(message, "tool block shadow invariant mismatch") {
			t.Errorf("unexpected shadow warning: %s", message)
		}
	}
}
