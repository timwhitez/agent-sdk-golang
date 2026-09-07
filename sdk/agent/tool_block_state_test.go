package agent

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestToolBlockStateTracksExecutedAndUnstartedClosure(t *testing.T) {
	block := terminalFixture(t, "run", "skip")
	block.markRunning(0)
	block.markAttemptReturned(0, false)
	acceptHistoryTerminal(t, block, 0, toolCallRunning, "handler_return")
	acceptHistoryTerminal(t, block, 1, toolCallAccepted, "turn_end")
	if err := block.validateClosed(); err != nil {
		t.Fatal(err)
	}
}

func TestToolBlockStateExecutionKnowledgeTransitions(t *testing.T) {
	block := terminalFixture(t, "observed", "indeterminate", "unstarted")
	if got := block.calls[0].executionKnowledge; got != toolExecutionNotStarted {
		t.Fatalf("accepted execution=%q want %q", got, toolExecutionNotStarted)
	}
	block.markRunning(0)
	if got := block.calls[0].executionKnowledge; got != toolExecutionAttemptStarted {
		t.Fatalf("running execution=%q want %q", got, toolExecutionAttemptStarted)
	}
	block.markAttemptReturned(0, false)
	acceptHistoryTerminal(t, block, 0, toolCallRunning, "handler_return")
	block.markRunning(1)
	block.markAttemptReturned(1, true)
	acceptHistoryTerminal(t, block, 1, toolCallRunning, "handler_return")
	acceptHistoryTerminal(t, block, 2, toolCallAccepted, "root_cancel_after_handler")

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

func TestToolBlockStateObservesOrdinaryAttemptOutcomes(t *testing.T) {
	for _, test := range []struct {
		name    string
		handler func() (any, error)
	}{
		{name: "success", handler: func() (any, error) { return "ok", nil }},
		{name: "error", handler: func() (any, error) { return nil, errors.New("failed") }},
		{name: "panic", handler: func() (any, error) { panic("failed") }},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := &cancelBoundaryScriptModel{toolCalls: []llm.ToolCall{cancelBoundaryCall("ordinary-1", "ordinary")}}
			tool := tools.Func[struct{}]("ordinary", "ordinary", func(context.Context, struct{}, *tools.Container) (any, error) {
				return test.handler()
			})
			agent, err := New(Config{LLM: model, Tools: []tools.Tool{tool}, MaxIterations: 4, Warningf: failOnToolBlockShadowWarning(t)})
			if err != nil {
				t.Fatal(err)
			}
			var observed []toolExecutionKnowledge
			agent.toolBlockStateObserved = func(block *toolBlockState) {
				for _, call := range block.calls {
					observed = append(observed, call.executionKnowledge)
				}
			}
			collectEvents(agent.QueryStream(context.Background(), llm.TextContent("run")))
			if want := []toolExecutionKnowledge{toolExecutionOutcomeObserved}; !slices.Equal(observed, want) {
				t.Fatalf("execution knowledge=%v want %v", observed, want)
			}
		})
	}
}

func TestToolBlockStateRejectsAmbiguousAcceptedIDs(t *testing.T) {
	for _, calls := range [][]llm.ToolCall{{{ID: ""}}, {{ID: "private-id"}, {ID: "private-id"}}} {
		block, err := newToolBlockState(calls)
		var typed *toolBlockTransitionError
		if block != nil || !errors.As(err, &typed) || strings.Contains(err.Error(), "private-id") {
			t.Fatalf("invalid block=%v err=%v", block, err)
		}
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
