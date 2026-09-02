package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type toolPlanScriptModel struct {
	calls     int
	toolCalls []llm.ToolCall
}

func (m *toolPlanScriptModel) Provider() string { return "fixture" }
func (m *toolPlanScriptModel) Model() string    { return "tool-plan" }
func (m *toolPlanScriptModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls == 1 {
		return &llm.Completion{StopReason: "tool_calls", ToolCalls: m.toolCalls}, nil
	}
	return &llm.Completion{StopReason: "stop", Content: llm.TextContent("finished")}, nil
}

func TestShadowToolCallPlanDefaultsExclusive(t *testing.T) {
	for _, observation := range []toolPlanningObservation{
		{ordinal: 0, resolution: toolResolutionExact, args: toolArgsNormalized},
		{ordinal: 1, resolution: toolResolutionNormalizedAlias, args: toolArgsNormalized},
		{ordinal: 2, resolution: toolResolutionUnknownFallback, args: toolArgsInvalid},
	} {
		if got, want := shadowToolCallPlan(observation), (toolCallPlan{ordinal: observation.ordinal, class: toolPlanExclusive}); got != want {
			t.Fatalf("plan=%#v want %#v", got, want)
		}
	}
}

func TestObserveToolCallPlanComparesEveryFieldSafely(t *testing.T) {
	legacy := toolCallPlan{ordinal: 2, class: toolPlanExclusive}
	var warnings []string
	agent := &Agent{warningf: func(format string, args ...any) { warnings = append(warnings, fmt.Sprintf(format, args...)) }}
	agent.observeToolCallPlan(legacy, legacy)
	agent.observeToolCallPlan(legacy, toolCallPlan{ordinal: 3, class: toolPlanExclusive})
	agent.observeToolCallPlan(legacy, toolCallPlan{ordinal: 2, class: toolPlanClass(255)})
	if len(warnings) != 2 || !strings.Contains(warnings[0], "legacy_ordinal=2 shadow_ordinal=3") || !strings.Contains(warnings[1], "shadow_class=unknown") {
		t.Fatalf("warnings=%v", warnings)
	}
}

func TestToolPlanShadowObservesResolutionWithoutChangingExecution(t *testing.T) {
	const secret = "secret-tool-or-args-sentinel"
	call := func(id, name, args string) llm.ToolCall {
		return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
	}
	model := &toolPlanScriptModel{toolCalls: []llm.ToolCall{
		call("exact-1", "exact", `{}`),
		call("alias-2", "read", `{}`),
		call("invalid-3", "batch", `[secret-tool-or-args-sentinel`),
		call("unknown-4", secret, `{"secret":"secret-tool-or-args-sentinel"}`),
	}}
	var starts []string
	makeTool := func(name string) tools.Tool {
		return tools.Func[struct{}](name, name, func(context.Context, struct{}, *tools.Container) (any, error) {
			starts = append(starts, name)
			return "ok", nil
		})
	}
	agent, err := New(Config{LLM: model, Tools: []tools.Tool{makeTool("exact"), makeTool("read_file"), makeTool("batch")}})
	if err != nil {
		t.Fatal(err)
	}
	var observations []toolPlanningObservation
	var warnings []string
	agent.toolPlanShadowEvaluator = func(observation toolPlanningObservation) toolCallPlan {
		observations = append(observations, observation)
		return toolCallPlan{ordinal: observation.ordinal, class: toolPlanUnknown}
	}
	agent.warningf = func(format string, args ...any) { warnings = append(warnings, fmt.Sprintf(format, args...)) }
	collectEvents(agent.QueryStream(context.Background(), llm.TextContent("run")))

	wantObservations := []toolPlanningObservation{
		{ordinal: 0, resolution: toolResolutionExact, args: toolArgsNormalized},
		{ordinal: 1, resolution: toolResolutionNormalizedAlias, args: toolArgsNormalized},
		{ordinal: 2, resolution: toolResolutionExact, args: toolArgsInvalid},
		{ordinal: 3, resolution: toolResolutionUnknownFallback, args: toolArgsNormalized},
	}
	if !slices.Equal(observations, wantObservations) || !slices.Equal(starts, []string{"exact", "read_file"}) || model.calls != 2 {
		t.Fatalf("observations=%#v starts=%v provider_calls=%d", observations, starts, model.calls)
	}
	if len(warnings) != 4 {
		t.Fatalf("warnings=%v want four mismatches", warnings)
	}
	for _, warning := range warnings {
		if !strings.Contains(warning, "tool planner shadow mismatch") || strings.Contains(warning, secret) {
			t.Fatalf("unsafe mismatch warning: %q", warning)
		}
	}
	toolResults := 0
	for _, message := range agent.Messages() {
		if message.Role == llm.RoleTool {
			toolResults++
		}
	}
	if toolResults != 4 {
		t.Fatalf("tool results=%d want 4", toolResults)
	}
}

func TestToolPlanShadowObservesOnlyReachedMixedBlockCalls(t *testing.T) {
	starts := map[string]int{}
	makeTool := func(name string, handler func() (llm.Content, error)) tools.Tool {
		return tools.Tool{Name: name, Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
			starts[name]++
			return handler()
		}}
	}
	invalid := makeTool("invalid", func() (llm.Content, error) { return llm.TextContent("invalid"), nil })
	panicTool := makeTool("panic_tool", func() (llm.Content, error) { panic("boom") })
	errorTool := makeTool("error_tool", func() (llm.Content, error) { return llm.Content{}, errors.New("failed") })
	done := makeTool("done", func() (llm.Content, error) { return llm.Content{}, tools.TaskComplete("finished") })
	tail := makeTool("tail_tool", func() (llm.Content, error) { return llm.TextContent("must not run"), nil })
	agent, err := New(Config{LLM: mixedToolBlockModel{}, Tools: []tools.Tool{invalid, panicTool, errorTool, done, tail}})
	if err != nil {
		t.Fatal(err)
	}
	var observations []toolPlanningObservation
	agent.toolPlanShadowEvaluator = func(observation toolPlanningObservation) toolCallPlan {
		observations = append(observations, observation)
		return shadowToolCallPlan(observation)
	}
	collectEvents(agent.QueryStream(context.Background(), llm.TextContent("run")))
	want := []toolPlanningObservation{
		{ordinal: 0, resolution: toolResolutionUnknownFallback, args: toolArgsNormalized},
		{ordinal: 1, resolution: toolResolutionExact, args: toolArgsNormalized},
		{ordinal: 2, resolution: toolResolutionExact, args: toolArgsNormalized},
		{ordinal: 3, resolution: toolResolutionExact, args: toolArgsNormalized},
	}
	if !slices.Equal(observations, want) || starts["tail_tool"] != 0 {
		t.Fatalf("observations=%#v tail_starts=%d", observations, starts["tail_tool"])
	}
}

func TestToolPlanShadowCancellationAndInvalidBlockBoundaries(t *testing.T) {
	t.Run("running cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		model := &runningCancelOutcomeModel{}
		running := tools.Func[struct{}]("running", "running", func(context.Context, struct{}, *tools.Container) (any, error) {
			cancel()
			return "possibly changed", nil
		})
		tail := tools.Func[struct{}]("tail", "tail", func(context.Context, struct{}, *tools.Container) (any, error) { return "must not run", nil })
		agent, err := New(Config{LLM: model, Tools: []tools.Tool{running, tail}})
		if err != nil {
			t.Fatal(err)
		}
		var ordinals []int
		agent.toolPlanShadowEvaluator = func(observation toolPlanningObservation) toolCallPlan {
			ordinals = append(ordinals, observation.ordinal)
			return shadowToolCallPlan(observation)
		}
		collectEvents(agent.QueryStream(ctx, llm.TextContent("run")))
		if !slices.Equal(ordinals, []int{0}) {
			t.Fatalf("planned ordinals=%v want [0]", ordinals)
		}
	})

	t.Run("cancellation before first call", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		dropped := make(chan struct{})
		var sawDrop atomic.Bool
		agent, err := New(Config{
			LLM:               cancelBeforeToolStartModel{},
			Tools:             []tools.Tool{tools.Func[struct{}]("mutate", "mutate", func(context.Context, struct{}, *tools.Container) (any, error) { return "must not run", nil })},
			EventBufferSize:   1,
			EventSendTimeout:  time.Millisecond,
			EventDropLogEvery: 1,
			Warningf: func(format string, _ ...any) {
				if strings.Contains(format, "dropping agent event") && sawDrop.CompareAndSwap(false, true) {
					cancel()
					close(dropped)
				}
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		observations := 0
		agent.toolPlanShadowEvaluator = func(observation toolPlanningObservation) toolCallPlan {
			observations++
			return shadowToolCallPlan(observation)
		}
		events := agent.QueryStream(ctx, llm.TextContent("run"))
		select {
		case <-dropped:
		case <-time.After(2 * time.Second):
			t.Fatal("timeout waiting for controlled event drop")
		}
		collectEvents(events)
		if observations != 0 {
			t.Fatalf("observations=%d want 0", observations)
		}
	})

	t.Run("duplicate ids", func(t *testing.T) {
		model := &cancelBoundaryScriptModel{toolCalls: []llm.ToolCall{cancelBoundaryCall("duplicate", "echo"), cancelBoundaryCall("duplicate", "echo")}}
		agent, err := New(Config{LLM: model, Tools: []tools.Tool{tools.Func[struct{}]("echo", "echo", func(context.Context, struct{}, *tools.Container) (any, error) { return "must not run", nil })}})
		if err != nil {
			t.Fatal(err)
		}
		observations := 0
		agent.toolPlanShadowEvaluator = func(observation toolPlanningObservation) toolCallPlan {
			observations++
			return shadowToolCallPlan(observation)
		}
		collectEvents(agent.QueryStream(context.Background(), llm.TextContent("run")))
		if observations != 0 {
			t.Fatalf("observations=%d want 0", observations)
		}
	})
}

func TestToolPlanShadowObservesBeforeLegacyGuards(t *testing.T) {
	t.Run("repeat suppression", func(t *testing.T) {
		agent, _ := newRepeatedInterventionCharacterizationAgent(t, &repeatedInterventionRecordingModel{}, 3)
		observations := 0
		agent.toolPlanShadowEvaluator = func(observation toolPlanningObservation) toolCallPlan {
			observations++
			return shadowToolCallPlan(observation)
		}
		collectEvents(agent.QueryStream(context.Background(), llm.TextContent("loop")))
		if observations != 4 {
			t.Fatalf("observations=%d want 4", observations)
		}
	})

	t.Run("evidence suppression", func(t *testing.T) {
		model := &evidenceFixtureModel{}
		read := tools.Func[evidenceReadArgs]("read", "read", func(context.Context, evidenceReadArgs, *tools.Container) (any, error) { return "same block", nil })
		readAlias := read
		readAlias.Name = "read_file"
		done := tools.Func[evidenceDoneArgs]("done", "done", func(_ context.Context, args evidenceDoneArgs, _ *tools.Container) (any, error) {
			return nil, tools.TaskComplete(args.Message)
		})
		agent, err := New(Config{LLM: model, Tools: []tools.Tool{read, readAlias, done}, MaxIterations: -1, RequireDoneTool: true})
		if err != nil {
			t.Fatal(err)
		}
		observations := 0
		agent.toolPlanShadowEvaluator = func(observation toolPlanningObservation) toolCallPlan {
			observations++
			return shadowToolCallPlan(observation)
		}
		collectEvents(agent.QueryStream(context.Background(), llm.TextContent("inspect")))
		if observations != 5 {
			t.Fatalf("observations=%d want 5", observations)
		}
	})
}

var benchmarkToolCallPlan toolCallPlan

func BenchmarkShadowToolCallPlan(b *testing.B) {
	observation := toolPlanningObservation{ordinal: 3, resolution: toolResolutionUnknownFallback, args: toolArgsInvalid}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkToolCallPlan = shadowToolCallPlan(observation)
	}
}
