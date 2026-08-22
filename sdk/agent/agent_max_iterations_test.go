package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type maxIterationsModel struct {
	calls int
}

type maxIterationsResponseIDModel struct {
	calls int
}

func (m *maxIterationsModel) Provider() string { return "stub" }
func (m *maxIterationsModel) Model() string    { return "stub" }

func (m *maxIterationsModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	// 返回一个工具调用以保持在 requireDone 模式下仍持续迭代，便于触发最大迭代错误
	return &llm.Completion{
		Content:    llm.TextContent("keep going"),
		StopReason: "stop",
		ToolCalls: []llm.ToolCall{
			{ID: "call_1", Type: "function", Function: llm.FunctionCall{Name: "nonexistent", Arguments: "{}"}},
		},
	}, nil
}

func (m *maxIterationsResponseIDModel) Provider() string { return "stub" }
func (m *maxIterationsResponseIDModel) Model() string    { return "stub" }
func (m *maxIterationsResponseIDModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	respID := ""
	if m.calls == 1 {
		respID = "resp-first"
	}
	return &llm.Completion{
		StopReason: "tool_calls",
		ResponseID: respID,
		ToolCalls: []llm.ToolCall{{
			ID:       "call_1",
			Type:     "function",
			Function: llm.FunctionCall{Name: "nonexistent", Arguments: "{}"},
		}},
	}, nil
}

// boundedToolThenDoneModel issues tool calls for the first N turns, then calls
// done. Used to prove an unlimited (negative MaxIterations) agent keeps looping
// past the legacy default cap without emitting a max_iterations error.
type boundedToolThenDoneModel struct {
	calls    int
	toolRuns int
}

func (m *boundedToolThenDoneModel) Provider() string { return "stub" }
func (m *boundedToolThenDoneModel) Model() string    { return "stub" }
func (m *boundedToolThenDoneModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls <= m.toolRuns {
		return &llm.Completion{
			StopReason: "tool_calls",
			ToolCalls: []llm.ToolCall{
				{ID: "echo_1", Type: "function", Function: llm.FunctionCall{Name: "echo", Arguments: `{"message":"go"}`}},
			},
		}, nil
	}
	return &llm.Completion{
		StopReason: "stop",
		ToolCalls: []llm.ToolCall{
			{ID: "done_1", Type: "function", Function: llm.FunctionCall{Name: "done", Arguments: `{"message":"finished"}`}},
		},
	}, nil
}

func TestNegativeMaxIterationsRunsUnbounded(t *testing.T) {
	echoTool := tools.Func[struct {
		Message string `json:"message"`
	}]("echo", "echo", func(_ context.Context, _ struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return "ok", nil
	})
	doneTool := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "complete task", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})

	// 250 tool turns exceeds the legacy default cap of 200.
	model := &boundedToolThenDoneModel{toolRuns: 250}
	ag, err := New(Config{
		LLM:             model,
		Tools:           []tools.Tool{echoTool, doneTool},
		MaxIterations:   -1,
		RequireDoneTool: true,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
	var final string
	for _, ev := range events {
		switch e := ev.(type) {
		case ErrorEvent:
			if e.Kind == "max_iterations" {
				t.Fatalf("unlimited agent should not emit max_iterations, got %#v", e)
			}
		case FinalResponseEvent:
			final = e.Content
		}
	}
	if model.calls != model.toolRuns+1 {
		t.Fatalf("expected %d model calls, got %d", model.toolRuns+1, model.calls)
	}
	if strings.TrimSpace(final) != "finished" {
		t.Fatalf("expected completion via done tool, got %q", final)
	}
}

func TestAgentEmitsErrorOnMaxIterations(t *testing.T) {
	model := &maxIterationsModel{}
	ag, err := New(Config{LLM: model, MaxIterations: 1, RequireDoneTool: true})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	if model.calls != 1 {
		t.Fatalf("expected 1 model call, got %d", model.calls)
	}

	var finalSeen bool
	var errSeen bool
	var errEvent ErrorEvent
	for _, ev := range events {
		switch e := ev.(type) {
		case FinalResponseEvent:
			finalSeen = true
			if !strings.Contains(strings.ToLower(e.Content), "max iterations reached") {
				t.Fatalf("expected final response to mention max iterations, got %q", e.Content)
			}
		case ErrorEvent:
			errSeen = true
			errEvent = e
		}
	}
	if !finalSeen {
		t.Fatalf("expected final response event")
	}
	if !errSeen {
		t.Fatalf("expected error event on max iterations")
	}
	if errEvent.Kind != "max_iterations" {
		t.Fatalf("expected error kind max_iterations, got %q", errEvent.Kind)
	}
	if !strings.Contains(strings.ToLower(errEvent.Message), "max iterations reached") {
		t.Fatalf("expected error message to mention max iterations, got %q", errEvent.Message)
	}
}

// TestRequireDoneToolWithoutToolCalls: when RequireDoneTool=true and the model
// never calls any tools, the run should terminate in a single turn — no done
// reminders, no safety valve, just a natural text-only completion.
type textOnlyModel struct{ calls int }

func (m *textOnlyModel) Provider() string { return "stub" }
func (m *textOnlyModel) Model() string    { return "stub" }
func (m *textOnlyModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	return &llm.Completion{
		Content:    llm.TextContent("Here is the answer you requested"),
		StopReason: "stop",
		ToolCalls:  nil, // no tool calls
	}, nil
}

func TestMaxIterationsPreservesLastNonEmptyResponseID(t *testing.T) {
	model := &maxIterationsResponseIDModel{}
	ag, err := New(Config{LLM: model, MaxIterations: 2, RequireDoneTool: true})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	finalResponseID := ""
	for _, ev := range events {
		if e, ok := ev.(FinalResponseEvent); ok {
			finalResponseID = e.ResponseID
		}
	}
	if finalResponseID != "resp-first" {
		t.Fatalf("expected max-iterations final response to preserve last non-empty response id, got %q", finalResponseID)
	}
}

func TestRequireDoneToolWithoutToolCalls(t *testing.T) {
	model := &textOnlyModel{}

	ag, err := New(Config{
		LLM:             model,
		MaxIterations:   20,
		RequireDoneTool: true,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hello")))

	// Single turn: no tools ever called → text-only accepted immediately.
	if model.calls != 1 {
		t.Fatalf("expected 1 model call (single-turn), got %d", model.calls)
	}

	var finalSeen bool
	for _, ev := range events {
		switch e := ev.(type) {
		case FinalResponseEvent:
			finalSeen = true
			if e.Content != "Here is the answer you requested" {
				t.Fatalf("unexpected final content: %q", e.Content)
			}
		case WarnEvent:
			if e.Kind == "require_done_safety" {
				t.Fatalf("should not trigger safety valve for pure text Q&A")
			}
		case ErrorEvent:
			if e.Kind == "max_iterations" {
				t.Fatalf("should not hit max_iterations for pure text Q&A")
			}
		}
	}
	if !finalSeen {
		t.Fatalf("expected FinalResponseEvent but none received")
	}
}

type doneToolModel struct{}

func (m *doneToolModel) Provider() string { return "stub" }
func (m *doneToolModel) Model() string    { return "stub" }
func (m *doneToolModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{
		StopReason: "stop",
		ToolCalls: []llm.ToolCall{
			{
				ID:   "done_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "done",
					Arguments: `{"message":"finished via done tool"}`,
				},
			},
		},
	}, nil
}

type earlyStopReminderModel struct {
	calls int
}

func (m *earlyStopReminderModel) Provider() string { return "stub" }
func (m *earlyStopReminderModel) Model() string    { return "stub" }

func (m *earlyStopReminderModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	switch m.calls {
	case 1:
		return &llm.Completion{
			StopReason: "tool_calls",
			ToolCalls: []llm.ToolCall{
				{
					ID:   "echo_1",
					Type: "function",
					Function: llm.FunctionCall{
						Name:      "echo",
						Arguments: `{"message":"working"}`,
					},
				},
			},
		}, nil
	case 2:
		return &llm.Completion{
			Content:    llm.TextContent("looks done"),
			StopReason: "stop",
			ResponseID: "resp-looks-done",
		}, nil
	default:
		return &llm.Completion{
			StopReason: "tool_calls",
			ToolCalls: []llm.ToolCall{
				{
					ID:   "done_1",
					Type: "function",
					Function: llm.FunctionCall{
						Name:      "done",
						Arguments: `{"message":"done after reminder"}`,
					},
				},
			},
		}, nil
	}
}

func TestRequireDoneToolCompletesWhenDoneToolRuns(t *testing.T) {
	doneTool := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "complete task", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})

	ag, err := New(Config{
		LLM:             &doneToolModel{},
		Tools:           []tools.Tool{doneTool},
		MaxIterations:   3,
		RequireDoneTool: true,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("finish now")))
	var finalSeen bool
	for _, ev := range events {
		switch e := ev.(type) {
		case ErrorEvent:
			if e.Kind == "max_iterations" {
				t.Fatalf("did not expect max_iterations error, got %#v", e)
			}
		case FinalResponseEvent:
			finalSeen = true
			if strings.TrimSpace(e.Content) != "finished via done tool" {
				t.Fatalf("unexpected final response %q", e.Content)
			}
		}
	}
	if !finalSeen {
		t.Fatalf("expected FinalResponseEvent")
	}
}

func TestEarlyStopReminderRunsWithoutTodoDependency(t *testing.T) {
	echoTool := tools.Func[struct {
		Message string `json:"message"`
	}]("echo", "echo", func(_ context.Context, _ struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return "ok", nil
	})
	doneTool := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "complete task", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})

	model := &earlyStopReminderModel{}
	ag, err := New(Config{
		LLM:   model,
		Tools: []tools.Tool{echoTool, doneTool},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
	if model.calls != 3 {
		t.Fatalf("expected early-stop reminder to request one extra turn, got %d calls", model.calls)
	}

	earlyStopWarn := 0
	final := ""
	finalStatus := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case WarnEvent:
			if e.Kind == "early_stop" {
				earlyStopWarn++
			}
		case FinalResponseEvent:
			final = e.Content
			finalStatus = e.Status
		}
	}
	if earlyStopWarn != 1 {
		t.Fatalf("expected one early_stop warning, got %d", earlyStopWarn)
	}
	if strings.TrimSpace(final) != "done after reminder" {
		t.Fatalf("unexpected final response: %q", final)
	}
	if finalStatus != "complete" {
		t.Fatalf("early-stop recovery final status = %q, want complete", finalStatus)
	}
	assertHistoryContainsNamedUserMessage(t, ag.Messages(), earlyStopReminderText, "sdk_internal_early_stop")
}

func TestRequireDoneReminderPreservesPriorAnswerOnDoneToolCompletion(t *testing.T) {
	echoTool := tools.Func[struct {
		Message string `json:"message"`
	}]("echo", "echo", func(_ context.Context, _ struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return "ok", nil
	})
	doneTool := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "complete task", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})

	model := &earlyStopReminderModel{}
	ag, err := New(Config{
		LLM:             model,
		Tools:           []tools.Tool{echoTool, doneTool},
		MaxIterations:   4,
		RequireDoneTool: true,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
	if model.calls != 3 {
		t.Fatalf("expected require-done reminder to request one extra turn, got %d calls", model.calls)
	}

	final := ""
	finalResponseID := ""
	for _, ev := range events {
		if e, ok := ev.(FinalResponseEvent); ok {
			final = e.Content
			finalResponseID = e.ResponseID
		}
	}
	if strings.TrimSpace(final) != "looks done" {
		t.Fatalf("expected preserved prior answer, got %q", final)
	}
	if finalResponseID != "resp-looks-done" {
		t.Fatalf("expected preserved prior response id, got %q", finalResponseID)
	}
	assertHistoryContainsNamedUserMessage(t, ag.Messages(), requireDoneReminderText, "sdk_internal_require_done")
}

// requireDoneSafetyValveModel: first call uses a tool, subsequent calls are text-only.
// This simulates a model that used tools but then stops with text instead of calling done.
type requireDoneSafetyValveModel struct{ calls int }

func (m *requireDoneSafetyValveModel) Provider() string { return "stub" }
func (m *requireDoneSafetyValveModel) Model() string    { return "stub" }
func (m *requireDoneSafetyValveModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	switch m.calls {
	case 1:
		return &llm.Completion{
			StopReason: "tool_calls",
			ToolCalls: []llm.ToolCall{
				{ID: "echo_1", Type: "function", Function: llm.FunctionCall{Name: "echo", Arguments: `{"message":"hi"}`}},
			},
		}, nil
	case 2:
		return &llm.Completion{
			Content:    llm.TextContent("first post-tool answer"),
			StopReason: "stop",
			ResponseID: "resp-first",
		}, nil
	case 3:
		return &llm.Completion{
			Content:    llm.TextContent("reminder answer 2"),
			StopReason: "stop",
			ResponseID: "resp-second",
		}, nil
	default:
		return &llm.Completion{
			Content:    llm.TextContent("reminder answer 3"),
			StopReason: "stop",
			ResponseID: "resp-third",
		}, nil
	}
}

// TestRequireDoneSafetyValveAfterToolUsage: when tools were used but model
// stops with text-only, the safety valve should fire after N reminders.
func TestRequireDoneSafetyValveAfterToolUsage(t *testing.T) {
	echoTool := tools.Func[struct {
		Message string `json:"message"`
	}]("echo", "echo", func(_ context.Context, _ struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return "ok", nil
	})

	model := &requireDoneSafetyValveModel{}
	ag, err := New(Config{
		LLM:             model,
		Tools:           []tools.Tool{echoTool},
		MaxIterations:   20,
		RequireDoneTool: true,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("do something")))

	// 1 tool call + (maxReminders+1) text-only = 1 + 3 = 4
	expectedCalls := 1 + defaultRequireDoneMaxReminders + 1
	if model.calls != expectedCalls {
		t.Fatalf("expected %d model calls, got %d", expectedCalls, model.calls)
	}

	var finalSeen bool
	var finalContent string
	var finalResponseID string
	var finalStatus string
	var finalReason string
	var safetyWarnSeen bool
	for _, ev := range events {
		switch e := ev.(type) {
		case WarnEvent:
			if e.Kind == "require_done_safety" {
				safetyWarnSeen = true
			}
		case FinalResponseEvent:
			finalSeen = true
			finalContent = e.Content
			finalResponseID = e.ResponseID
			finalStatus = e.Status
			finalReason = e.Reason
		case ErrorEvent:
			if e.Kind == "max_iterations" {
				t.Fatalf("safety valve should fire before max_iterations")
			}
		}
	}
	if !finalSeen {
		t.Fatalf("expected FinalResponseEvent")
	}
	if strings.TrimSpace(finalContent) != "reminder answer 3" {
		t.Fatalf("expected safety valve to preserve latest post-tool answer, got %q", finalContent)
	}
	if finalResponseID != "resp-third" {
		t.Fatalf("expected safety valve to preserve latest post-tool response_id, got %q", finalResponseID)
	}
	if finalStatus != "partial" || finalReason != "require_done_safety" {
		t.Fatalf("safety final status/reason = %q/%q, want partial/require_done_safety", finalStatus, finalReason)
	}
	if !safetyWarnSeen {
		t.Fatalf("expected require_done_safety WarnEvent after tool usage")
	}
}

type requireDoneToolChoiceRecoveryModel struct {
	calls                       int
	recoveryToolChoice          llm.ToolChoice
	recoveryDisableThinking     bool
	postRecoveryToolChoice      llm.ToolChoice
	postRecoveryDisableThinking bool
}

func (m *requireDoneToolChoiceRecoveryModel) Provider() string { return "stub" }
func (m *requireDoneToolChoiceRecoveryModel) Model() string    { return "stub" }
func (m *requireDoneToolChoiceRecoveryModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	switch m.calls {
	case 1:
		return &llm.Completion{
			StopReason: "tool_calls",
			ToolCalls: []llm.ToolCall{{
				ID:   "echo_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "echo",
					Arguments: `{"message":"first step"}`,
				},
			}},
		}, nil
	case 2:
		return &llm.Completion{
			Content:    llm.TextContent(""),
			StopReason: "stop",
		}, nil
	case 3:
		m.recoveryToolChoice = req.ToolChoice
		m.recoveryDisableThinking = req.DisableThinking
		return &llm.Completion{
			StopReason: "tool_calls",
			ToolCalls: []llm.ToolCall{{
				ID:   "echo_2",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "echo",
					Arguments: `{"message":"continued work"}`,
				},
			}},
		}, nil
	default:
		m.postRecoveryToolChoice = req.ToolChoice
		m.postRecoveryDisableThinking = req.DisableThinking
		return &llm.Completion{
			StopReason: "tool_calls",
			ToolCalls: []llm.ToolCall{{
				ID:   "done_1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "done",
					Arguments: `{"message":"finished after continued tool work"}`,
				},
			}},
		}, nil
	}
}

func TestRequireDoneEmptyStopForcesToolChoiceAndContinues(t *testing.T) {
	echoTool := tools.Func[struct {
		Message string `json:"message"`
	}]("echo", "echo", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return args.Message, nil
	})
	doneTool := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "complete task", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})

	model := &requireDoneToolChoiceRecoveryModel{}
	ag, err := New(Config{
		LLM:             model,
		Tools:           []tools.Tool{echoTool, doneTool},
		ToolChoice:      llm.ToolChoice("auto"),
		MaxIterations:   10,
		RequireDoneTool: true,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("continue until complete")))
	if model.calls != 4 {
		t.Fatalf("model calls = %d, want 4", model.calls)
	}
	if model.recoveryToolChoice != llm.ToolChoice("required") {
		t.Fatalf("require-done recovery tool choice = %q, want required", model.recoveryToolChoice)
	}
	if !model.recoveryDisableThinking {
		t.Fatalf("require-done recovery call should set DisableThinking so a forced tool_choice stays legal under extended thinking")
	}
	if model.postRecoveryToolChoice != llm.ToolChoice("auto") {
		t.Fatalf("post-recovery tool choice = %q, want auto after ordinary recovery tool", model.postRecoveryToolChoice)
	}
	if !model.postRecoveryDisableThinking {
		t.Fatalf("post-recovery call should keep DisableThinking active until the done tool completes")
	}

	final := ""
	finalStatus := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case WarnEvent:
			if e.Kind == "require_done_safety" {
				t.Fatalf("tool-required recovery should avoid safety termination: %#v", e)
			}
		case FinalResponseEvent:
			final = e.Content
			finalStatus = e.Status
		}
	}
	if final != "finished after continued tool work" {
		t.Fatalf("final response = %q", final)
	}
	if finalStatus != "complete" {
		t.Fatalf("normal done completion status = %q, want complete", finalStatus)
	}
}
