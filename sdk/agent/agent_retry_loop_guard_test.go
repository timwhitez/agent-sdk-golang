package agent

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type transientInvokeModel struct {
	calls   int
	failFor int
	err     error
}

func (m *transientInvokeModel) Provider() string { return "stub" }
func (m *transientInvokeModel) Model() string    { return "stub" }

func (m *transientInvokeModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls <= m.failFor {
		return nil, m.err
	}
	return &llm.Completion{Content: llm.TextContent("ok"), StopReason: "stop"}, nil
}

type streamingTransientErrorModel struct {
	streamCalls int
}

func (m *streamingTransientErrorModel) Provider() string { return "stub" }
func (m *streamingTransientErrorModel) Model() string    { return "stub" }

func (m *streamingTransientErrorModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}

func (m *streamingTransientErrorModel) InvokeStream(_ context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.streamCalls++
	ch := make(chan llm.StreamEvent, 2)
	go func() {
		defer close(ch)
		ch <- llm.StreamTextDeltaEvent{Delta: "partial"}
		ch <- llm.StreamErrorEvent{Err: &net.DNSError{Err: "i/o timeout", IsTimeout: true}}
	}()
	return ch, nil
}

type repeatedSignatureModel struct {
	calls int
}

func (m *repeatedSignatureModel) Provider() string { return "stub" }
func (m *repeatedSignatureModel) Model() string    { return "stub" }

func (m *repeatedSignatureModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls >= 4 {
		return &llm.Completion{
			ToolCalls: []llm.ToolCall{{
				ID:   "done-call",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "done",
					Arguments: `{"message":"finished"}`,
				},
			}},
			StopReason: "tool_calls",
		}, nil
	}
	return &llm.Completion{
		ToolCalls: []llm.ToolCall{{
			ID:   fmt.Sprintf("call-%d", m.calls),
			Type: "function",
			Function: llm.FunctionCall{
				Name:      "echo",
				Arguments: `{"text":"repeat"}`,
			},
		}},
		StopReason: "tool_calls",
	}, nil
}

type staggeredStrikeModel struct {
	calls int
}

func (m *staggeredStrikeModel) Provider() string { return "stub" }
func (m *staggeredStrikeModel) Model() string    { return "stub" }

func (m *staggeredStrikeModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	echoRepeat := llm.ToolCall{
		ID:   fmt.Sprintf("repeat-%d", m.calls),
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "echo",
			Arguments: `{"text":"repeat"}`,
		},
	}
	switch m.calls {
	case 1, 2, 3, 5, 6, 7:
		return &llm.Completion{ToolCalls: []llm.ToolCall{echoRepeat}, StopReason: "tool_calls"}, nil
	case 4:
		return &llm.Completion{ToolCalls: []llm.ToolCall{{
			ID:   "variant-4",
			Type: "function",
			Function: llm.FunctionCall{
				Name:      "echo",
				Arguments: `{"text":"variant"}`,
			},
		}}, StopReason: "tool_calls"}, nil
	case 8:
		return &llm.Completion{ToolCalls: []llm.ToolCall{{
			ID:   "done-call",
			Type: "function",
			Function: llm.FunctionCall{
				Name:      "done",
				Arguments: `{"message":"finished"}`,
			},
		}}, StopReason: "tool_calls"}, nil
	default:
		return &llm.Completion{Content: llm.TextContent("unexpected"), StopReason: "stop"}, nil
	}
}

func TestInvokeRetryRetriesTransientErrorsAndSucceeds(t *testing.T) {
	model := &transientInvokeModel{failFor: 1, err: &net.DNSError{Err: "i/o timeout", IsTimeout: true}}
	ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hello")))
	if model.calls != 2 {
		t.Fatalf("expected 2 invoke attempts, got %d", model.calls)
	}

	final := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case ErrorEvent:
			t.Fatalf("did not expect terminal error event after retry success: %#v", e)
		case FinalResponseEvent:
			final = e.Content
		}
	}
	if final != "ok" {
		t.Fatalf("expected final response %q after retry success, got %q", "ok", final)
	}
}

func TestInvokeRetrySkipsNonTransientErrors(t *testing.T) {
	model := &transientInvokeModel{failFor: 1, err: errors.New("invalid payload")}
	ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 3})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hello")))
	if model.calls != 1 {
		t.Fatalf("expected non-transient failure to skip retries, got %d invoke calls", model.calls)
	}

	var errEvent ErrorEvent
	foundErr := false
	for _, ev := range events {
		if e, ok := ev.(ErrorEvent); ok {
			errEvent = e
			foundErr = true
			break
		}
	}
	if !foundErr {
		t.Fatal("expected error event for non-transient failure")
	}
	if errEvent.Kind != "unknown" {
		t.Fatalf("expected unknown kind for non-transient synthetic error, got %q", errEvent.Kind)
	}
}

func TestInvokeRetryDoesNotRetryAfterStreamingPartialOutput(t *testing.T) {
	model := &streamingTransientErrorModel{}
	ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hello")))
	if model.streamCalls != 1 {
		t.Fatalf("expected no invoke retry once partial stream output is emitted, got %d stream calls", model.streamCalls)
	}

	textDeltas := 0
	errSeen := false
	for _, ev := range events {
		switch e := ev.(type) {
		case TextDeltaEvent:
			if e.Delta == "partial" {
				textDeltas++
			}
		case ErrorEvent:
			errSeen = true
		}
	}
	if textDeltas != 1 {
		t.Fatalf("expected single streamed partial delta, got %d", textDeltas)
	}
	if !errSeen {
		t.Fatal("expected terminal error event from streaming failure")
	}
}

func TestRepeatToolSignatureGuardInjectsReminderAndContinuesUntilDone(t *testing.T) {
	model := &repeatedSignatureModel{}
	toolCalls := 0
	reminderText := "stop loop and continue"
	echoTool := tools.Func[struct {
		Text string `json:"text"`
	}]("echo", "echo", func(_ context.Context, _ struct {
		Text string `json:"text"`
	}, _ *tools.Container) (any, error) {
		toolCalls++
		return "ok", nil
	})
	doneTool := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "done", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})

	ag, err := New(Config{
		LLM:                          model,
		Tools:                        []tools.Tool{echoTool, doneTool},
		MaxIterations:                20,
		RepeatToolSignatureThreshold: 3,
		RepeatToolSignatureWindow:    6,
		LoopGuardStrikeThreshold:     2,
		LoopGuardUserMessage:         reminderText,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("loop")))
	if model.calls != 4 {
		t.Fatalf("expected loop guard warning to allow continued execution, got %d model calls", model.calls)
	}
	if toolCalls != 2 {
		t.Fatalf("expected repeated tool execution to be skipped after warning, got %d", toolCalls)
	}

	var warnEvent WarnEvent
	foundWarn := false
	hiddenReminder := 0
	loopGuardErrors := 0
	doomLoopErrors := 0
	final := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case WarnEvent:
			if e.Kind == "loop_guard" {
				warnEvent = e
				foundWarn = true
			}
		case HiddenUserMessageEvent:
			if strings.Contains(e.Content, reminderText) {
				hiddenReminder++
			}
		case ErrorEvent:
			if e.Kind == "loop_guard" {
				loopGuardErrors++
			}
			if e.Kind == "doom_loop" {
				doomLoopErrors++
			}
		case FinalResponseEvent:
			final = e.Content
		}
	}
	if !foundWarn {
		t.Fatal("expected loop guard warning event")
	}
	if !strings.Contains(strings.ToLower(warnEvent.Message), "repeated tool-call signature") {
		t.Fatalf("expected loop guard warning message, got %q", warnEvent.Message)
	}
	if loopGuardErrors != 0 {
		t.Fatalf("expected no loop_guard fatal errors, got %d", loopGuardErrors)
	}
	if doomLoopErrors != 0 {
		t.Fatalf("expected no doom_loop errors, got %d", doomLoopErrors)
	}
	if hiddenReminder != 1 {
		t.Fatalf("expected one hidden reminder event, got %d", hiddenReminder)
	}
	if strings.TrimSpace(final) != "finished" {
		t.Fatalf("expected final done response, got %q", final)
	}
	history := ag.Messages()
	foundReminder := false
	for _, msg := range history {
		if msg.Role == llm.RoleUser && strings.Contains(msg.Content.PlainText(), reminderText) {
			foundReminder = true
			break
		}
	}
	if !foundReminder {
		t.Fatalf("expected loop-guard reminder in history")
	}
}

func TestRepeatToolSignatureGuardAbortsAfterStrikeThreshold(t *testing.T) {
	model := &repeatedSignatureModel{}
	toolCalls := 0
	echoTool := tools.Func[struct {
		Text string `json:"text"`
	}]("echo", "echo", func(_ context.Context, _ struct {
		Text string `json:"text"`
	}, _ *tools.Container) (any, error) {
		toolCalls++
		return "ok", nil
	})
	doneTool := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "done", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})

	ag, err := New(Config{
		LLM:                          model,
		Tools:                        []tools.Tool{echoTool, doneTool},
		MaxIterations:                20,
		RepeatToolSignatureThreshold: 3,
		RepeatToolSignatureWindow:    6,
		LoopGuardStrikeThreshold:     1,
		LoopGuardUserMessage:         "stop repeating",
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("loop")))
	if model.calls != 3 {
		t.Fatalf("expected doom loop abort before done tool, got %d model calls", model.calls)
	}
	if toolCalls != 2 {
		t.Fatalf("expected only pre-strike tool calls to execute, got %d", toolCalls)
	}

	doomLoopErrors := 0
	final := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case ErrorEvent:
			if e.Kind == "doom_loop" {
				doomLoopErrors++
			}
		case FinalResponseEvent:
			final = e.Content
		}
	}
	if doomLoopErrors != 1 {
		t.Fatalf("expected one doom_loop error, got %d", doomLoopErrors)
	}
	if strings.TrimSpace(final) != doomLoopFinalResponse {
		t.Fatalf("expected doom loop final response %q, got %q", doomLoopFinalResponse, final)
	}
}

func TestRepeatToolSignatureGuardStrikePersistsAcrossInterveningTurns(t *testing.T) {
	model := &staggeredStrikeModel{}
	toolCalls := 0
	echoTool := tools.Func[struct {
		Text string `json:"text"`
	}]("echo", "echo", func(_ context.Context, _ struct {
		Text string `json:"text"`
	}, _ *tools.Container) (any, error) {
		toolCalls++
		return "ok", nil
	})
	doneTool := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "done", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})

	ag, err := New(Config{
		LLM:                          model,
		Tools:                        []tools.Tool{echoTool, doneTool},
		MaxIterations:                30,
		RepeatToolSignatureThreshold: 3,
		RepeatToolSignatureWindow:    6,
		LoopGuardStrikeThreshold:     2,
		LoopGuardUserMessage:         "stop repeating",
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("loop")))
	if model.calls != 7 {
		t.Fatalf("expected second strike to abort before done call, got model calls=%d", model.calls)
	}
	if toolCalls != 5 {
		t.Fatalf("expected 5 executed echo calls before abort, got %d", toolCalls)
	}

	doomLoopErrors := 0
	final := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case ErrorEvent:
			if e.Kind == "doom_loop" {
				doomLoopErrors++
			}
		case FinalResponseEvent:
			final = e.Content
		}
	}
	if doomLoopErrors != 1 {
		t.Fatalf("expected one doom_loop error, got %d", doomLoopErrors)
	}
	if strings.TrimSpace(final) != doomLoopFinalResponse {
		t.Fatalf("expected doom loop final response %q, got %q", doomLoopFinalResponse, final)
	}
}
