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

type streamingMetadataTransientErrorModel struct {
	streamCalls int
}

func (m *streamingMetadataTransientErrorModel) Provider() string { return "stub" }
func (m *streamingMetadataTransientErrorModel) Model() string    { return "stub" }

func (m *streamingMetadataTransientErrorModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}

func (m *streamingMetadataTransientErrorModel) InvokeStream(_ context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.streamCalls++
	ch := make(chan llm.StreamEvent, 4)
	go func() {
		defer close(ch)
		if m.streamCalls == 1 {
			ch <- llm.StreamResponseEvent{ResponseID: "resp_failed"}
			ch <- llm.StreamUsageEvent{Usage: llm.Usage{PromptTokens: 1, CompletionTokens: 0, TotalTokens: 1}}
			ch <- llm.StreamErrorEvent{Err: &llm.RateLimitError{Provider: "stub", Message: "Too Many Requests"}}
			return
		}
		ch <- llm.StreamResponseEvent{ResponseID: "resp_ok"}
		ch <- llm.StreamTextDeltaEvent{Delta: "ok"}
		ch <- llm.StreamDoneEvent{StopReason: "stop"}
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

func TestInvokeRetryRetriesTextualRateLimitAndServerErrors(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{name: "auth_401", err: errors.New("provider failed with HTTP status 401 unauthorized")},
		{name: "permission_403", err: errors.New("provider failed with HTTP status 403 permission denied")},
		{name: "rate_limit_429", err: errors.New("openai-responses (429): Too Many Requests")},
		{name: "server_status_529", err: errors.New("provider failed with HTTP status 529 overloaded")},
		{name: "server_text", err: errors.New("upstream service unavailable, please retry")},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			model := &transientInvokeModel{failFor: 1, err: tc.err}
			ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2})
			if err != nil {
				t.Fatalf("new agent: %v", err)
			}

			events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hello")))
			if model.calls != 2 {
				t.Fatalf("expected 2 invoke attempts, got %d", model.calls)
			}
			for _, ev := range events {
				if e, ok := ev.(ErrorEvent); ok {
					t.Fatalf("did not expect terminal error event after retry success: %#v", e)
				}
			}
		})
	}
}

func TestInvokeRetrySkipsTextualNonRetryableProviderErrors(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{name: "bad_request_400", err: errors.New("provider failed with HTTP status 400 invalid request")},
		{name: "invalid_request_422", err: errors.New("provider failed with HTTP status 422 invalid request")},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			model := &transientInvokeModel{failFor: 1, err: tc.err}
			ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 3})
			if err != nil {
				t.Fatalf("new agent: %v", err)
			}

			events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hello")))
			if model.calls != 1 {
				t.Fatalf("expected non-retryable failure to skip retries, got %d invoke calls", model.calls)
			}
			var foundErr bool
			for _, ev := range events {
				if _, ok := ev.(ErrorEvent); ok {
					foundErr = true
					break
				}
			}
			if !foundErr {
				t.Fatal("expected terminal error event")
			}
		})
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

func TestInvokeRetryRetriesAfterMetadataOnlyStreamingError(t *testing.T) {
	model := &streamingMetadataTransientErrorModel{}
	ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hello")))
	if model.streamCalls != 2 {
		t.Fatalf("expected metadata-only stream failure to retry, got %d stream calls", model.streamCalls)
	}

	usageEvents := 0
	finalText := ""
	finalResponseID := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case UsageEvent:
			usageEvents++
		case FinalResponseEvent:
			finalText = e.Content
			finalResponseID = e.ResponseID
		case ErrorEvent:
			t.Fatalf("did not expect terminal error event after metadata-only retry success: %#v", e)
		}
	}
	if usageEvents != 0 {
		t.Fatalf("expected failed attempt usage metadata not to leak into events, got %d usage events", usageEvents)
	}
	if finalText != "ok" {
		t.Fatalf("final text = %q, want ok", finalText)
	}
	if finalResponseID != "resp_ok" {
		t.Fatalf("final response id = %q, want resp_ok", finalResponseID)
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
	assertContiguousToolResults(t, history)
	foundSkippedResult := false
	for _, msg := range history {
		if msg.Role == llm.RoleTool && msg.ToolCallID == "call-3" && msg.IsError && strings.Contains(msg.Content.PlainText(), "Tool call skipped by loop guard") {
			foundSkippedResult = true
			break
		}
	}
	if !foundSkippedResult {
		t.Fatalf("expected synthetic skipped tool result for blocked repeated call in history")
	}
}

func TestRepeatToolSignatureGuardRetreatsAfterStrikeThreshold(t *testing.T) {
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

	// When the strike threshold is exhausted the guard retreats (disables
	// itself) instead of aborting the run, so the model continues and the
	// done tool is eventually reached. This matches Codex's loop, which has
	// no repeated-call abort.
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("loop")))
	if model.calls != 4 {
		t.Fatalf("expected run to continue to done tool after retreat, got %d model calls", model.calls)
	}
	if toolCalls != 2 {
		t.Fatalf("expected only pre-strike tool calls to execute (strike call is skipped), got %d", toolCalls)
	}

	doomLoopErrors := 0
	retreatWarnings := 0
	final := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case ErrorEvent:
			if e.Kind == "doom_loop" {
				doomLoopErrors++
			}
		case WarnEvent:
			if e.Kind == "loop_guard" && strings.Contains(e.Message, "protection exhausted") {
				retreatWarnings++
			}
		case FinalResponseEvent:
			final = e.Content
		}
	}
	if doomLoopErrors != 0 {
		t.Fatalf("expected no doom_loop error after retreat, got %d", doomLoopErrors)
	}
	if retreatWarnings != 1 {
		t.Fatalf("expected one loop_guard retreat warning, got %d", retreatWarnings)
	}
	if strings.TrimSpace(final) != "finished" {
		t.Fatalf("expected done tool final response %q, got %q", "finished", final)
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

	// Strikes persist across intervening turns until the threshold is
	// exhausted, at which point the guard retreats and the run continues to
	// the done tool instead of aborting.
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("loop")))
	if model.calls != 8 {
		t.Fatalf("expected run to continue to done call after retreat, got model calls=%d", model.calls)
	}
	if toolCalls != 5 {
		t.Fatalf("expected 5 executed echo calls before retreat, got %d", toolCalls)
	}

	doomLoopErrors := 0
	retreatWarnings := 0
	final := ""
	for _, ev := range events {
		switch e := ev.(type) {
		case ErrorEvent:
			if e.Kind == "doom_loop" {
				doomLoopErrors++
			}
		case WarnEvent:
			if e.Kind == "loop_guard" && strings.Contains(e.Message, "protection exhausted") {
				retreatWarnings++
			}
		case FinalResponseEvent:
			final = e.Content
		}
	}
	if doomLoopErrors != 0 {
		t.Fatalf("expected no doom_loop error after retreat, got %d", doomLoopErrors)
	}
	if retreatWarnings != 1 {
		t.Fatalf("expected one loop_guard retreat warning, got %d", retreatWarnings)
	}
	if strings.TrimSpace(final) != "finished" {
		t.Fatalf("expected done tool final response %q, got %q", "finished", final)
	}
}

func assertContiguousToolResults(t *testing.T, messages []llm.Message) {
	t.Helper()
	for i, msg := range messages {
		if msg.Role != llm.RoleAssistant || len(msg.ToolCalls) == 0 {
			continue
		}
		if i+len(msg.ToolCalls) >= len(messages)+1 {
			t.Fatalf("assistant tool calls at index %d exceed message history length", i)
		}
		for j, tc := range msg.ToolCalls {
			nextIdx := i + 1 + j
			if nextIdx >= len(messages) {
				t.Fatalf("assistant tool call %q at index %d is missing tool result", tc.ID, i)
			}
			next := messages[nextIdx]
			if next.Role != llm.RoleTool {
				t.Fatalf("message after assistant tool call %q at index %d has role %q, want tool", tc.ID, nextIdx, next.Role)
			}
			if next.ToolCallID != tc.ID {
				t.Fatalf("tool result at index %d has id %q, want %q", nextIdx, next.ToolCallID, tc.ID)
			}
		}
	}
}
