package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type steeringProviderStateModel struct {
	calls          int
	stateDelivered chan struct{}
	secondErr      error
	firstUsage     *llm.Usage
}

func (m *steeringProviderStateModel) Provider() string { return "test-responses" }
func (m *steeringProviderStateModel) Model() string    { return "test-model" }
func (m *steeringProviderStateModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, fmt.Errorf("buffered invoke is not expected")
}

func (m *steeringProviderStateModel) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.calls++
	call := m.calls
	stream := make(chan llm.StreamEvent)
	if call == 1 {
		go func() {
			defer close(stream)
			if m.firstUsage != nil {
				stream <- llm.StreamUsageEvent{Usage: *m.firstUsage}
			}
			stream <- llm.StreamProviderStateEvent{State: []llm.ProviderState{{
				Provider: "openai-responses",
				Kind:     "response.output_item.v1",
				Data:     json.RawMessage(`{"id":"rs_steering","type":"reasoning","encrypted_content":"ciphertext"}`),
			}}}
			close(m.stateDelivered)
			<-ctx.Done()
		}()
		return stream, nil
	}
	if call == 2 {
		var replayed []llm.ProviderState
		for _, message := range req.Messages {
			if message.Role != llm.RoleAssistant || !llm.HasProviderState(message.Content) {
				continue
			}
			state, err := llm.ProviderStateFromContent(message.Content)
			if err != nil {
				m.secondErr = err
				break
			}
			replayed = append(replayed, state...)
		}
		if len(replayed) != 1 || !strings.Contains(string(replayed[0].Data), "rs_steering") {
			m.secondErr = fmt.Errorf("second request provider state = %#v", replayed)
		}
		go func() {
			defer close(stream)
			stream <- llm.StreamTextDeltaEvent{Delta: "steered"}
			stream <- llm.StreamDoneEvent{StopReason: "end_turn"}
		}()
		return stream, nil
	}
	return nil, fmt.Errorf("unexpected invocation %d", call)
}

func mustProviderStateContent(t *testing.T, visible llm.Content, state []llm.ProviderState) llm.Content {
	t.Helper()
	content, err := llm.WithProviderState(visible, state)
	if err != nil {
		t.Fatal(err)
	}
	return content
}

func providerStateCount(t *testing.T, content llm.Content) int {
	t.Helper()
	state, err := llm.ProviderStateFromContent(content)
	if err != nil {
		t.Fatal(err)
	}
	return len(state)
}

func TestToolPairRepairClearsStaleProviderState(t *testing.T) {
	call := llm.ToolCall{ID: "call_1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}
	state := []llm.ProviderState{{Provider: "openai-responses", Kind: "response.output_item.v1", Data: json.RawMessage(`{"type":"function_call","call_id":"call_1"}`)}}
	broken := []llm.Message{{Role: llm.RoleAssistant, Content: mustProviderStateContent(t, llm.Content{}, state), ToolCalls: []llm.ToolCall{call}}}
	repaired, changed, _ := repairToolCallPairsDetailed(broken)
	if !changed || len(repaired) != 1 || len(repaired[0].ToolCalls) != 0 || providerStateCount(t, repaired[0].Content) != 0 {
		t.Fatalf("repair retained stale provider state: %#v", repaired)
	}

	complete := append(append([]llm.Message(nil), broken...), llm.NewToolMessage("call_1", "read", llm.TextContent("ok"), false))
	repaired, changed, _ = repairToolCallPairsDetailed(complete)
	if changed || len(repaired) != 2 || providerStateCount(t, repaired[0].Content) != 1 {
		t.Fatalf("valid pair lost provider state: changed=%t messages=%#v", changed, repaired)
	}

	pending := append(append([]llm.Message(nil), broken...), messageorigin.NewInternalUserMessage(
		messageorigin.KindToolCallContinuation,
		messageorigin.ResponseTruncatedContinuationText,
	))
	repaired, changed, unexpected := repairToolCallPairsDetailed(pending)
	if !changed || unexpected || len(repaired) != 2 || len(repaired[0].ToolCalls) != 0 || providerStateCount(t, repaired[0].Content) != 1 {
		t.Fatalf("pending continuation lost exact provider state: changed=%t unexpected=%t messages=%#v", changed, unexpected, repaired)
	}
}

func TestToolContinuationCleanupClearsStaleProviderState(t *testing.T) {
	call := llm.ToolCall{ID: "call_1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{"path":`}}
	messages := []llm.Message{{
		Role: llm.RoleAssistant,
		Content: mustProviderStateContent(t, llm.Content{}, []llm.ProviderState{{
			Provider: "openai-responses",
			Kind:     "response.output_item.v1",
			Data:     json.RawMessage(`{"type":"function_call","call_id":"call_1"}`),
		}}),
		ToolCalls: []llm.ToolCall{call},
	}, messageorigin.NewInternalUserMessage(
		messageorigin.KindToolCallContinuation,
		messageorigin.ResponseTruncatedContinuationText,
	)}
	continuation := newToolCallContinuation(2)
	continuation.addPartial([]llm.ToolCall{call})
	continuation.clearPartialToolCalls(messages, 2)
	if len(messages[0].ToolCalls) != 0 || providerStateCount(t, messages[0].Content) != 0 {
		t.Fatalf("continuation cleanup retained stale state: %#v", messages[0])
	}
}

func TestSteeringBetweenProviderStateAndDonePersistsStateForNextRequest(t *testing.T) {
	model := &steeringProviderStateModel{stateDelivered: make(chan struct{})}
	ag, err := New(Config{LLM: model, MaxIterations: 3})
	if err != nil {
		t.Fatal(err)
	}
	steering := make(chan SteeringMsg, 1)
	events := ag.QueryStreamWithSteering(context.Background(), llm.TextContent("start"), steering)
	<-model.stateDelivered
	steering <- SteeringMsg{Content: "change direction"}

	final := ""
	for event := range events {
		switch event := event.(type) {
		case FinalResponseEvent:
			final = event.Content
		case ErrorEvent:
			t.Fatalf("query error: %#v", event)
		}
	}
	if model.secondErr != nil {
		t.Fatal(model.secondErr)
	}
	if model.calls != 2 || final != "steered" {
		t.Fatalf("calls/final = %d/%q", model.calls, final)
	}
}

func TestIdleRecoveryBetweenProviderStateAndDonePersistsStateForNextRequest(t *testing.T) {
	model := &steeringProviderStateModel{stateDelivered: make(chan struct{}), firstUsage: llm.NewProviderUsage(10, 2, 12)}
	ag, err := New(Config{LLM: model, MaxIterations: 3, InvokeRetryMaxAttempts: 1, StreamIdleTimeout: 20 * time.Millisecond, StreamIdleMaxRecoveries: 1, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	usage, accounting, finals := 0, 0, 0
	for event := range ag.QueryStream(ctx, llm.TextContent("start")) {
		switch event := event.(type) {
		case ErrorEvent:
			t.Fatalf("query error: %#v", event)
		case UsageEvent:
			usage++
			if event.Usage.PromptTokens != 10 || event.Usage.CompletionTokens != 2 {
				t.Fatalf("partial usage changed: %+v", event.Usage)
			}
		case AccountingEvent:
			accounting++
		case FinalResponseEvent:
			finals++
			if event.Content != "steered" || event.StallRecoveries != 1 {
				t.Fatalf("unexpected final: %+v", event)
			}
		}
	}
	if model.secondErr != nil {
		t.Fatal(model.secondErr)
	}
	if model.calls != 2 || finals != 1 || usage != 1 || accounting != 1 {
		t.Fatalf("calls/finals/usage/accounting=%d/%d/%d/%d", model.calls, finals, usage, accounting)
	}
	states := 0
	for _, message := range ag.Messages() {
		if llm.HasProviderState(message.Content) {
			states++
			if message.Role != llm.RoleAssistant || message.PlainText() != "" {
				t.Fatal("opaque provider state became visible text")
			}
		}
	}
	if states != 1 {
		t.Fatalf("provider-state messages=%d", states)
	}
}
