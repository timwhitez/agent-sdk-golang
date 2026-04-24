package agent

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type streamingErrorModel struct {
	calls int
}

func (m *streamingErrorModel) Provider() string { return "stub" }
func (m *streamingErrorModel) Model() string    { return "stub" }

func (m *streamingErrorModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}

func (m *streamingErrorModel) InvokeStream(_ context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.calls++
	ch := make(chan llm.StreamEvent, 3)
	go func() {
		defer close(ch)
		ch <- llm.StreamTextDeltaEvent{Delta: "partial "}
		ch <- llm.StreamTextDeltaEvent{Delta: "response"}
		ch <- llm.StreamErrorEvent{Err: errors.New("boom")}
	}()
	return ch, nil
}

func TestStreamingErrorAppendsPartialAssistantMessage(t *testing.T) {
	model := &streamingErrorModel{}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	var errSeen bool
	for _, ev := range events {
		if _, ok := ev.(ErrorEvent); ok {
			errSeen = true
			break
		}
	}
	if !errSeen {
		t.Fatalf("expected error event from streaming failure")
	}

	msgs := ag.Messages()
	var lastAssistant *llm.Message
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == llm.RoleAssistant {
			lastAssistant = &msgs[i]
			break
		}
	}
	if lastAssistant == nil {
		t.Fatalf("expected assistant message after streaming error")
	}
	if got := lastAssistant.PlainText(); got != "partial response" {
		t.Fatalf("expected partial response in history, got %q", got)
	}
}

type streamingErrorMetaModel struct{}

func (m *streamingErrorMetaModel) Provider() string { return "stub" }
func (m *streamingErrorMetaModel) Model() string    { return "stub" }

func (m *streamingErrorMetaModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}

func (m *streamingErrorMetaModel) InvokeStream(_ context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent, 1)
	go func() {
		defer close(ch)
		ch <- llm.StreamErrorEvent{
			Provider:   "stub",
			StatusCode: 429,
			Message:    "rate limited",
			RetryAfter: 2 * time.Second,
		}
	}()
	return ch, nil
}

func TestStreamingErrorPreservesMetadata(t *testing.T) {
	model := &streamingErrorMetaModel{}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	var errEvent ErrorEvent
	var found bool
	for _, ev := range events {
		if e, ok := ev.(ErrorEvent); ok {
			errEvent = e
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected error event from streaming failure")
	}
	if errEvent.Provider != "stub" {
		t.Fatalf("expected provider stub, got %q", errEvent.Provider)
	}
	if errEvent.StatusCode != 429 {
		t.Fatalf("expected status 429, got %d", errEvent.StatusCode)
	}
	if !strings.Contains(errEvent.Message, "rate limited") {
		t.Fatalf("expected error message to include rate limited, got %q", errEvent.Message)
	}
	if !strings.Contains(strings.ToLower(errEvent.Message), "retry after") {
		t.Fatalf("expected error message to include retry after, got %q", errEvent.Message)
	}
	if errEvent.RetryAfterMS != 2000 {
		t.Fatalf("expected retry_after_ms=2000, got %d", errEvent.RetryAfterMS)
	}
	if errEvent.Kind != "rate_limit" {
		t.Fatalf("expected rate_limit kind, got %q", errEvent.Kind)
	}
}
