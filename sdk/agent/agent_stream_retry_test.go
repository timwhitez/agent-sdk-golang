package agent

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type streamRetryThenTextModel struct{}

func (streamRetryThenTextModel) Provider() string { return "fake" }
func (streamRetryThenTextModel) Model() string    { return "fake" }
func (streamRetryThenTextModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("ok")}, nil
}
func (streamRetryThenTextModel) InvokeStream(context.Context, llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent, 3)
	ch <- llm.StreamRetryEvent{
		Provider:   "fake",
		StatusCode: 429,
		Message:    "Too Many Requests",
		RetryAfter: 2 * time.Second,
		Attempt:    1,
		MaxRetries: 5,
	}
	ch <- llm.StreamTextDeltaEvent{Delta: "ok"}
	ch <- llm.StreamDoneEvent{StopReason: "stop"}
	close(ch)
	return ch, nil
}

func TestQueryStreamConvertsStreamRetryEventToWarning(t *testing.T) {
	ag, err := New(Config{LLM: streamRetryThenTextModel{}, InvokeRetryMaxAttempts: 1})
	if err != nil {
		t.Fatalf("New agent: %v", err)
	}
	ch := ag.QueryStream(context.Background(), llm.TextContent("hello"))
	var warn WarnEvent
	var final FinalResponseEvent
	for ev := range ch {
		switch e := ev.(type) {
		case WarnEvent:
			warn = e
		case FinalResponseEvent:
			final = e
		case ErrorEvent:
			t.Fatalf("retry event should not become terminal error: %#v", e)
		}
	}
	if warn.Kind != "rate_limit_retry" {
		t.Fatalf("warn kind = %q, want rate_limit_retry", warn.Kind)
	}
	if !strings.Contains(warn.Message, "retry 1/5") || !strings.Contains(warn.Message, "2s") {
		t.Fatalf("warn message missing retry details: %q", warn.Message)
	}
	if final.Content != "ok" {
		t.Fatalf("final content = %q, want ok", final.Content)
	}
}
