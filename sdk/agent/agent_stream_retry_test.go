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

type streamRetryEventModel struct{ event llm.StreamRetryEvent }

func (m streamRetryEventModel) Provider() string { return "fake" }
func (m streamRetryEventModel) Model() string    { return "fake" }
func (m streamRetryEventModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("ok")}, nil
}
func (m streamRetryEventModel) InvokeStream(context.Context, llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent, 3)
	ch <- m.event
	ch <- llm.StreamTextDeltaEvent{Delta: "ok"}
	ch <- llm.StreamDoneEvent{StopReason: "stop"}
	close(ch)
	return ch, nil
}

func TestQueryStreamRetryWarningKindFollowsStatus(t *testing.T) {
	cases := []struct {
		name       string
		event      llm.StreamRetryEvent
		wantKind   string
		wantPhrase string
	}{
		{name: "429", event: llm.StreamRetryEvent{StatusCode: 429}, wantKind: "rate_limit_retry", wantPhrase: "rate limited"},
		{name: "503", event: llm.StreamRetryEvent{StatusCode: 503}, wantKind: "provider_retry", wantPhrase: "failed transiently"},
		{name: "stream cut", event: llm.StreamRetryEvent{Message: "unexpected EOF"}, wantKind: "provider_retry", wantPhrase: "unexpected EOF"},
		{name: "500", event: llm.StreamRetryEvent{StatusCode: 500}, wantKind: "provider_retry", wantPhrase: "failed transiently"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ag, err := New(Config{LLM: streamRetryEventModel{event: tc.event}, InvokeRetryMaxAttempts: 1})
			if err != nil {
				t.Fatalf("New agent: %v", err)
			}
			var warn WarnEvent
			for ev := range ag.QueryStream(context.Background(), llm.TextContent("hello")) {
				if w, ok := ev.(WarnEvent); ok {
					warn = w
				}
			}
			if warn.Kind != tc.wantKind || !strings.Contains(warn.Message, tc.wantPhrase) {
				t.Fatalf("warning = %#v, want kind %q and phrase %q", warn, tc.wantKind, tc.wantPhrase)
			}
			if got := warn.Metadata["status_code"]; got != tc.event.StatusCode {
				t.Fatalf("status_code metadata = %#v, want %d", got, tc.event.StatusCode)
			}
		})
	}
}
