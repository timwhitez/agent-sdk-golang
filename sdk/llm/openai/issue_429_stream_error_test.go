package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type issue429ChatErrorTransport struct {
	body  string
	calls atomic.Int32
}

func (r *issue429ChatErrorTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r.calls.Add(1)
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader("data: " + r.body + "\n\ndata: [DONE]\n\n")), Request: req}, nil
}
func TestIssue429ChatStreamRetainsStructuredError(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		check      func(error) bool
	}{
		{"quota", `{"error":{"code":"insufficient_quota","type":"insufficient_quota","message":"rate limit exceeded"}}`, llm.IsQuotaExhausted},
		{"quota-code-only", `{"error":{"code":"insufficient_quota"}}`, llm.IsQuotaExhausted},
		{"unknown-empty", `{"error":{}}`, func(err error) bool { var e *llm.ProviderError; return errors.As(err, &e) && e.Message != "" }},
		{"unknown-code", `{"error":{"code":"unrecognized"}}`, func(err error) bool { var e *llm.ProviderError; return errors.As(err, &e) }},
		{"rate-limit", `{"error":{"type":"rate_limit_error","retry_after":1.5}}`, func(err error) bool {
			var e *llm.RateLimitError
			return errors.As(err, &e) && e.RetryAfter == 1500*time.Millisecond && e.Reason == ""
		}},
		{"server-error", `{"error":{"status_code":503,"retry_after_ms":250}}`, func(err error) bool {
			var e *llm.ProviderError
			return errors.As(err, &e) && e.StatusCode == 503 && e.RetryAfter == 250*time.Millisecond
		}},
		{"context-overflow", `{"error":{"code":"context_length_exceeded","message":"input is too large"}}`, llm.IsContextOverflow},
		{"empty-message-error", `{"error":{"code":"context_length_exceeded","message":""}}`, llm.IsContextOverflow},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rt := &issue429ChatErrorTransport{body: tc.body}
			c := &ChatClient{ModelName: "issue429", MaxRetries: 1, HTTPClient: &http.Client{Transport: rt}}
			events, err := c.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
			if err != nil {
				t.Fatal(err)
			}
			var observed error
			done := 0
			for event := range events {
				switch e := event.(type) {
				case llm.StreamErrorEvent:
					observed = e.AsError()
				case llm.StreamDoneEvent:
					done++
				}
			}
			if !tc.check(observed) || done != 0 {
				t.Fatalf("structured error lost: got=%T %v, done=%d", observed, observed, done)
			}
		})
	}
}
func TestIssue429ChatStreamQuotaDoesNotRetryAtAgent(t *testing.T) {
	rt := &issue429ChatErrorTransport{body: `{"error":{"code":"insufficient_quota","type":"insufficient_quota","message":"rate limit exceeded"}}`}
	c := &ChatClient{ModelName: "issue429", MaxRetries: 1, HTTPClient: &http.Client{Transport: rt}}
	a, err := agent.New(agent.Config{LLM: c, MaxIterations: 1, InvokeRetryMaxAttempts: 3, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	reason := ""
	for event := range a.QueryStream(context.Background(), llm.TextContent("hello")) {
		if e, ok := event.(agent.ErrorEvent); ok {
			reason = e.Reason
		}
	}
	if rt.calls.Load() != 1 || reason != llm.RateLimitReasonQuotaExhausted {
		t.Fatalf("quota error requests=%d reason=%q; want one request and quota_exhausted", rt.calls.Load(), reason)
	}
}
