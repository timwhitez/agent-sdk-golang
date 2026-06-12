package agent

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type errEventModel struct{}

func (m *errEventModel) Provider() string { return "stub" }
func (m *errEventModel) Model() string    { return "stub" }

func (m *errEventModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("ok")}, nil
}

func TestErrEventRateLimitPreservesRetryAfterMs(t *testing.T) {
	t.Parallel()

	ag, err := New(Config{LLM: &errEventModel{}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	ev := ag.errEvent(&llm.RateLimitError{
		Provider:   "openai",
		Message:    "rate limited",
		RetryAfter: 2500 * time.Millisecond,
	})
	if ev.Kind != "rate_limit" {
		t.Fatalf("kind = %q, want %q", ev.Kind, "rate_limit")
	}
	if ev.StatusCode != 429 {
		t.Fatalf("status = %d, want 429", ev.StatusCode)
	}
	if ev.RetryAfterMS != 2500 {
		t.Fatalf("retry_after_ms = %d, want 2500", ev.RetryAfterMS)
	}
	if ev.Provider != "openai" {
		t.Fatalf("provider = %q, want %q", ev.Provider, "openai")
	}
}

func TestErrEventProviderStatusKindMapping(t *testing.T) {
	t.Parallel()

	ag, err := New(Config{LLM: &errEventModel{}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	tests := []struct {
		name       string
		statusCode int
		wantKind   string
	}{
		{name: "auth", statusCode: 401, wantKind: "auth"},
		{name: "permission", statusCode: 403, wantKind: "permission"},
		{name: "invalid_request", statusCode: 422, wantKind: "invalid_request"},
		{name: "provider", statusCode: 503, wantKind: "provider"},
		{name: "rate_limit", statusCode: 429, wantKind: "rate_limit"},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ev := ag.errEvent(&llm.ProviderError{Provider: "anthropic", StatusCode: tc.statusCode, Message: "upstream failed"})
			if ev.Kind != tc.wantKind {
				t.Fatalf("kind = %q, want %q", ev.Kind, tc.wantKind)
			}
			if ev.StatusCode != tc.statusCode {
				t.Fatalf("status = %d, want %d", ev.StatusCode, tc.statusCode)
			}
		})
	}
}

func TestErrEventClassifiesGenericFailures(t *testing.T) {
	t.Parallel()

	ag, err := New(Config{LLM: &errEventModel{}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	tests := []struct {
		name     string
		err      error
		wantKind string
	}{
		{name: "canceled", err: context.Canceled, wantKind: "canceled"},
		{name: "timeout", err: context.DeadlineExceeded, wantKind: "timeout"},
		{name: "network", err: &net.DNSError{Err: "no such host", Name: "example.invalid"}, wantKind: "network"},
		{name: "decode", err: errors.New("invalid character x looking for beginning of value"), wantKind: "decode"},
		{name: "textual_rate_limit", err: errors.New("openai-responses (429): Too Many Requests"), wantKind: "rate_limit"},
		{name: "textual_provider", err: errors.New("provider failed with HTTP status 529 overloaded"), wantKind: "provider"},
		{name: "textual_auth", err: errors.New("provider failed with HTTP status 401 unauthorized"), wantKind: "auth"},
		{name: "textual_permission", err: errors.New("provider failed with HTTP status 403 forbidden"), wantKind: "permission"},
		{name: "textual_invalid_request", err: errors.New("provider failed with HTTP status 400 invalid request"), wantKind: "invalid_request"},
		{name: "unknown", err: errors.New("boom"), wantKind: "unknown"},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ev := ag.errEvent(tc.err)
			if ev.Kind != tc.wantKind {
				t.Fatalf("kind = %q, want %q", ev.Kind, tc.wantKind)
			}
		})
	}
}

func TestErrEventFallsBackToAgentProviderWhenErrorProviderMissing(t *testing.T) {
	t.Parallel()

	ag, err := New(Config{LLM: &errEventModel{}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	ev := ag.errEvent(&llm.RateLimitError{Provider: "", Message: "slow down", RetryAfter: 500 * time.Millisecond})
	if ev.Provider != "stub" {
		t.Fatalf("provider = %q, want fallback %q", ev.Provider, "stub")
	}
	if ev.RetryAfterMS != 500 {
		t.Fatalf("retry_after_ms = %d, want 500", ev.RetryAfterMS)
	}
}
