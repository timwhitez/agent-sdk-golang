package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestErrorEventCarriesRateLimitReason(t *testing.T) {
	quota := &llm.RateLimitError{Provider: "p", Message: "no credit", Reason: llm.RateLimitReasonQuotaExhausted}
	cases := []struct {
		name      string
		err       error
		wantKind  string
		wantQuota bool
		wantReas  string
	}{
		{"quota 429", quota, "rate_limit", true, "quota_exhausted"},
		{"wrapped quota 429", fmt.Errorf("invoke: %w", quota), "rate_limit", true, "quota_exhausted"},
		{"ordinary 429", &llm.RateLimitError{Provider: "p", Message: "slow down"}, "rate_limit", false, ""},
		{"other reason is not quota", &llm.RateLimitError{Provider: "p", Reason: "other"}, "rate_limit", false, "other"},
		{"provider 500", &llm.ProviderError{Provider: "p", StatusCode: 500, Message: "boom"}, "provider", false, ""},
		{"plain error", errors.New("boom"), "unknown", false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ev := errEventForModel(tc.err, nil)
			if tc.wantKind != "unknown" && ev.Kind != tc.wantKind {
				t.Fatalf("Kind = %q, want %q", ev.Kind, tc.wantKind)
			}
			if ev.Reason != tc.wantReas || ev.IsQuotaExhausted() != tc.wantQuota {
				t.Fatalf("Reason=%q IsQuotaExhausted=%v, want %q/%v", ev.Reason, ev.IsQuotaExhausted(), tc.wantReas, tc.wantQuota)
			}
			if ev.IsQuotaExhausted() != llm.IsQuotaExhausted(tc.err) {
				t.Fatalf("event disagrees with llm.IsQuotaExhausted(err)")
			}
		})
	}
}

func TestErrorEventReasonJSONIsBackwardCompatible(t *testing.T) {
	ordinary, err := json.Marshal(errEventForModel(&llm.RateLimitError{Provider: "p", Message: "slow"}, nil))
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"Provider":"p","StatusCode":429,"Message":"slow","RetryAfterMS":0,"Kind":"rate_limit","StallRecoveries":0}`
	if string(ordinary) != want {
		t.Fatalf("ordinary event JSON changed:\n got %s\nwant %s", ordinary, want)
	}

	raw, err := json.Marshal(errEventForModel(&llm.RateLimitError{Provider: "p", Reason: llm.RateLimitReasonQuotaExhausted}, nil))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"Reason":"quota_exhausted"`) {
		t.Fatalf("quota event JSON lacks reason: %s", raw)
	}
	var back ErrorEvent
	if err := json.Unmarshal(raw, &back); err != nil || !back.IsQuotaExhausted() {
		t.Fatalf("round trip lost quota reason: err=%v event=%+v", err, back)
	}
	// An event serialized before the field existed still decodes, as not quota.
	var old ErrorEvent
	if err := json.Unmarshal([]byte(want), &old); err != nil || old.IsQuotaExhausted() {
		t.Fatalf("old JSON: err=%v quota=%v", err, old.IsQuotaExhausted())
	}
}

type quotaStreamModel struct{ calls atomic.Int32 }

func (m *quotaStreamModel) Provider() string { return "fixture" }
func (m *quotaStreamModel) Model() string    { return "quota" }
func (m *quotaStreamModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	panic("buffered path unused")
}
func (m *quotaStreamModel) InvokeStream(context.Context, llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.calls.Add(1)
	ch := make(chan llm.StreamEvent, 1)
	ch <- llm.StreamErrorEvent{Err: fmt.Errorf("stream: %w", &llm.RateLimitError{Provider: "fixture", Message: "no credit", Reason: llm.RateLimitReasonQuotaExhausted})}
	close(ch)
	return ch, nil
}

// End to end through the agent loop: the quota reason reaches the host event
// and the SDK still does not retry it.
func TestQueryStreamErrorEventReportsQuotaExhausted(t *testing.T) {
	model := &quotaStreamModel{}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatal(err)
	}
	var got *ErrorEvent
	for _, ev := range collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi"))) {
		if e, ok := ev.(ErrorEvent); ok {
			got = &e
		}
	}
	if got == nil || !got.IsQuotaExhausted() || got.Kind != "rate_limit" || got.StatusCode != 429 {
		t.Fatalf("ErrorEvent = %+v, want quota-exhausted rate_limit", got)
	}
	if n := model.calls.Load(); n != 1 {
		t.Fatalf("model called %d times, want 1 (quota is not retried)", n)
	}
}

// compactWithRetry bounds a quota 429 whose Retry-After exceeds the wait limit
// into a RetryStopError; the reason must survive that wrapping.
func TestErrorEventKeepsQuotaReasonThroughRetryStop(t *testing.T) {
	quota := &llm.RateLimitError{Provider: "p", Message: "no credit", RetryAfter: time.Hour, Reason: llm.RateLimitReasonQuotaExhausted}
	err := llm.BoundRetryAfter(context.Background(), quota, 5*time.Second)
	if !llm.IsRetryStop(err) {
		t.Fatalf("setup: %v is not a retry stop", err)
	}
	ev := errEventForModel(err, nil)
	if ev.Kind != "retry_stopped" || ev.RetryStop == nil {
		t.Fatalf("Kind=%q RetryStop=%v, want retry_stopped unchanged", ev.Kind, ev.RetryStop)
	}
	if !ev.IsQuotaExhausted() || ev.IsQuotaExhausted() != llm.IsQuotaExhausted(err) {
		t.Fatalf("Reason=%q, want quota_exhausted matching llm.IsQuotaExhausted", ev.Reason)
	}
	ordinary := llm.BoundRetryAfter(context.Background(), &llm.RateLimitError{Provider: "p", RetryAfter: time.Hour}, 5*time.Second)
	if ev := errEventForModel(ordinary, nil); ev.Reason != "" || ev.IsQuotaExhausted() {
		t.Fatalf("ordinary bounded 429 got Reason=%q", ev.Reason)
	}
}
