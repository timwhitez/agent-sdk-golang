package agent

import (
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// A quota-exhausted 429 is terminal for the invoke retry loop, still reports
// the rate_limit kind, and an ordinary 429 stays retryable.
func TestQuotaExhaustedRateLimitIsNotRetried(t *testing.T) {
	t.Parallel()
	ag, err := New(Config{LLM: &errEventModel{}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	quota := &llm.RateLimitError{Provider: "openai", Message: "quota", Reason: llm.RateLimitReasonQuotaExhausted}
	if _, retry := ag.transientInvokeRetryDelay(quota, nil, false, 1); retry {
		t.Fatal("quota-exhausted rate limit must not be retried")
	}
	if ev := ag.errEvent(quota); ev.Kind != "rate_limit" || ev.StatusCode != 429 {
		t.Fatalf("event = %+v, want rate_limit/429", ev)
	}
	if _, retry := ag.transientInvokeRetryDelay(&llm.RateLimitError{Provider: "openai", Message: "slow"}, nil, false, 1); !retry {
		t.Fatal("ordinary rate limit must stay retryable")
	}
}
