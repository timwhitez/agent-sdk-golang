package openai

import (
	"errors"
	"testing"
	"time"
)

func TestResolveRetryPolicyDefaults(t *testing.T) {
	policy := resolveRetryPolicy(0, 0, 0)
	if policy.maxRetries != defaultRetryMaxAttempts {
		t.Fatalf("expected max retries %d, got %d", defaultRetryMaxAttempts, policy.maxRetries)
	}
	if policy.baseDelay != defaultRetryBaseDelay {
		t.Fatalf("expected base delay %v, got %v", defaultRetryBaseDelay, policy.baseDelay)
	}
	if policy.maxDelay != defaultRetryMaxDelay {
		t.Fatalf("expected max delay %v, got %v", defaultRetryMaxDelay, policy.maxDelay)
	}
}

func TestResolveRetryDelayHonorsRetryAfterAndCap(t *testing.T) {
	origRand := backoffRandRead
	defer func() { backoffRandRead = origRand }()
	backoffRandRead = func(b []byte) (int, error) {
		for i := range b {
			b[i] = 0
		}
		return len(b), nil
	}

	if got := resolveRetryDelay(3, time.Second, 60*time.Second, 0); got != 8*time.Second {
		t.Fatalf("expected 8s delay, got %v", got)
	}
	if got := resolveRetryDelay(3, time.Second, 60*time.Second, 20*time.Second); got != 20*time.Second {
		t.Fatalf("expected retry-after 20s, got %v", got)
	}
	if got := resolveRetryDelay(3, time.Second, 60*time.Second, 90*time.Second); got != 60*time.Second {
		t.Fatalf("expected max delay 60s, got %v", got)
	}
}

func TestResolveRetryDelayAddsJitterWithinBound(t *testing.T) {
	origRand := backoffRandRead
	defer func() { backoffRandRead = origRand }()
	backoffRandRead = func(b []byte) (int, error) {
		for i := range b {
			b[i] = 0xFF
		}
		return len(b), nil
	}

	base := 10 * time.Second
	got := resolveRetryDelay(0, base, 60*time.Second, 0)
	if got <= base {
		t.Fatalf("expected positive jitter over base %v, got %v", base, got)
	}
	max := base + time.Duration(float64(base)*0.1)
	if got > max {
		t.Fatalf("expected jitter <=10%% bound %v, got %v", max, got)
	}
}

func TestRandomBackoffFractionFallbackRange(t *testing.T) {
	origRand := backoffRandRead
	defer func() { backoffRandRead = origRand }()
	backoffRandRead = func([]byte) (int, error) {
		return 0, errors.New("entropy unavailable")
	}

	v := randomBackoffFraction()
	if v < 0 || v >= 1 {
		t.Fatalf("expected jitter fraction in [0,1), got %f", v)
	}
}
