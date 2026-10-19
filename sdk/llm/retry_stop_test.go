package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestRetryStopPreservesFloorAndCause(t *testing.T) {
	cause := &RateLimitError{Provider: "fixture", Message: "fixture-only-secret", RetryAfter: 90 * time.Second}
	err := BoundRetryAfter(context.Background(), cause, 60*time.Second)
	var stop *RetryStopError
	if !IsRetryStop(fmt.Errorf("wrapped: %w", err)) || !errors.As(err, &stop) || stop.RetryAfter != 90*time.Second || stop.Reason != "wait_limit" || !errors.Is(err, cause) {
		t.Fatalf("lost terminal floor/cause: %#v", err)
	}
	if strings.Contains(err.Error(), cause.Message) {
		t.Fatal("stop diagnostic duplicated provider body")
	}
	if BoundRetryAfter(context.Background(), err, time.Second) != err {
		t.Fatal("stop identity changed")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	err = BoundRetryAfter(ctx, cause, 120*time.Second)
	if !errors.As(err, &stop) || stop.Reason != "deadline" {
		t.Fatalf("deadline error=%v", err)
	}
	for _, cause := range []error{errors.New("ordinary"), &RateLimitError{}, &ProviderError{RetryAfter: -time.Second}} {
		if BoundRetryAfter(context.Background(), cause, time.Second) != cause {
			t.Fatalf("no valid floor must retain classification: %v", cause)
		}
	}
}
