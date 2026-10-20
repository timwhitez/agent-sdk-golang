package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestStreamOutcomeRetainsNativeStopAndAbsoluteFloor(t *testing.T) {
	ctx, outcome := WithStreamOutcome(context.Background())
	producer := BeginStreamOutcome(ctx)
	cause := &RateLimitError{Provider: "fixture", RetryAfter: time.Second}
	before := time.Now()
	producer.Observe(StreamErrorEvent{Err: cause})
	originalFloor := outcome.Snapshot().RetryAfterNotBefore
	stop := &RetryStopError{RetryAfter: time.Second, Reason: "wait_limit", Cause: cause}
	producer.Observe(StreamErrorEvent{Err: stop})
	producer.Observe(StreamErrorEvent{Err: context.Canceled})
	producer.Observe(StreamUsageEvent{})
	if got := outcome.Snapshot(); !got.Started || got.Complete || got.Visible || !errors.Is(got.Error, cause) || !IsRetryStop(got.Error) || got.RetryAfter != time.Second || got.RetryAfterNotBefore.Before(before.Add(time.Second)) || got.RetryAfterNotBefore != originalFloor {
		t.Fatalf("lost in-flight native evidence: %+v", got)
	}
	producer.Finish()
	frozen := outcome.Snapshot()
	producer.Observe(StreamTextDeltaEvent{Delta: "late"})
	producer.Observe(StreamErrorEvent{Err: &RateLimitError{RetryAfter: time.Hour}})
	got := outcome.Snapshot()
	if !got.Complete || got.Error != stop || got.Visible || got.RetryAfterNotBefore != frozen.RetryAfterNotBefore {
		t.Fatal("completion receipt changed after publication")
	}
}

func TestStreamOutcomeNoHintVisibleAndIndependentScopes(t *testing.T) {
	ctx, outcome := WithStreamOutcome(context.Background())
	producer := BeginStreamOutcome(ctx)
	producer.Observe(StreamErrorEvent{Err: context.DeadlineExceeded})
	producer.Finish()
	got := outcome.Snapshot()
	if !got.Complete || got.RetryAfter != 0 || !got.RetryAfterNotBefore.IsZero() || !errors.Is(got.Error, context.DeadlineExceeded) || IsRetryStop(got.Error) {
		t.Fatal("ordinary no-hint timeout became a retry stop")
	}
	freshCtx, fresh := WithStreamOutcome(ctx)
	p := BeginStreamOutcome(freshCtx)
	p.Observe(StreamTextDeltaEvent{Delta: "visible"})
	p.Finish()
	if next := fresh.Snapshot(); !next.Complete || !next.Visible || next.Error != nil || outcome.Snapshot().Visible {
		t.Fatal("explicit stream scopes shared state or lost visible output")
	}
	_ = BeginStreamOutcome(freshCtx)
	if fresh.Snapshot().Complete {
		t.Fatal("reused scope falsely certifies distinct producers")
	}
}

func TestStreamOutcomeConcurrentObservationAndSnapshot(t *testing.T) {
	ctx, outcome := WithStreamOutcome(context.Background())
	p := BeginStreamOutcome(ctx)
	var readers sync.WaitGroup
	for i := 0; i < 4; i++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for n := 0; n < 100; n++ {
				_ = outcome.Snapshot()
			}
		}()
	}
	p.Observe(StreamThinkingDeltaEvent{Delta: "visible"})
	p.Observe(StreamErrorEvent{Err: &RateLimitError{RetryAfter: time.Second}})
	p.Finish()
	readers.Wait()
	if got := outcome.Snapshot(); !got.Complete || !got.Visible || got.Error == nil {
		t.Fatal("concurrent reader lost final producer facts")
	}
}

func TestStreamOutcomeFinalPermanentErrorKeepsHistoricalFloor(t *testing.T) {
	for _, hinted := range []bool{false, true} {
		for _, status := range []int{401, 403} {
			t.Run(fmt.Sprintf("hinted=%t/final=%d", hinted, status), func(t *testing.T) {
				ctx, outcome := WithStreamOutcome(context.Background())
				producer := BeginStreamOutcome(ctx)
				var first error = &ProviderError{StatusCode: 500}
				if hinted {
					first = &RateLimitError{RetryAfter: time.Hour}
				}
				producer.Observe(StreamErrorEvent{Err: first})
				original := outcome.Snapshot()
				final := &ProviderError{StatusCode: status}
				producer.Observe(StreamErrorEvent{Err: final})
				producer.Observe(StreamErrorEvent{Err: context.Canceled})
				producer.Finish()
				got := outcome.Snapshot()
				if !got.Complete || got.Error != final || IsRetryStop(got.Error) {
					t.Fatalf("final permanent error hidden by historical error or cancellation: %+v", got)
				}
				if got.RetryAfterCause != original.RetryAfterCause || (hinted && got.RetryAfterCause != first) || got.RetryAfter != original.RetryAfter || got.RetryAfterNotBefore != original.RetryAfterNotBefore {
					t.Fatal("final error erased or restarted historical floor")
				}
			})
		}
	}
}

func TestStreamOutcomeLatestErrorAndMaximumFloorAreIndependent(t *testing.T) {
	ctx, outcome := WithStreamOutcome(context.Background())
	producer := BeginStreamOutcome(ctx)
	long := &RateLimitError{RetryAfter: time.Hour}
	producer.Observe(StreamErrorEvent{Err: long})
	original := outcome.Snapshot()
	short := &ProviderError{StatusCode: 503, RetryAfter: time.Second}
	producer.Observe(StreamErrorEvent{Err: short})
	producer.Observe(StreamErrorEvent{Err: context.DeadlineExceeded})
	producer.Finish()
	got := outcome.Snapshot()
	if got.Error != short || got.RetryAfterCause != long || got.RetryAfter != time.Hour || got.RetryAfterNotBefore != original.RetryAfterNotBefore {
		t.Fatalf("latest error and maximum original floor conflated: %+v", got)
	}
}
