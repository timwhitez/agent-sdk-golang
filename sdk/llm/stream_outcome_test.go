package llm

import (
	"context"
	"errors"
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
	stop := &RetryStopError{RetryAfter: time.Second, Reason: "wait_limit", Cause: cause}
	producer.Observe(StreamErrorEvent{Err: stop})
	producer.Observe(StreamErrorEvent{Err: context.Canceled})
	producer.Observe(StreamUsageEvent{})
	if got := outcome.Snapshot(); !got.Started || got.Complete || got.Visible || !errors.Is(got.Error, cause) || !IsRetryStop(got.Error) || got.RetryAfter != time.Second || got.RetryAfterNotBefore.Before(before.Add(time.Second)) {
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
