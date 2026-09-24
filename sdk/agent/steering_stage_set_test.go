package agent

import (
	"context"
	"testing"
)

// Concurrent stages (tool calls of one wave) are all interruptible for
// steering: one interrupt cancels every active stage, each reports it, and a
// finished stage is never cancelled afterwards.
func TestSteeringInterruptReachesEveryActiveStage(t *testing.T) {
	a := &Agent{}
	first, finishFirst := a.beginSteeringInterruptibleStage(context.Background())
	second, finishSecond := a.beginSteeringInterruptibleStage(context.Background())
	done, finishDone := a.beginSteeringInterruptibleStage(context.Background())
	if finishDone() {
		t.Fatal("finished stage reported a steering interrupt")
	}
	if !a.InterruptActiveStageForSteering() {
		t.Fatal("no active stage interrupted")
	}
	for name, ctx := range map[string]context.Context{"first": first, "second": second} {
		select {
		case <-ctx.Done():
		default:
			t.Fatalf("%s stage was not interrupted", name)
		}
	}
	if !finishFirst() || !finishSecond() {
		t.Fatal("interrupted stages did not report steering")
	}
	if done.Err() == nil {
		t.Fatal("finished stage context should be cancelled by its own finish")
	}
	if a.InterruptActiveStageForSteering() {
		t.Fatal("interrupt reported an active stage after all finished")
	}
	// A later stage starts clean.
	_, finishLater := a.beginSteeringInterruptibleStage(context.Background())
	if finishLater() {
		t.Fatal("new stage inherited an earlier interrupt")
	}
}
