package agent

import (
	"context"
	"sync/atomic"
	"testing"
)

// runPlannedBlock runs n gated calls with the given plan through the public
// RunSequentialBlock entry and returns the fixture pieces.
func runPlannedBlock(t *testing.T, n, maxWorkers int, plan func(int) BlockCallPlan) (*gatedHandlers, *atomic.Int32, <-chan error) {
	t.Helper()
	g := newGatedHandlers(n)
	trace := &parallelTrace{}
	calls, a := parallelFixture(n, g, trace, maxWorkers, allEligible)
	a.Parallel.Plan = plan
	var admitted atomic.Int32
	admit := a.Admit
	a.Admit = func(ctx context.Context, i int) (BlockAdmission, error) {
		admitted.Add(1)
		return admit(ctx, i)
	}
	done := make(chan error, 1)
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		_, err := RunSequentialBlock(context.Background(), calls, a)
		done <- err
	}()
	// Every exit path releases the gates and waits for the owner.
	t.Cleanup(func() {
		g.releaseAll()
		<-exited
	})
	return g, &admitted, done
}

// Calls that share a declared resource never run in the same wave: the
// conflicting call is admitted only after the earlier call has finished.
func TestBlockParallelismResourceConflictSplitsWave(t *testing.T) {
	plan := func(i int) BlockCallPlan {
		switch i {
		case 0, 1:
			return BlockCallPlan{Concurrent: true, Resources: []string{"file:a"}}
		default:
			return BlockCallPlan{Concurrent: true, Resources: []string{"file:b"}}
		}
	}
	g, admitted, done := runPlannedBlock(t, 3, 4, plan)
	g.waitStarted(t, 1)
	// Call 0 runs alone: call 1 shares file:a, so it cannot be admitted yet.
	if got := admitted.Load(); got != 1 {
		t.Fatalf("admitted %d calls while the conflicting call 0 is running", got)
	}
	close(g.release[0])
	// Calls 1 and 2 have disjoint resources and share the next wave.
	g.waitStarted(t, 2)
	if peak := g.peak.Load(); peak != 2 {
		t.Fatalf("peak concurrency=%d, want 2 (calls 1 and 2)", peak)
	}
	close(g.release[1])
	close(g.release[2])
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

// A panicking or missing plan is Exclusive, never concurrent.
func TestBlockParallelismPlanFailureIsExclusive(t *testing.T) {
	plan := func(i int) BlockCallPlan {
		if i == 1 {
			panic("planner bug")
		}
		return BlockCallPlan{Concurrent: true}
	}
	g, admitted, done := runPlannedBlock(t, 3, 4, plan)
	g.waitStarted(t, 1)
	if got := admitted.Load(); got != 1 {
		t.Fatalf("admitted %d calls; the call before a failed plan must run alone", got)
	}
	close(g.release[0])
	g.waitStarted(t, 1)
	if got := admitted.Load(); got != 2 {
		t.Fatalf("admitted %d; the failed-plan call must run Exclusive", got)
	}
	close(g.release[1])
	close(g.release[2])
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if peak := g.peak.Load(); peak != 1 {
		t.Fatalf("peak=%d, want every call Exclusive", peak)
	}
}

// MaxWorkers above the hard ceiling is capped: the next wave is not admitted
// until the first MaxBlockWorkers calls have finished.
func TestBlockParallelismWorkerCeiling(t *testing.T) {
	n := MaxBlockWorkers + 4
	g, admitted, done := runPlannedBlock(t, n, 1000, func(int) BlockCallPlan { return BlockCallPlan{Concurrent: true} })
	g.waitStarted(t, MaxBlockWorkers)
	if got := admitted.Load(); got != MaxBlockWorkers {
		t.Fatalf("admitted %d calls, want the %d-worker ceiling", got, MaxBlockWorkers)
	}
	g.releaseAll()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if peak := g.peak.Load(); peak > MaxBlockWorkers {
		t.Fatalf("peak=%d over ceiling", peak)
	}
}
