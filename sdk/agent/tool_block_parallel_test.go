package agent

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// gatedHandlers lets a test release each handler explicitly and observe how
// many run at the same time.
type gatedHandlers struct {
	release       []chan struct{}
	started       chan int
	active, peak  atomic.Int32
	mu            sync.Mutex
	returnedOrder []int
}

func newGatedHandlers(n int) *gatedHandlers {
	g := &gatedHandlers{release: make([]chan struct{}, n), started: make(chan int, n)}
	for i := range g.release {
		g.release[i] = make(chan struct{})
	}
	return g
}

func (g *gatedHandlers) handler(ctx context.Context, i int) (any, error) {
	now := g.active.Add(1)
	for {
		peak := g.peak.Load()
		if now <= peak || g.peak.CompareAndSwap(peak, now) {
			break
		}
	}
	g.started <- i
	defer g.active.Add(-1)
	select {
	case <-g.release[i]:
	case <-ctx.Done():
		<-g.release[i]
		g.record(i)
		return nil, ctx.Err()
	}
	g.record(i)
	if i == 99 {
		return nil, errors.New("unused")
	}
	return fmt.Sprint("result-", i), nil
}

func (g *gatedHandlers) record(i int) {
	g.mu.Lock()
	g.returnedOrder = append(g.returnedOrder, i)
	g.mu.Unlock()
}

func (g *gatedHandlers) waitStarted(t *testing.T, count int) []int {
	t.Helper()
	var got []int
	for len(got) < count {
		select {
		case i := <-g.started:
			got = append(got, i)
		case <-time.After(5 * time.Second):
			t.Fatalf("only %v started", got)
		}
	}
	return got
}

type parallelTrace struct {
	mu        sync.Mutex
	committed []string
	published []int
}

func parallelFixture(n int, g *gatedHandlers, trace *parallelTrace, maxWorkers int, eligible func(int) bool) ([]llm.ToolCall, SequentialBlockAdapter) {
	calls, a := sequentialFixture(n, g.handler)
	a.Commit = func(ts []BlockTerminal) error {
		trace.mu.Lock()
		defer trace.mu.Unlock()
		for _, t := range ts {
			trace.committed = append(trace.committed, t.History.ToolCallID+"="+t.History.Content.PlainText())
		}
		return nil
	}
	a.Publish = func(i int, _ BlockTerminal, _ time.Duration) {
		trace.mu.Lock()
		trace.published = append(trace.published, i)
		trace.mu.Unlock()
	}
	a.parallel = &blockParallelism{maxWorkers: maxWorkers, eligible: eligible}
	return calls, a
}

func allEligible(int) bool { return true }

// requireSingleTerminal checks that every accepted call ended with exactly one
// terminal record and that the block is closed.
func requireSingleTerminal(t *testing.T, state *toolBlockState) {
	t.Helper()
	for i, call := range state.calls {
		if call.phase != toolCallTerminal || call.terminalCount != 1 {
			t.Fatalf("call %d state=%+v", i, call)
		}
	}
	if err := state.validateClosed(); err != nil {
		t.Fatal(err)
	}
}

// Handlers finish 3 -> 1 -> 2 (ordinals 2, 0, 1); the owner commits and
// publishes in model order with exactly one terminal per call.
func TestToolBlockParallelCompletionCommitsInModelOrder(t *testing.T) {
	g := newGatedHandlers(3)
	trace := &parallelTrace{}
	calls, a := parallelFixture(3, g, trace, 3, allEligible)
	state, _ := newToolBlockState(calls)
	type result struct {
		stop BlockStop
		err  error
	}
	done := make(chan result, 1)
	go func() {
		stop, err := runSequentialBlock(context.Background(), state, calls, a, false)
		done <- result{stop, err}
	}()
	settled := make(chan int, 3)
	waveCompletionSettled = func(i int) { settled <- i }
	t.Cleanup(func() { waveCompletionSettled = nil })
	g.waitStarted(t, 3)
	committedAfter := map[int][]string{}
	for _, i := range []int{2, 0, 1} {
		close(g.release[i])
		// The owner has fully processed this completion before we look.
		select {
		case got := <-settled:
			if got != i {
				t.Fatalf("settled %d, want %d", got, i)
			}
		case res := <-done:
			t.Fatalf("owner exited before settling call %d: stop=%s err=%v", i, res.stop, res.err)
		case <-time.After(5 * time.Second):
			// Failure path only: unblock the remaining workers so the owner
			// (which must wait for them) can return, then fail.
			for _, ch := range g.release {
				select {
				case <-ch:
				default:
					close(ch)
				}
			}
			t.Fatalf("owner never settled call %d in model order", i)
		}
		trace.mu.Lock()
		committedAfter[i] = append([]string(nil), trace.committed...)
		trace.mu.Unlock()
	}
	if len(committedAfter[2]) != 0 {
		t.Fatalf("call-2 returned first but %v was committed before call-0", committedAfter[2])
	}
	if want := []string{"call-0=result-0"}; !reflect.DeepEqual(committedAfter[0], want) {
		t.Fatalf("after call-0 returned committed=%v want %v", committedAfter[0], want)
	}
	res := <-done
	if res.err != nil || res.stop != BlockContinue {
		t.Fatalf("stop=%s err=%v", res.stop, res.err)
	}
	if want := []string{"call-0=result-0", "call-1=result-1", "call-2=result-2"}; !reflect.DeepEqual(trace.committed, want) {
		t.Fatalf("committed=%v want %v", trace.committed, want)
	}
	if want := []int{0, 1, 2}; !reflect.DeepEqual(trace.published, want) {
		t.Fatalf("published=%v", trace.published)
	}
	if !reflect.DeepEqual(g.returnedOrder, []int{2, 0, 1}) {
		t.Fatalf("handlers returned %v, want out of order", g.returnedOrder)
	}
	for _, call := range state.calls {
		if call.terminalCount != 1 || call.executionKnowledge != toolExecutionOutcomeObserved || call.phase != toolCallTerminal {
			t.Fatalf("state=%+v", call)
		}
	}
	if err := state.validateClosed(); err != nil {
		t.Fatal(err)
	}
}

// The worker bound holds across consecutive waves: a call of wave k starts
// only after every call of the earlier waves has settled.
func TestToolBlockParallelRespectsWorkerBound(t *testing.T) {
	g := newGatedHandlers(5)
	trace := &parallelTrace{}
	var settled atomic.Int32
	waveCompletionSettled = func(int) { settled.Add(1) }
	t.Cleanup(func() { waveCompletionSettled = nil })
	var settledAtStart [5]int32
	calls, a := parallelFixture(5, g, trace, 2, allEligible)
	admit := a.Admit
	a.Admit = func(ctx context.Context, i int) (BlockAdmission, error) {
		settledAtStart[i] = settled.Load()
		return admit(ctx, i)
	}
	for _, ch := range g.release {
		close(ch)
	}
	state, _ := newToolBlockState(calls)
	if _, err := runSequentialBlock(context.Background(), state, calls, a, false); err != nil {
		t.Fatal(err)
	}
	for i, want := range []int32{0, 0, 2, 2, 4} {
		if settledAtStart[i] < want {
			t.Fatalf("call %d admitted after %d settled completions, want >= %d (bound exceeded)", i, settledAtStart[i], want)
		}
	}
	if peak := g.peak.Load(); peak > 2 {
		t.Fatalf("peak concurrency=%d exceeds bound 2", peak)
	}
	if !reflect.DeepEqual(trace.published, []int{0, 1, 2, 3, 4}) {
		t.Fatalf("published=%v", trace.published)
	}
	requireSingleTerminal(t, state)
}

// An ineligible call is a barrier: it starts only after every earlier call
// has committed, and later calls start only after it has committed.
func TestToolBlockParallelIneligibleCallIsBarrier(t *testing.T) {
	g := newGatedHandlers(4)
	trace := &parallelTrace{}
	calls, a := parallelFixture(4, g, trace, 4, func(i int) bool { return i != 2 })
	var committedAtStart [4][]string
	admit := a.Admit
	a.Admit = func(ctx context.Context, i int) (BlockAdmission, error) {
		trace.mu.Lock()
		committedAtStart[i] = append([]string(nil), trace.committed...)
		trace.mu.Unlock()
		return admit(ctx, i)
	}
	for _, ch := range g.release {
		close(ch)
	}
	state, _ := newToolBlockState(calls)
	if _, err := runSequentialBlock(context.Background(), state, calls, a, false); err != nil {
		t.Fatal(err)
	}
	if len(committedAtStart[2]) != 2 {
		t.Fatalf("barrier call admitted after commits %v, want both wave calls", committedAtStart[2])
	}
	if len(committedAtStart[3]) != 3 {
		t.Fatalf("call after barrier admitted after commits %v, want three", committedAtStart[3])
	}
	if !reflect.DeepEqual(trace.published, []int{0, 1, 2, 3}) {
		t.Fatalf("published=%v", trace.published)
	}
	requireSingleTerminal(t, state)
}

// Root cancellation mid-wave: started handlers settle with their real
// outcome (indeterminate knowledge), later calls are closed not_started.
func TestToolBlockParallelRootCancelKeepsStartedKnowledge(t *testing.T) {
	g := newGatedHandlers(4)
	trace := &parallelTrace{}
	calls, a := parallelFixture(4, g, trace, 2, allEligible)
	ctx, cancel := context.WithCancel(context.Background())
	state, _ := newToolBlockState(calls)
	type result struct {
		stop BlockStop
		err  error
	}
	done := make(chan result, 1)
	go func() {
		stop, err := runSequentialBlock(ctx, state, calls, a, false)
		done <- result{stop, err}
	}()
	g.waitStarted(t, 2)
	cancel()
	close(g.release[1])
	close(g.release[0])
	var res result
	select {
	case res = <-done:
	case <-time.After(5 * time.Second):
		// Release anything a wrong implementation started, then fail.
		close(g.release[2])
		close(g.release[3])
		t.Fatal("owner did not stop admitting after root cancellation")
	}
	if res.err != nil || res.stop != BlockRootAfterHandler {
		t.Fatalf("stop=%s err=%v", res.stop, res.err)
	}
	want := []toolExecutionKnowledge{toolExecutionIndeterminate, toolExecutionIndeterminate, toolExecutionNotStarted, toolExecutionNotStarted}
	for i, call := range state.calls {
		if call.executionKnowledge != want[i] || call.terminalCount != 1 {
			t.Fatalf("call %d state=%+v", i, call)
		}
	}
	if len(trace.committed) != 4 {
		t.Fatalf("committed=%v", trace.committed)
	}
	requireSingleTerminal(t, state)
}

// A panicking worker settles through OnPanic in order; siblings still commit
// their own results and nothing runs twice.
func TestToolBlockParallelPanicSettlesInOrder(t *testing.T) {
	var runs [3]atomic.Int32
	calls, a := sequentialFixture(3, func(_ context.Context, i int) (any, error) {
		runs[i].Add(1)
		if i == 1 {
			panic("boom")
		}
		return fmt.Sprint("result-", i), nil
	})
	var committed []string
	a.Commit = func(ts []BlockTerminal) error {
		for _, t := range ts {
			committed = append(committed, t.History.ToolCallID)
		}
		return nil
	}
	a.OnPanic = func(int, context.Context, any) (llm.Content, error) {
		return llm.TextContent("panicked"), errors.New("panicked")
	}
	a.parallel = &blockParallelism{maxWorkers: 3, eligible: allEligible}
	state, _ := newToolBlockState(calls)
	if _, err := runSequentialBlock(context.Background(), state, calls, a, false); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(committed, []string{"call-0", "call-1", "call-2"}) {
		t.Fatalf("committed=%v", committed)
	}
	for i := range runs {
		if runs[i].Load() != 1 {
			t.Fatalf("call %d ran %d times", i, runs[i].Load())
		}
	}
	requireSingleTerminal(t, state)
}

// A commit failure after out-of-order returns stops settlement without
// re-running effects, waits for every started worker, and closes the rest.
func TestToolBlockParallelCommitFailureWaitsForWorkers(t *testing.T) {
	g := newGatedHandlers(3)
	trace := &parallelTrace{}
	calls, a := parallelFixture(3, g, trace, 3, allEligible)
	commit := a.Commit
	failed := false
	a.Commit = func(ts []BlockTerminal) error {
		if !failed && len(ts) == 1 && ts[0].History.ToolCallID == "call-0" {
			failed = true
			return errors.New("commit failed")
		}
		return commit(ts)
	}
	state, _ := newToolBlockState(calls)
	done := make(chan error, 1)
	go func() {
		_, err := runSequentialBlock(context.Background(), state, calls, a, false)
		done <- err
	}()
	g.waitStarted(t, 3)
	close(g.release[0])
	select {
	case err := <-done:
		t.Fatalf("owner returned before started workers finished: %v", err)
	case <-time.After(30 * time.Millisecond):
	}
	close(g.release[2])
	close(g.release[1])
	if err := <-done; err == nil {
		t.Fatal("expected commit failure")
	}
	if len(g.returnedOrder) != 3 {
		t.Fatalf("handlers returned %v", g.returnedOrder)
	}
	// The failed commit's call and every still-open call are closed exactly
	// once (the latter by the owner's lifecycle-failure recovery).
	requireSingleTerminal(t, state)
}

// Without the in-package opt-in the executor keeps the legacy sequential
// order exactly.
func TestToolBlockParallelDefaultStaysSequential(t *testing.T) {
	var active, peak atomic.Int32
	calls, a := sequentialFixture(3, func(context.Context, int) (any, error) {
		now := active.Add(1)
		if now > peak.Load() {
			peak.Store(now)
		}
		time.Sleep(5 * time.Millisecond)
		active.Add(-1)
		return "ok", nil
	})
	state, _ := newToolBlockState(calls)
	if _, err := runSequentialBlock(context.Background(), state, calls, a, false); err != nil {
		t.Fatal(err)
	}
	if peak.Load() != 1 {
		t.Fatalf("default executor ran %d handlers concurrently", peak.Load())
	}
	requireSingleTerminal(t, state)
}
