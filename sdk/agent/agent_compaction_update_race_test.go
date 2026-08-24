package agent

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type blockingCompactionUpdateModel struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
	calls   atomic.Int32
}

func (m *blockingCompactionUpdateModel) Provider() string { return "stub" }
func (m *blockingCompactionUpdateModel) Model() string    { return "stub" }
func (m *blockingCompactionUpdateModel) Invoke(ctx context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls.Add(1)
	m.once.Do(func() { close(m.started) })
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-m.release:
		return &llm.Completion{Content: llm.TextContent("ok")}, nil
	}
}

func drainCompactionUpdateTurn(t *testing.T, events <-chan Event) {
	t.Helper()
	timeout := time.NewTimer(10 * time.Second)
	defer timeout.Stop()
	for {
		select {
		case _, ok := <-events:
			if !ok {
				return
			}
		case <-timeout.C:
			t.Fatal("timed out waiting for turn to finish")
		}
	}
}

func newCompactionUpdateAgent(t *testing.T, model llm.ChatModel, window int) *Agent {
	t.Helper()
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:       true,
			ContextWindow: window,
		},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	return ag
}

func compactionRuntimeWindow(a *Agent) int {
	if a == nil || !a.hasCompactor || a.compactor == nil {
		return 0
	}
	return a.compactor.ContextWindow
}

func TestUpdateCompactionConfigDefersAlternatingUpdatesUntilActiveTurnCompletes(t *testing.T) {
	model := &blockingCompactionUpdateModel{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	ag := newCompactionUpdateAgent(t, model, 1000)

	events := ag.QueryStream(context.Background(), llm.TextContent("hold"))
	select {
	case <-model.started:
	case <-time.After(10 * time.Second):
		t.Fatal("provider invocation did not start")
	}

	const updates = 500
	const finalWindow = 2000 + updates - 1
	firstDisableQueued := make(chan struct{})
	continueUpdates := make(chan struct{})
	updatesDone := make(chan struct{})
	go func() {
		defer close(updatesDone)
		ag.UpdateCompactionConfig(&compaction.Config{Enabled: false})
		close(firstDisableQueued)
		<-continueUpdates
		for i := 0; i < updates; i++ {
			ag.UpdateCompactionConfig(&compaction.Config{Enabled: false})
			ag.UpdateCompactionConfig(&compaction.Config{
				Enabled:       true,
				ContextWindow: 2000 + i,
			})
			runtime.Gosched()
		}
	}()

	<-firstDisableQueued
	activeRuntimeStable := true
	observedEnabled := ag.hasCompactor
	observedService := ag.compactor
	observedWindow := compactionRuntimeWindow(ag)
	if !observedEnabled || observedService == nil || observedWindow != 1000 {
		activeRuntimeStable = false
	}

	close(continueUpdates)
	for activeRuntimeStable {
		select {
		case <-updatesDone:
			goto updatesFinished
		default:
			observedEnabled = ag.hasCompactor
			observedService = ag.compactor
			observedWindow = compactionRuntimeWindow(ag)
			if !observedEnabled || observedService == nil || observedWindow != 1000 {
				activeRuntimeStable = false
				break
			}
			runtime.Gosched()
		}
	}
	<-updatesDone

updatesFinished:
	close(model.release)
	drainCompactionUpdateTurn(t, events)

	if !activeRuntimeStable {
		t.Fatalf("active turn observed replacement runtime: enabled=%v service=%#v window=%d", observedEnabled, observedService, observedWindow)
	}
	if got := compactionRuntimeWindow(ag); got != finalWindow {
		t.Fatalf("installed context window = %d, want %d", got, finalWindow)
	}
	if got := model.calls.Load(); got != 1 {
		t.Fatalf("provider calls = %d, want 1", got)
	}
}

func TestCompactionRuntimeUpdateBlocksLaterTopLevelUse(t *testing.T) {
	ag := newCompactionUpdateAgent(t, noopCompactionUpdateModel{}, 1000)
	releaseOld, acquired := ag.tryBeginCompactionRuntimeUse()
	if !acquired {
		t.Fatal("initial runtime acquisition unexpectedly blocked")
	}
	ag.UpdateCompactionConfig(&compaction.Config{Enabled: true, ContextWindow: 2000})

	type result struct {
		release func()
		window  int
		err     error
	}
	resultCh := make(chan result, 1)
	go func() {
		release, err := ag.beginCompactionRuntimeUse(context.Background())
		if err != nil {
			resultCh <- result{err: err}
			return
		}
		resultCh <- result{release: release, window: compactionRuntimeWindow(ag)}
	}()

	select {
	case got := <-resultCh:
		if got.release != nil {
			got.release()
		}
		t.Fatalf("later operation joined superseded runtime: window=%d err=%v", got.window, got.err)
	case <-time.After(50 * time.Millisecond):
	}

	releaseOld()
	select {
	case got := <-resultCh:
		if got.err != nil {
			t.Fatalf("later operation failed: %v", got.err)
		}
		defer got.release()
		if got.window != 2000 {
			t.Fatalf("later operation observed window %d, want 2000", got.window)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("later operation did not resume after replacement")
	}
}

func TestRetainedChildKeepsParentGenerationWhileLaterOperationWaits(t *testing.T) {
	ag := newCompactionUpdateAgent(t, noopCompactionUpdateModel{}, 1000)
	releaseParent, acquired := ag.tryBeginCompactionRuntimeUse()
	if !acquired {
		t.Fatal("initial runtime acquisition unexpectedly blocked")
	}
	releaseChild := ag.retainCompactionRuntimeUse()
	ag.UpdateCompactionConfig(&compaction.Config{Enabled: true, ContextWindow: 2000})
	releaseParent()

	if got := compactionRuntimeWindow(ag); got != 1000 {
		t.Fatalf("retained child observed window %d, want old window 1000", got)
	}

	resumed := make(chan int, 1)
	go func() {
		release, err := ag.beginCompactionRuntimeUse(context.Background())
		if err != nil {
			resumed <- -1
			return
		}
		defer release()
		resumed <- compactionRuntimeWindow(ag)
	}()
	select {
	case got := <-resumed:
		t.Fatalf("later operation resumed before retained child ended: window=%d", got)
	case <-time.After(50 * time.Millisecond):
	}

	releaseChild()
	select {
	case got := <-resumed:
		if got != 2000 {
			t.Fatalf("later operation observed window %d, want 2000", got)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("later operation did not resume after retained child ended")
	}
}

func TestCompactionRuntimeWaitHonorsContextCancellationWithoutLeakingUse(t *testing.T) {
	ag := newCompactionUpdateAgent(t, noopCompactionUpdateModel{}, 1000)
	releaseOld, acquired := ag.tryBeginCompactionRuntimeUse()
	if !acquired {
		t.Fatal("initial runtime acquisition unexpectedly blocked")
	}
	ag.UpdateCompactionConfig(&compaction.Config{Enabled: true, ContextWindow: 2000})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if release, err := ag.beginCompactionRuntimeUse(ctx); !errors.Is(err, context.Canceled) {
		if release != nil {
			release()
		}
		t.Fatalf("wait error = %v, want context.Canceled", err)
	}
	releaseOld()

	releaseNew, err := ag.beginCompactionRuntimeUse(context.Background())
	if err != nil {
		t.Fatalf("acquire after canceled waiter: %v", err)
	}
	defer releaseNew()
	if got := compactionRuntimeWindow(ag); got != 2000 {
		t.Fatalf("runtime after canceled waiter = %d, want 2000", got)
	}
}

func TestCompactionRuntimeQueuedUpdatesAreLatestWins(t *testing.T) {
	ag := newCompactionUpdateAgent(t, noopCompactionUpdateModel{}, 1000)
	releaseOld, acquired := ag.tryBeginCompactionRuntimeUse()
	if !acquired {
		t.Fatal("initial runtime acquisition unexpectedly blocked")
	}
	ag.UpdateCompactionConfig(&compaction.Config{Enabled: true, ContextWindow: 2000})
	ag.UpdateCompactionConfig(&compaction.Config{Enabled: false})
	ag.UpdateCompactionConfig(&compaction.Config{Enabled: true, ContextWindow: 3000})
	releaseOld()

	releaseNew, err := ag.beginCompactionRuntimeUse(context.Background())
	if err != nil {
		t.Fatalf("acquire latest runtime: %v", err)
	}
	defer releaseNew()
	if got := compactionRuntimeWindow(ag); got != 3000 {
		t.Fatalf("latest queued runtime window = %d, want 3000", got)
	}
}

func TestBusyTurnRejectionDoesNotAcquireCompactionRuntime(t *testing.T) {
	model := &blockingCompactionUpdateModel{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	ag := newCompactionUpdateAgent(t, model, 1000)
	first := ag.QueryStream(context.Background(), llm.TextContent("first"))
	select {
	case <-model.started:
	case <-time.After(10 * time.Second):
		t.Fatal("first provider invocation did not start")
	}

	ag.compactionRuntimeMu.Lock()
	usesBefore := ag.compactionRuntimeUses
	ag.compactionRuntimeMu.Unlock()
	second := ag.QueryStream(context.Background(), llm.TextContent("second"))
	drainCompactionUpdateTurn(t, second)
	ag.compactionRuntimeMu.Lock()
	usesAfter := ag.compactionRuntimeUses
	ag.compactionRuntimeMu.Unlock()
	if usesAfter != usesBefore {
		t.Fatalf("busy rejection changed runtime uses from %d to %d", usesBefore, usesAfter)
	}

	close(model.release)
	drainCompactionUpdateTurn(t, first)
	if got := model.calls.Load(); got != 1 {
		t.Fatalf("provider calls = %d, want 1", got)
	}
}

func TestRuntimeReplacementInvalidatesSupersededState(t *testing.T) {
	ag := newCompactionUpdateAgent(t, noopCompactionUpdateModel{}, 1000)
	ag.todoCompactionPending.Store(true)
	ag.compactionRetryPending.Store(true)
	ag.compactionFailureStreak.Store(3)
	ag.compactionCooldownUntil.Store(time.Now().Add(time.Hour).UnixNano())
	ag.pendingCompactionMu.Lock()
	ag.pendingCompaction = &pendingCompaction{}
	ag.pendingCompactionMu.Unlock()

	ag.UpdateCompactionConfig(&compaction.Config{Enabled: true, ContextWindow: 2000})
	if ag.compactionRetryPending.Load() {
		t.Fatal("replacement inherited retry state from superseded compactor")
	}
	if got := ag.compactionFailureStreak.Load(); got != 0 {
		t.Fatalf("replacement inherited failure streak %d", got)
	}
	if got := ag.compactionCooldownUntil.Load(); got != 0 {
		t.Fatalf("replacement inherited cooldown %d", got)
	}
	if !ag.todoCompactionPending.Load() {
		t.Fatal("enabled replacement unexpectedly discarded independent todo checkpoint signal")
	}
	ag.pendingCompactionMu.Lock()
	pending := ag.pendingCompaction
	ag.pendingCompactionMu.Unlock()
	if pending != nil {
		t.Fatal("replacement retained a result produced by the superseded compactor")
	}

	ag.UpdateCompactionConfig(&compaction.Config{Enabled: false})
	if ag.todoCompactionPending.Load() {
		t.Fatal("disabled replacement retained todo checkpoint signal")
	}
	if ag.hasCompactor || ag.compactor != nil {
		t.Fatalf("disabled replacement not installed: enabled=%v service=%#v", ag.hasCompactor, ag.compactor)
	}
}
