package agent

import (
	"context"
	"sync"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
)

type compactionRuntimeUpdate struct {
	service *compaction.Service
	enabled bool
}

// tryBeginCompactionRuntimeUse synchronously reserves the currently installed
// runtime when no replacement barrier is pending. QueryStream uses this before
// returning so an update made after query submission cannot overtake an already
// admitted turn. A false result means the caller must wait asynchronously with
// beginCompactionRuntimeUse instead of joining the superseded generation.
func (a *Agent) tryBeginCompactionRuntimeUse() (func(), bool) {
	if a == nil {
		return func() {}, true
	}
	a.compactionRuntimeMu.Lock()
	acquired := a.beginCompactionRuntimeUseLocked(false)
	a.compactionRuntimeMu.Unlock()
	if !acquired {
		return nil, false
	}
	return a.newCompactionRuntimeRelease(), true
}

// beginCompactionRuntimeUse starts a top-level operation. Once an update is
// queued, new top-level operations wait for the old generation to drain and
// then observe the latest replacement. This prevents a continuous stream of
// later operations from extending the superseded generation indefinitely.
func (a *Agent) beginCompactionRuntimeUse(ctx context.Context) (func(), error) {
	if a == nil {
		return func() {}, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		a.compactionRuntimeMu.Lock()
		if a.beginCompactionRuntimeUseLocked(false) {
			a.compactionRuntimeMu.Unlock()
			return a.newCompactionRuntimeRelease(), nil
		}
		wait := a.compactionRuntimeWaitCh
		if wait == nil {
			wait = make(chan struct{})
			a.compactionRuntimeWaitCh = wait
		}
		a.compactionRuntimeMu.Unlock()

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-wait:
		}
	}
}

// retainCompactionRuntimeUse extends the generation already owned by a parent
// operation. It is intentionally allowed to join a superseded generation: an
// asynchronously launched compaction is part of the turn that created it and
// must finish against the same service/configuration. Call this synchronously
// before launching the child goroutine.
func (a *Agent) retainCompactionRuntimeUse() func() {
	if a == nil {
		return func() {}
	}
	a.compactionRuntimeMu.Lock()
	_ = a.beginCompactionRuntimeUseLocked(true)
	a.compactionRuntimeMu.Unlock()
	return a.newCompactionRuntimeRelease()
}

// beginCompactionRuntimeUseLocked acquires one lifecycle use. Top-level callers
// pass joinSuperseded=false and stop at a queued-update barrier. Retained child
// work passes true so it can extend its parent's coherent generation.
func (a *Agent) beginCompactionRuntimeUseLocked(joinSuperseded bool) bool {
	if a.compactionRuntimeUses == 0 && a.pendingCompactionRuntime != nil {
		update := a.pendingCompactionRuntime
		a.pendingCompactionRuntime = nil
		a.applyCompactionRuntimeUpdateLocked(update)
		a.signalCompactionRuntimeWaitersLocked()
	}
	if !joinSuperseded && a.pendingCompactionRuntime != nil {
		return false
	}
	a.compactionRuntimeUses++
	return true
}

func (a *Agent) newCompactionRuntimeRelease() func() {
	var once sync.Once
	return func() {
		once.Do(a.finishCompactionRuntimeUse)
	}
}

func (a *Agent) finishCompactionRuntimeUse() {
	if a == nil {
		return
	}
	a.compactionRuntimeMu.Lock()
	if a.compactionRuntimeUses <= 0 {
		a.compactionRuntimeMu.Unlock()
		a.warnf("warning: compaction runtime use released without a matching acquisition")
		return
	}
	a.compactionRuntimeUses--
	if a.compactionRuntimeUses == 0 && a.pendingCompactionRuntime != nil {
		update := a.pendingCompactionRuntime
		a.pendingCompactionRuntime = nil
		a.applyCompactionRuntimeUpdateLocked(update)
		a.signalCompactionRuntimeWaitersLocked()
	}
	a.compactionRuntimeMu.Unlock()
}

// installOrQueueCompactionRuntime never blocks the caller. In particular, an
// UpdateCompactionConfig call made from a warning, estimator, checkpoint, or
// other host callback cannot deadlock the operation that invoked that callback.
// While the old generation is active, updates are coalesced and latest wins.
func (a *Agent) installOrQueueCompactionRuntime(update *compactionRuntimeUpdate) {
	if a == nil {
		return
	}
	a.compactionRuntimeMu.Lock()
	if a.compactionRuntimeUses > 0 {
		a.pendingCompactionRuntime = update
		if a.compactionRuntimeWaitCh == nil {
			a.compactionRuntimeWaitCh = make(chan struct{})
		}
		a.compactionRuntimeMu.Unlock()
		return
	}
	a.pendingCompactionRuntime = nil
	a.applyCompactionRuntimeUpdateLocked(update)
	a.signalCompactionRuntimeWaitersLocked()
	a.compactionRuntimeMu.Unlock()
}

// applyCompactionRuntimeUpdateLocked publishes one coherent service/enabled
// pair. The caller holds compactionRuntimeMu and there are no active lifecycle
// users, so every direct compactor/hasCompactor read remains race-free. State
// produced by the superseded service is invalidated rather than crossing the
// generation boundary.
func (a *Agent) applyCompactionRuntimeUpdateLocked(update *compactionRuntimeUpdate) {
	a.pendingCompactionMu.Lock()
	a.pendingCompaction = nil
	a.pendingCompactionMu.Unlock()

	if update == nil || !update.enabled || update.service == nil {
		a.compactor = nil
		a.hasCompactor = false
	} else {
		a.compactor = update.service
		a.hasCompactor = true
	}

	// Retry/cooldown state describes failures of the superseded service. Carrying
	// it into a replacement can incorrectly suppress the new configuration.
	a.compactionRetryPending.Store(false)
	a.compactionFailureStreak.Store(0)
	a.compactionCooldownUntil.Store(0)
	if !a.hasCompactor {
		a.todoCompactionPending.Store(false)
	}
}

func (a *Agent) signalCompactionRuntimeWaitersLocked() {
	wait := a.compactionRuntimeWaitCh
	a.compactionRuntimeWaitCh = nil
	if wait != nil {
		close(wait)
	}
}
