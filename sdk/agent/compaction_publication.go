package agent

import (
	"context"
	"errors"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// ErrStaleCompactionHistory means the candidate's source history no longer
// matches the Agent. No checkpoint or history publication was attempted.
var ErrStaleCompactionHistory = errors.New("agent: compaction source history changed before publication")

// beginManualCompaction shares the synchronous query admission boundary.
// Runtime uses acquired afterward must be released before this qualification.
func (a *Agent) beginManualCompaction(ctx context.Context) (func(), error) {
	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	a.mu.Lock()
	if !a.turnActive.CompareAndSwap(false, true) {
		a.mu.Unlock()
		return nil, ErrAgentBusy
	}
	a.manualCompactionActive = true
	a.mu.Unlock()
	return func() {
		a.mu.Lock()
		a.manualCompactionActive = false
		a.turnActive.Store(false)
		a.mu.Unlock()
	}, nil
}

// CommitCompactionHistory commits and publishes a host-computed candidate
// under one query/manual-history admission boundary. expected must be the same
// snapshot used to compute messages, not a fresh read made just before commit.
// Stale history, pending work and admission conflicts reject before persistence.
// A successful writer acknowledgement is followed by publication even if ctx
// was canceled in the meantime; this is not an automatic durable rollback.
// Without a writer, publication is memory-only and no CheckpointID is invented.
// This content check is not a host session/runtime revision or an external-store
// transaction. Legacy checkpoint-only calls and external writers are not fenced.
func (a *Agent) CommitCompactionHistory(ctx context.Context, expected, messages []llm.Message, res compaction.Result) (compaction.Result, error) {
	failed := res
	failed.Compacted, failed.CheckpointID, failed.CheckpointMessages = false, "", 0
	releasePublication, err := a.beginManualCompaction(ctx)
	if err != nil {
		return failed, err
	}
	defer releasePublication()
	if !res.Compacted {
		return failed, nil
	}
	source, candidate := llm.CloneMessages(expected), llm.CloneMessages(messages)
	releaseRuntime, acquired := a.tryBeginCompactionRuntimeUse()
	if !acquired {
		return failed, ErrAgentBusy
	}
	defer releaseRuntime()
	if !a.compactionInFlight.CompareAndSwap(false, true) {
		return failed, ErrAgentBusy
	}
	defer a.releaseCompactionInFlight()
	if a.hasPendingCompaction() {
		return failed, ErrAgentBusy
	}
	a.mu.Lock()
	matches := (len(a.messages) == 0 && len(source) == 0) || messageJSONEqual(a.messages, source)
	a.mu.Unlock()
	if !matches {
		return failed, ErrStaleCompactionHistory
	}
	if ctx != nil && ctx.Err() != nil {
		return failed, ctx.Err()
	}
	res.CheckpointID, res.CheckpointMessages = "", 0
	commit, err := a.persistCompactionCheckpoint(ctx, candidate, res)
	if err != nil {
		return commit.result, err
	}
	a.mu.Lock()
	a.messages = candidate
	a.resetEphemeralTrackingLocked()
	a.compactionGeneration.Add(1)
	a.mu.Unlock()
	if commit.persisted {
		a.compactor.FinalizePendingLedger(&commit.transaction)
	}
	return commit.result, nil
}
