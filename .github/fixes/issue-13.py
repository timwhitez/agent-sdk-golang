from pathlib import Path

path = Path("sdk/agent/agent.go")
text = path.read_text()
old_import = '\t"os"\n\t"strings"\n'
new_import = '\t"os"\n\t"reflect"\n\t"strings"\n'
if text.count(old_import) != 1:
    raise SystemExit(f"reflect import anchor count={text.count(old_import)}")
text = text.replace(old_import, new_import)

start = text.index("func (a *Agent) applyPendingCompaction(out chan Event) {")
end = text.index("\n// CompactNow forces a compaction run", start)
new_apply = r'''func (a *Agent) applyPendingCompaction(out chan Event) {
	if !a.hasCompactor {
		return
	}
	a.pendingCompactionMu.Lock()
	pending := a.pendingCompaction
	if pending != nil {
		a.pendingCompaction = nil
	}
	a.pendingCompactionMu.Unlock()
	if pending == nil || !pending.result.Compacted {
		return
	}

	// Build an immutable candidate under the history lock, then release the
	// lock before any host-controlled ledger/checkpoint I/O. The source snapshot
	// is compared again before publication so a concurrent history mutation is
	// never overwritten by a stale compaction result.
	a.mu.Lock()
	source := llm.CloneMessages(a.messages)
	currentLen := len(source)
	if currentLen < pending.snapshotLen {
		a.warnf("compaction apply skipped: history shrank (%d < %d); scheduling retry", currentLen, pending.snapshotLen)
		a.mu.Unlock()
		a.requeuePendingCompaction(pending)
		a.compactionRetryPending.Store(true)
		return
	}
	tailCap := currentLen - pending.snapshotLen
	merged := llm.CloneMessages(pending.messages)
	if tailCap > 0 {
		merged = append(merged, llm.CloneMessages(source[pending.snapshotLen:])...)
		// pending.messages dropped every assistant tool_use block, so a tail
		// that starts inside a tool block would splice orphaned tool results
		// onto the summary. Repair rather than trust the caller to compact only
		// on user-message boundaries.
		if repaired, changed := repairToolCallPairs(merged); changed {
			a.warnf("compaction apply repaired tool-call pairing at the summary/tail splice point")
			merged = repaired
		}
	}
	pending.result = a.reconcileCompactionTelemetry(pending.result, source, merged, 0)
	a.mu.Unlock()

	commit, commitErr := a.persistCompactionCheckpoint(context.Background(), merged, pending.result)
	if commitErr != nil {
		a.requeuePendingCompaction(pending)
		a.compactionRetryPending.Store(true)
		warning := commitErr.Error()
		if len(commit.result.Warnings) > 0 {
			warning = commit.result.Warnings[len(commit.result.Warnings)-1]
		}
		a.warnf("%s", warning)
		return
	}

	a.mu.Lock()
	if !reflect.DeepEqual(a.messages, source) {
		a.mu.Unlock()
		rollbackErr := error(nil)
		if commit.persisted {
			rollbackErr = a.compactor.RollbackPendingLedger(context.Background(), &commit.transaction)
		}
		a.requeuePendingCompaction(pending)
		a.compactionRetryPending.Store(true)
		if rollbackErr != nil {
			a.warnf("compaction apply deferred because history changed while checkpoint persistence was running; ledger rollback failed: %v", rollbackErr)
		} else {
			a.warnf("compaction apply deferred because history changed while checkpoint persistence was running; ledger state was rolled back and the unreferenced checkpoint can be garbage-collected")
		}
		return
	}
	if commit.persisted {
		a.compactor.FinalizePendingLedger(&commit.transaction)
	}
	a.messages = merged
	a.resetEphemeralTrackingLocked()
	a.compactionGeneration.Add(1)
	a.mu.Unlock()

	a.emitCompactionWithAccounting(out, CompactionEvent{Result: commit.result, TriggerUsage: pending.triggerUsage})
}

func (a *Agent) requeuePendingCompaction(pending *pendingCompaction) {
	if a == nil || pending == nil {
		return
	}
	a.pendingCompactionMu.Lock()
	if a.pendingCompaction == nil {
		a.pendingCompaction = pending
	}
	a.pendingCompactionMu.Unlock()
}
'''
text = text[:start] + new_apply + text[end:]

start = text.index("func (a *Agent) CommitCompactionCheckpoint(ctx context.Context, messages []llm.Message, res compaction.Result)")
end = text.index("\n// CompactLocalNow forces the local snip/prune reducers", start)
new_commit = r'''type pendingCheckpointCommit struct {
	result      compaction.Result
	transaction compaction.Result
	persisted   bool
}

// persistCompactionCheckpoint performs all potentially blocking persistence
// without finalizing the deferred ledger transaction. The caller finalizes only
// after it has atomically published the matching in-memory history.
func (a *Agent) persistCompactionCheckpoint(ctx context.Context, messages []llm.Message, res compaction.Result) (pendingCheckpointCommit, error) {
	commit := pendingCheckpointCommit{result: res, transaction: res}
	if a == nil || !res.Compacted || a.compactor == nil || a.compactor.Config.CheckpointWriter == nil {
		return commit, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	transaction := res
	if err := a.compactor.CommitPendingLedger(ctx, &transaction); err != nil {
		warning := fmt.Sprintf("[WARN] Compaction ledger persistence failed before runtime checkpoint - original in-memory history was preserved and compaction remains retryable. (stage=save_compaction_ledger action=check ledger storage and retry: %v)", err)
		failed := res
		failed.Compacted = false
		failed.CheckpointID = ""
		failed.Warnings = append(failed.Warnings, warning)
		commit.result = failed
		return commit, fmt.Errorf("compaction ledger persistence failed: %w", err)
	}
	checkpoint, err := compaction.NewCompactionCheckpoint(messages, transaction)
	if err == nil {
		err = a.compactor.Config.CheckpointWriter.SaveCompactionCheckpoint(ctx, checkpoint)
	}
	if err != nil {
		rollbackErr := a.compactor.RollbackPendingLedger(context.Background(), &transaction)
		warning := fmt.Sprintf("[WARN] Compaction checkpoint persistence failed - original in-memory history was preserved and compaction remains retryable. (stage=append_compaction_checkpoint action=check checkpoint storage and retry: %v)", err)
		failed := res
		failed.Compacted = false
		failed.CheckpointID = ""
		failed.Warnings = append(failed.Warnings, warning)
		if rollbackErr != nil {
			rollbackWarning := fmt.Sprintf("[ERROR] Compaction ledger rollback failed - stale ledger metadata may require a safe full rebuild on retry. (stage=rollback_compaction_ledger action=check ledger storage and retry: %v)", rollbackErr)
			failed.Warnings = append(failed.Warnings, rollbackWarning)
			commit.result = failed
			return commit, fmt.Errorf("compaction checkpoint persistence failed: %w (ledger rollback failed: %v)", err, rollbackErr)
		}
		commit.result = failed
		return commit, fmt.Errorf("compaction checkpoint persistence failed: %w", err)
	}
	commit.result = checkpoint.Result
	commit.transaction = transaction
	commit.persisted = true
	return commit, nil
}

// CommitCompactionCheckpoint durably records compacted provider history before
// callers replace in-memory history. A persistence failure is fail-closed: the
// returned result is not reported as compacted and the caller keeps old state.
func (a *Agent) CommitCompactionCheckpoint(ctx context.Context, messages []llm.Message, res compaction.Result) (compaction.Result, error) {
	commit, err := a.persistCompactionCheckpoint(ctx, messages, res)
	if err != nil {
		return commit.result, err
	}
	if commit.persisted {
		a.compactor.FinalizePendingLedger(&commit.transaction)
	}
	return commit.result, nil
}
'''
text = text[:start] + new_commit + text[end:]
path.write_text(text)

Path("sdk/agent/agent_compaction_callback_lock_test.go").write_text(r'''package agent

import (
	"context"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type compactionCallbackModel struct{}

func (compactionCallbackModel) Provider() string { return "stub" }
func (compactionCallbackModel) Model() string    { return "stub" }
func (compactionCallbackModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("ok")}, nil
}

func installPendingCompactionForCallbackTest(t *testing.T, writer compaction.CompactionCheckpointWriter) *Agent {
	t.Helper()
	agent, err := New(Config{
		LLM: compactionCallbackModel{},
		Compaction: &compaction.Config{
			Enabled:          true,
			ContextWindow:    4096,
			CheckpointWriter: writer,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	agent.mu.Lock()
	agent.messages = []llm.Message{llm.NewUserMessage("before")}
	agent.mu.Unlock()
	agent.pendingCompactionMu.Lock()
	agent.pendingCompaction = &pendingCompaction{
		messages:    []llm.Message{llm.NewUserMessage("after")},
		snapshotLen: 1,
		result:      compaction.Result{Compacted: true},
	}
	agent.pendingCompactionMu.Unlock()
	return agent
}

func TestApplyPendingCompactionWriterCanReadMessages(t *testing.T) {
	var agent *Agent
	writerEntered := make(chan struct{})
	writer := compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
		close(writerEntered)
		_ = agent.Messages()
		return nil
	})
	agent = installPendingCompactionForCallbackTest(t, writer)

	done := make(chan struct{})
	go func() {
		agent.applyPendingCompaction(nil)
		close(done)
	}()

	select {
	case <-writerEntered:
	case <-time.After(time.Second):
		t.Fatal("checkpoint writer was not entered")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("applyPendingCompaction deadlocked when writer called Messages")
	}
	messages := agent.Messages()
	if len(messages) != 1 || messages[0].Content.PlainText() != "after" {
		t.Fatalf("published messages = %#v", messages)
	}
}

func TestBlockedCheckpointWriterDoesNotBlockHistoryReads(t *testing.T) {
	writerEntered := make(chan struct{})
	releaseWriter := make(chan struct{})
	writer := compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
		close(writerEntered)
		<-releaseWriter
		return nil
	})
	agent := installPendingCompactionForCallbackTest(t, writer)

	done := make(chan struct{})
	go func() {
		agent.applyPendingCompaction(nil)
		close(done)
	}()
	select {
	case <-writerEntered:
	case <-time.After(time.Second):
		t.Fatal("checkpoint writer was not entered")
	}

	readDone := make(chan struct{})
	go func() {
		_ = agent.Messages()
		close(readDone)
	}()
	select {
	case <-readDone:
	case <-time.After(250 * time.Millisecond):
		t.Fatal("Messages blocked behind checkpoint persistence")
	}
	close(releaseWriter)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("applyPendingCompaction did not finish")
	}
}
''')
