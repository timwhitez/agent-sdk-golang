from pathlib import Path

agent = Path("sdk/agent/agent.go")
text = agent.read_text()
old = '''\tif currentLen < pending.snapshotLen {
\t\ta.warnf("compaction apply skipped: history shrank (%d < %d); scheduling retry", currentLen, pending.snapshotLen)
\t\ta.mu.Unlock()
\t\ta.requeuePendingCompaction(pending)
'''
new = '''\tif currentLen < pending.snapshotLen {
\t\ta.mu.Unlock()
\t\ta.warnf("compaction apply skipped: history shrank (%d < %d); scheduling retry", currentLen, pending.snapshotLen)
\t\ta.requeuePendingCompaction(pending)
'''
if text.count(old) != 1:
    raise SystemExit(f"shrink warning anchor count={text.count(old)}")
text = text.replace(old, new)
old = '''\ttailCap := currentLen - pending.snapshotLen
\tmerged := llm.CloneMessages(pending.messages)
'''
new = '''\ttailCap := currentLen - pending.snapshotLen
\tpairingRepaired := false
\tmerged := llm.CloneMessages(pending.messages)
'''
if text.count(old) != 1:
    raise SystemExit(f"pairing flag anchor count={text.count(old)}")
text = text.replace(old, new)
old = '''\t\tif repaired, changed := repairToolCallPairs(merged); changed {
\t\t\ta.warnf("compaction apply repaired tool-call pairing at the summary/tail splice point")
\t\t\tmerged = repaired
\t\t}
\t}
\tpending.result = a.reconcileCompactionTelemetry(pending.result, source, merged, 0)
\ta.mu.Unlock()

\tcommit, commitErr := a.persistCompactionCheckpoint(context.Background(), merged, pending.result)
'''
new = '''\t\tif repaired, changed := repairToolCallPairs(merged); changed {
\t\t\tpairingRepaired = true
\t\t\tmerged = repaired
\t\t}
\t}
\ta.mu.Unlock()

\tif pairingRepaired {
\t\ta.warnf("compaction apply repaired tool-call pairing at the summary/tail splice point")
\t}
\t// Estimation may invoke a host-supplied TokenEstimator; keep it outside the
\t// history lock for the same re-entrancy reason as checkpoint persistence.
\tpending.result = a.reconcileCompactionTelemetry(pending.result, source, merged, 0)
\tcommit, commitErr := a.persistCompactionCheckpoint(context.Background(), merged, pending.result)
'''
if text.count(old) != 1:
    raise SystemExit(f"telemetry/warning anchor count={text.count(old)}")
agent.write_text(text.replace(old, new))

test = Path("sdk/agent/agent_compaction_callback_lock_test.go")
text = test.read_text()
text = text.replace('\t"context"\n\t"testing"\n', '\t"context"\n\t"sync"\n\t"testing"\n', 1)
text += r'''

func TestApplyPendingCompactionWarningCallbackCanReadMessages(t *testing.T) {
	var agent *Agent
	warningEntered := make(chan struct{})
	var once sync.Once
	created, err := New(Config{
		LLM: compactionCallbackModel{},
		Warningf: func(string, ...any) {
			once.Do(func() { close(warningEntered) })
			_ = agent.Messages()
		},
		Compaction: &compaction.Config{Enabled: true, ContextWindow: 4096},
	})
	if err != nil {
		t.Fatal(err)
	}
	agent = created
	agent.mu.Lock()
	agent.messages = []llm.Message{llm.NewUserMessage("short")}
	agent.mu.Unlock()
	agent.pendingCompactionMu.Lock()
	agent.pendingCompaction = &pendingCompaction{
		messages:    []llm.Message{llm.NewUserMessage("after")},
		snapshotLen: 2,
		result:      compaction.Result{Compacted: true},
	}
	agent.pendingCompactionMu.Unlock()

	done := make(chan struct{})
	go func() {
		agent.applyPendingCompaction(nil)
		close(done)
	}()
	select {
	case <-warningEntered:
	case <-time.After(time.Second):
		t.Fatal("warning callback was not entered")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("warning callback deadlocked while reading Messages")
	}
}

func TestApplyPendingCompactionTokenEstimatorCanReadMessages(t *testing.T) {
	var agent *Agent
	estimatorEntered := make(chan struct{})
	var once sync.Once
	created, err := New(Config{
		LLM: compactionCallbackModel{},
		Compaction: &compaction.Config{
			Enabled:       true,
			ContextWindow: 4096,
			TokenEstimator: func(text string) int {
				once.Do(func() { close(estimatorEntered) })
				_ = agent.Messages()
				if text == "" {
					return 0
				}
				return 1
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	agent = created
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

	done := make(chan struct{})
	go func() {
		agent.applyPendingCompaction(nil)
		close(done)
	}()
	select {
	case <-estimatorEntered:
	case <-time.After(time.Second):
		t.Fatal("token estimator was not entered")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("token estimator deadlocked while reading Messages")
	}
}
'''
test.write_text(text)
