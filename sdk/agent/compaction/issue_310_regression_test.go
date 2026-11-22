package compaction

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"strings"
	"testing"
)

func TestD3CompactLocalDefersLedgerWithCheckpointWriter(t *testing.T) {
	const session = "review-local-ledger"
	store := &memoryLedgerStore{ledger: NewLedger(session)}
	cfg := canonicalCompactionConfig(session, store, newCanonicalCompactionHost())
	cfg.CheckpointWriter = CompactionCheckpointWriterFunc(func(context.Context, CompactionCheckpoint) error { return nil })
	svc := NewService(cfg)
	_, res, err := svc.CompactLocal(context.Background(), snipTestMessages(strings.Repeat("sample line\n", 200)), &llm.Usage{TotalTokens: 3000})
	if err != nil || !res.Compacted {
		t.Fatalf("snip setup: %+v, %v", res, err)
	}
	if store.saves != 0 {
		t.Fatalf("CompactLocal wrote %d ledger versions before checkpoint commit; pending ledger=%t rollback version=%t", store.saves, res.pendingLedger != nil, res.previousLedger != nil)
	}
	if res.pendingLedger == nil || res.previousLedger == nil {
		t.Fatal("local candidate has no ledger transaction")
	}
	if err := svc.CommitPendingLedger(context.Background(), &res); err != nil || len(store.ledger.Replacements) != 1 {
		t.Fatalf("commit local ledger: %v", err)
	}
	if err := svc.RollbackPendingLedger(context.Background(), &res); err != nil || len(store.ledger.Replacements) != 0 {
		t.Fatalf("rollback local ledger: %v", err)
	}
}
