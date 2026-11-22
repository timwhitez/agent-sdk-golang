package compaction

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"strings"
	"testing"
)

func TestD3ProtectedRecentTokensApplyToPriorSnips(t *testing.T) {
	const session = "review-protected"
	store := &memoryLedgerStore{ledger: NewLedger(session)}
	host := newCanonicalCompactionHost()
	cfg := canonicalCompactionConfig(session, store, host)
	messages := snipTestMessages(strings.Repeat("sample line\n", 200))
	ctx := context.Background()
	first, res, err := NewService(cfg).CompactLocal(ctx, messages, &llm.Usage{TotalTokens: 3000})
	if err != nil || !res.Compacted || !strings.Contains(first[2].PlainText(), "snipped") {
		t.Fatalf("initial snip: %+v, %v", res, err)
	}
	cfg.ProtectedRecentTokens = 10000
	saves := store.saves
	svc := NewService(cfg)
	out, res, err := svc.CompactPipeline(ctx, nil, first, PipelineRequest{EstimatedTokens: 3250, TargetWatermark: "prune"})
	if err != nil {
		t.Fatal(err)
	}
	if out[2].PlainText() != first[2].PlainText() {
		t.Fatalf("protected all %d history tokens, but upgraded protected snip to prune: tiers=%v ledger_saves=%d", svc.EstimateMessages(first), res.TiersApplied, store.saves)
	}
	if res.Compacted || store.saves != saves {
		t.Fatalf("protected preview still changed ledger: %+v saves=%d", res, store.saves)
	}
}
