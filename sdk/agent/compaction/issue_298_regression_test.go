package compaction

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestD3ReplayReplacementPreservesExactToolBytes(t *testing.T) {
	const session = "review-line-endings"
	host := newCanonicalCompactionHost()
	store := &memoryLedgerStore{ledger: NewLedger(session)}
	svc := NewService(canonicalCompactionConfig(session, store, host))
	ctx := context.Background()
	lf := strings.Repeat("source line\n", 200)
	crlf := strings.ReplaceAll(lf, "\n", "\r\n")
	req := PipelineRequest{EstimatedTokens: 3000, TargetWatermark: "snip"}
	_, firstRes, err := svc.CompactPipeline(ctx, nil, snipTestMessages(crlf), req)
	if err != nil || !firstRes.Compacted {
		t.Fatalf("initial snip: %+v, %v", firstRes, err)
	}
	// Durable reloads omit OriginalText, so correctness cannot depend on the
	// in-memory original retained by the first reducer.
	data, err := json.Marshal(store.ledger)
	if err != nil {
		t.Fatal(err)
	}
	store.ledger = &Ledger{}
	if err := json.Unmarshal(data, store.ledger); err != nil {
		t.Fatal(err)
	}
	next, res, err := svc.CompactPipeline(ctx, nil, snipTestMessages(lf), req)
	if err != nil {
		t.Fatal(err)
	}
	if next[2].PlainText() == lf {
		if res.Compacted || store.saves != 1 || host.puts != 1 || !strings.Contains(strings.Join(res.Warnings, " "), "source bytes") {
			t.Fatalf("mismatched source should leave both versions untouched: %+v saves=%d writes=%d", res, store.saves, host.puts)
		}
		return
	}
	for ref, obj := range host.objects {
		if strings.Contains(next[2].PlainText(), ref) {
			if string(obj.content) == lf {
				return
			}
			t.Fatalf("replayed replacement recovers %d bytes, current tool output has %d bytes, tiers=%v, writes=%d", len(obj.content), len(lf), res.TiersApplied, host.puts)
		}
	}
	t.Fatal("changed history without a recoverable exact source")
}
