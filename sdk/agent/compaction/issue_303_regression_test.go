package compaction

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"strings"
	"testing"
)

func TestD3TiersHonorDocumentedDecisionTokens(t *testing.T) {
	const session = "review-thresholds"
	svc := NewService(canonicalCompactionConfig(session, &memoryLedgerStore{ledger: NewLedger(session)}, newCanonicalCompactionHost()))
	// A compatible provider counts its prompt in total_tokens and separately reports the completion.
	usage := llm.NewProviderUsage(2700, 300, 2700)
	original := snipTestMessages(strings.Repeat("sample line\n", 200))
	if got := svc.WatermarkForUsage(usage); got != "snip" {
		t.Fatalf("fixture watermark is %q", got)
	}
	out, res, err := svc.CompactAuto(context.Background(), nil, original, usage, "")
	if err != nil {
		t.Fatal(err)
	}
	if !res.Compacted || out[2].PlainText() == original[2].PlainText() {
		t.Fatalf("watermark=%s, decision=%d, snip threshold=%d, raw total=%d, tiers applied=%v", res.Watermark, svc.DecisionTokens(usage), svc.snipThreshold(), usage.TotalTokens, res.TiersApplied)
	}
}
