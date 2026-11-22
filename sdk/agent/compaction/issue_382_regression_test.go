package compaction

import (
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
)

func TestEmptyLedgerCloneDoesNotShareAppendCapacity(t *testing.T) {
	ledger := NewLedger("session")
	ledger.Replacements = make([]LedgerReplacement, 0, 1)
	first, second := ledger.Clone(), ledger.Clone()
	first.Replacements = append(first.Replacements, LedgerReplacement{MessageKey: "first", PartKey: "text"})
	second.Replacements = append(second.Replacements, LedgerReplacement{MessageKey: "second", PartKey: "text"})
	ledger.Replacements = append(ledger.Replacements, LedgerReplacement{MessageKey: "original", PartKey: "text"})
	if first.Replacements[0].MessageKey != "first" {
		t.Fatalf("independent append to second clone changed first clone to %q", first.Replacements[0].MessageKey)
	}
	if second.Replacements[0].MessageKey != "second" {
		t.Fatalf("append to original changed second clone to %q", second.Replacements[0].MessageKey)
	}
}

func TestLedgerCloneKeepsPopulatedNestedStateIndependent(t *testing.T) {
	ledger := NewLedger("session")
	ledger.Summary = &LedgerSummary{SummaryHash: "original"}
	ledger.Replacements = []LedgerReplacement{{
		MessageKey: "original",
		CanonicalArtifact: &artifact.Manifest{
			Preview: artifact.Preview{Ranges: []artifact.Range{{Unit: artifact.RangeUnitBytes, Start: 5, End: 10}}},
		},
	}}
	clone := ledger.Clone()
	clone.Summary.SummaryHash = "clone"
	clone.Replacements[0].MessageKey = "clone"
	clone.Replacements[0].CanonicalArtifact.Preview.Ranges[0].End = 7
	if ledger.Summary.SummaryHash != "original" || ledger.Replacements[0].MessageKey != "original" || ledger.Replacements[0].CanonicalArtifact.Preview.Ranges[0].End != 10 {
		t.Fatal("cloning changed nested source state")
	}
	if (*Ledger)(nil).Clone() != nil || NewLedger("empty").Clone().Replacements != nil {
		t.Fatal("nil ledger or replacement semantics changed")
	}
}
