package compaction

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestLedgerStableMessageKeyIncludesRoleToolHashAndIndex(t *testing.T) {
	keyA := StableMessageKey(MessageKeyInput{
		Role:           "tool",
		ToolCallID:     "call-1",
		ToolName:       "grep",
		OriginalText:   " line one \r\nline two\n",
		FirstSeenIndex: 7,
	})
	keyB := StableMessageKey(MessageKeyInput{
		Role:           " tool ",
		ToolCallID:     "call-1",
		ToolName:       "grep",
		OriginalText:   "line one\nline two",
		FirstSeenIndex: 7,
	})
	if keyA == "" {
		t.Fatal("StableMessageKey returned empty key")
	}
	if keyA != keyB {
		t.Fatalf("stable key mismatch for normalized content:\n%s\n%s", keyA, keyB)
	}
	for _, want := range []string{"tool", "call-1", "grep", "idx:7", "sha256:"} {
		if !strings.Contains(keyA, want) {
			t.Fatalf("stable key %q missing %q", keyA, want)
		}
	}
}

func TestLedgerValidateRejectsHashMismatchAndDuplicates(t *testing.T) {
	repl := LedgerReplacement{
		MessageKey:      "msg-1",
		PartKey:         "content-0",
		Tier:            "snip",
		ReplacementText: "replacement",
		ReplacementHash: ContentHash("different"),
	}
	ledger := NewLedger("sess-1")
	ledger.Replacements = []LedgerReplacement{repl}
	if err := ledger.Validate("sess-1"); err == nil || !strings.Contains(err.Error(), "replacement hash") {
		t.Fatalf("Validate hash mismatch error = %v", err)
	}

	repl.ReplacementHash = ContentHash(repl.ReplacementText)
	ledger.Replacements = []LedgerReplacement{repl, repl}
	if err := ledger.Validate("sess-1"); err == nil || !strings.Contains(err.Error(), "duplicate replacement") {
		t.Fatalf("Validate duplicate error = %v", err)
	}
}

func TestLedgerValidateAcceptsSummaryAndReplacementSchema(t *testing.T) {
	created := time.Date(2026, 6, 8, 1, 2, 3, 0, time.UTC)
	ledger := NewLedger("sess-2")
	ledger.UpdatedAt = created
	ledger.ContextWindow = 128000
	ledger.PolicyHash = ContentHash("policy")
	ledger.Summary = &LedgerSummary{
		Version:         1,
		MessageName:     CompactionSummaryMessageName,
		SummaryHash:     ContentHash("summary"),
		CoveredStartKey: "start",
		CoveredEndKey:   "end",
		SourceSnapshot:  ".goode/truncated/context.md",
	}
	ledger.Replacements = []LedgerReplacement{{
		MessageKey:            "msg-1",
		PartKey:               "content-0",
		Role:                  "tool",
		ToolName:              "grep",
		Tier:                  "snip",
		OriginalHash:          ContentHash("original"),
		ReplacementHash:       ContentHash("replacement"),
		ReplacementText:       "replacement",
		FullArtifact:          ".goode/truncated/tool.txt",
		ParentReplacementHash: "",
		CreatedAt:             created,
		OriginalText:          "original",
	}}

	if err := ledger.Validate("sess-2"); err != nil {
		t.Fatalf("Validate returned error: %v", err)
	}
	if ledger.SchemaVersion != LedgerSchemaVersion {
		t.Fatalf("schema version = %d, want %d", ledger.SchemaVersion, LedgerSchemaVersion)
	}
}

func TestLedgerStoreInterfaceCompiles(t *testing.T) {
	var _ LedgerStore = noopLedgerStore{}
}

type noopLedgerStore struct{}

func (noopLedgerStore) Load(context.Context, string) (*Ledger, error) { return NewLedger(""), nil }
func (noopLedgerStore) Save(context.Context, string, *Ledger) error   { return nil }
