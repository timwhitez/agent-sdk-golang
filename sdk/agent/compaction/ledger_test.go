package compaction

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
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

func TestLedgerValidateCanonicalArtifactContract(t *testing.T) {
	manifest := ledgerCanonicalManifest()
	validText := "object_ref=" + manifest.ObjectRef + " recovery_tool=" + manifest.Recovery.Tool
	tests := []struct {
		name   string
		mutate func(*LedgerReplacement)
		want   string
	}{
		{
			name: "canonical and legacy slots",
			mutate: func(repl *LedgerReplacement) {
				repl.FullArtifact = ".goode/truncated/legacy.txt"
			},
			want: "mutually exclusive",
		},
		{
			name: "ephemeral canonical manifest",
			mutate: func(repl *LedgerReplacement) {
				repl.CanonicalArtifact.Retention.Class = artifact.RetentionEphemeral
			},
			want: "durable",
		},
		{
			name: "stub omits object ref",
			mutate: func(repl *LedgerReplacement) {
				repl.ReplacementText = "recovery_tool=" + repl.CanonicalArtifact.Recovery.Tool
				repl.ReplacementHash = ContentHash(repl.ReplacementText)
			},
			want: "omits canonical object_ref",
		},
		{
			name: "stub omits recovery tool",
			mutate: func(repl *LedgerReplacement) {
				repl.ReplacementText = "object_ref=" + repl.CanonicalArtifact.ObjectRef
				repl.ReplacementHash = ContentHash(repl.ReplacementText)
			},
			want: "omits canonical recovery tool",
		},
		{
			name: "empty canonical stub",
			mutate: func(repl *LedgerReplacement) {
				repl.ReplacementText = ""
				repl.ReplacementHash = ""
			},
			want: "replacement text is required",
		},
		{
			name: "provider visible view is not a source",
			mutate: func(repl *LedgerReplacement) {
				repl.CanonicalArtifact.ObjectKind = artifact.ObjectKindProviderVisibleView
			},
			want: "provider-visible view",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clone := manifest.Clone()
			repl := LedgerReplacement{
				MessageKey:        "msg-canonical",
				PartKey:           "content-0",
				Role:              "tool",
				ToolName:          "grep",
				Tier:              "snip",
				ReplacementText:   validText,
				ReplacementHash:   ContentHash(validText),
				CanonicalArtifact: &clone,
			}
			tt.mutate(&repl)
			ledger := NewLedger("sess-canonical")
			ledger.Replacements = []LedgerReplacement{repl}
			if err := ledger.Validate("sess-canonical"); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate error = %v, want %q", err, tt.want)
			}
		})
	}

	valid := NewLedger("sess-canonical")
	valid.Replacements = []LedgerReplacement{{
		MessageKey:        "msg-canonical",
		PartKey:           "content-0",
		ReplacementText:   validText,
		ReplacementHash:   ContentHash(validText),
		CanonicalArtifact: cloneCanonicalManifestPointer(manifest),
	}}
	if err := valid.Validate("sess-canonical"); err != nil {
		t.Fatalf("valid canonical ledger: %v", err)
	}
}

func ledgerCanonicalManifest() artifact.Manifest {
	bytesCount := int64(17)
	return artifact.Manifest{
		SchemaVersion: artifact.SchemaVersion,
		ObjectRef:     "obj:v1:ledger-canonical",
		ObjectKind:    artifact.ObjectKindLogicalToolResult,
		Owner: artifact.Owner{
			WorkspaceID: "workspace",
			SubjectKind: artifact.SubjectKindSession,
			SubjectID:   "sess-canonical",
			ToolName:    "grep",
			ToolCallID:  "call-grep",
		},
		Complete:          true,
		Recoverable:       true,
		ObjectMeasurement: artifact.Measurement{Bytes: &bytesCount, SHA256: strings.Repeat("a", 64)},
		Preview:           artifact.Preview{Kind: artifact.PreviewKindNone},
		Retention: artifact.Retention{
			Class:     artifact.RetentionDurable,
			CreatedAt: time.Date(2026, 7, 20, 1, 2, 3, 0, time.UTC),
		},
		ContentType: "text/plain",
		Encoding:    artifact.EncodingUTF8,
		Recovery: artifact.Recovery{
			Capability:  "artifact.resolve.v1",
			Tool:        "artifact_read",
			Instruction: "Call artifact_read with object_ref.",
		},
	}
}

func TestLedgerStoreInterfaceCompiles(t *testing.T) {
	var _ LedgerStore = noopLedgerStore{}
}

type noopLedgerStore struct{}

func (noopLedgerStore) Load(context.Context, string) (*Ledger, error) { return NewLedger(""), nil }
func (noopLedgerStore) Save(context.Context, string, *Ledger) error   { return nil }
