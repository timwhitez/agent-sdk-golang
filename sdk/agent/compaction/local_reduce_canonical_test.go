package compaction

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

var canonicalCompactionRecovery = artifact.Recovery{
	Capability:        "test.artifact.v1",
	Tool:              "artifact_read",
	AllowedRangeUnits: []artifact.RangeUnit{artifact.RangeUnitBytes},
	Instruction:       "Call artifact_read with object_ref, or artifact_range with object_ref, start, and end.",
}

type canonicalCompactionObject struct {
	manifest artifact.Manifest
	content  []byte
}

type canonicalCompactionHost struct {
	mu            sync.Mutex
	objects       map[string]canonicalCompactionObject
	puts          int
	resolves      int
	putErr        error
	resolveErr    error
	mutatePut     func(*artifact.Manifest)
	mutateResolve func(*artifact.ResolveResult)
}

func newCanonicalCompactionHost() *canonicalCompactionHost {
	return &canonicalCompactionHost{objects: map[string]canonicalCompactionObject{}}
}

func (h *canonicalCompactionHost) Put(_ context.Context, req artifact.PutRequest) (artifact.Manifest, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.putErr != nil {
		return artifact.Manifest{}, h.putErr
	}
	h.puts++
	ref := fmt.Sprintf("obj:v1:compaction-%d", h.puts)
	manifest := canonicalCompactionManifest(ref, req)
	stored := manifest.Clone()
	content := append([]byte(nil), req.Content...)
	h.objects[ref] = canonicalCompactionObject{manifest: stored, content: content}
	if h.mutatePut != nil {
		h.mutatePut(&manifest)
	}
	return manifest, nil
}

func (h *canonicalCompactionHost) Resolve(_ context.Context, req artifact.ResolveRequest) (artifact.ResolveResult, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.resolves++
	if h.resolveErr != nil {
		return artifact.ResolveResult{}, h.resolveErr
	}
	object, ok := h.objects[req.ObjectRef]
	if !ok {
		return artifact.ResolveResult{}, fmt.Errorf("unknown object_ref %q", req.ObjectRef)
	}
	if !reflect.DeepEqual(req.Owner, object.manifest.Owner) {
		return artifact.ResolveResult{}, fmt.Errorf("owner mismatch")
	}
	result := artifact.ResolveResult{
		Manifest: object.manifest.Clone(),
		Content:  append([]byte(nil), object.content...),
	}
	if h.mutateResolve != nil {
		h.mutateResolve(&result)
	}
	return result, nil
}

func (h *canonicalCompactionHost) counts() (puts, resolves int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.puts, h.resolves
}

func canonicalCompactionManifest(ref string, req artifact.PutRequest) artifact.Manifest {
	bytesCount := int64(len(req.Content))
	linesCount := int64(0)
	if len(req.Content) > 0 {
		linesCount = int64(strings.Count(string(req.Content), "\n") + 1)
	}
	complete := true
	visibleBytes := int64(0)
	visibleLines := int64(0)
	return artifact.Manifest{
		SchemaVersion: reqManifestSchemaVersion(),
		ObjectRef:     ref,
		ObjectKind:    req.ObjectKind,
		Owner:         req.Owner,
		Lineage:       cloneTestLineage(req.Lineage),
		Complete:      true,
		Recoverable:   true,
		ObjectMeasurement: artifact.Measurement{
			Bytes:             &bytesCount,
			Lines:             &linesCount,
			SHA256:            artifact.DigestSHA256(req.Content),
			MeasurementSource: "test_sink",
			Complete:          &complete,
		},
		VisibleMeasurement: artifact.Measurement{
			Bytes:             &visibleBytes,
			Lines:             &visibleLines,
			MeasurementSource: "test_sink",
			Complete:          &complete,
		},
		Preview:     artifact.Preview{Kind: artifact.PreviewKindNone},
		Retention:   req.Retention,
		ContentType: req.ContentType,
		Encoding:    req.Encoding,
		Recovery:    cloneCanonicalRecovery(req.Recovery),
	}
}

func reqManifestSchemaVersion() int { return artifact.SchemaVersion }

func cloneTestLineage(lineage *artifact.Lineage) *artifact.Lineage {
	if lineage == nil {
		return nil
	}
	out := *lineage
	out.DerivedFrom = append([]string(nil), lineage.DerivedFrom...)
	return &out
}

func canonicalCompactionOwner(sessionID, toolName, toolCallID string) artifact.Owner {
	return artifact.Owner{
		WorkspaceID: "workspace-v1",
		SubjectKind: artifact.SubjectKindSession,
		SubjectID:   sessionID,
		ToolName:    toolName,
		ToolCallID:  toolCallID,
	}
}

func canonicalCompactionConfig(sessionID string, store *memoryLedgerStore, host *canonicalCompactionHost) *Config {
	return &Config{
		Enabled:                 true,
		ContextWindow:           4000,
		SessionID:               sessionID,
		LedgerStore:             store,
		ProtectedRecentMessages: 1,
		ArtifactOwnerProvider: func(context.Context) (artifact.Owner, error) {
			return canonicalCompactionOwner(sessionID, "", ""), nil
		},
		ArtifactSink:     host,
		ArtifactResolver: host,
		ArtifactResolverCapability: artifact.ResolverCapability{
			Registered: true,
			Recovery:   cloneCanonicalRecovery(canonicalCompactionRecovery),
		},
		ArtifactEnvelopeCodec: artifact.JSONEnvelopeCodec{},
	}
}

func putCanonicalToolObject(t *testing.T, host *canonicalCompactionHost, owner artifact.Owner, content string, retention artifact.Retention) artifact.Manifest {
	t.Helper()
	manifest, err := host.Put(context.Background(), artifact.PutRequest{
		ObjectKind:  artifact.ObjectKindLogicalToolResult,
		Owner:       owner,
		Content:     []byte(content),
		Retention:   retention,
		ContentType: "text/plain",
		Encoding:    artifact.EncodingUTF8,
		Recovery:    cloneCanonicalRecovery(canonicalCompactionRecovery),
	})
	if err != nil {
		t.Fatalf("seed canonical object: %v", err)
	}
	if err := manifest.Validate(); err != nil {
		t.Fatalf("seed manifest invalid: %v", err)
	}
	return manifest
}

func encodeCanonicalToolEnvelope(t *testing.T, manifest artifact.Manifest, preview string) string {
	t.Helper()
	projected := manifest.Clone()
	projected.Preview = artifact.Preview{Kind: artifact.PreviewKindPrefix, Truncated: true}
	projected.VisibleMeasurement = artifact.Measurement{}
	encoded, _, err := (artifact.JSONEnvelopeCodec{}).Encode(artifact.Envelope{
		Manifest: projected,
		Preview:  preview,
		Continuation: &artifact.Continuation{
			ObjectRef: projected.ObjectRef,
			RangeUnit: artifact.RangeUnitBytes,
		},
	}, artifact.Budget{MaxBytes: 16 * 1024, MaxTokens: 16 * 1024})
	if err != nil {
		t.Fatalf("encode canonical envelope: %v", err)
	}
	return encoded
}

func TestCanonicalLocalSnipReusesValidatedEnvelopeWithoutArtifactRewrite(t *testing.T) {
	const sessionID = "sess-canonical-envelope"
	host := newCanonicalCompactionHost()
	owner := canonicalCompactionOwner(sessionID, "grep", "call-grep")
	source := strings.Repeat("source-tail-sentinel\n", 600)
	manifest := putCanonicalToolObject(t, host, owner, source, artifact.Retention{
		Class:     artifact.RetentionDurable,
		CreatedAt: time.Date(2026, 7, 20, 1, 2, 3, 0, time.UTC),
	})
	envelope := encodeCanonicalToolEnvelope(t, manifest, strings.Repeat("bounded-preview\n", 80))
	store := &memoryLedgerStore{ledger: NewLedger(sessionID)}
	svc := NewService(canonicalCompactionConfig(sessionID, store, host))
	messages := snipTestMessages(envelope)

	first, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3000})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if !res.Compacted || len(store.ledger.Replacements) != 1 {
		t.Fatalf("canonical snip result=%#v ledger=%#v", res, store.ledger)
	}
	puts, resolves := host.counts()
	if puts != 1 || resolves == 0 {
		t.Fatalf("host counts after envelope reuse: puts=%d resolves=%d, want one seed put and validation resolve", puts, resolves)
	}
	repl := store.ledger.Replacements[0]
	if repl.CanonicalArtifact == nil || repl.CanonicalArtifact.ObjectRef != manifest.ObjectRef || repl.FullArtifact != "" {
		t.Fatalf("canonical ledger replacement = %#v", repl)
	}
	stub := first[2].Content.PlainText()
	for _, want := range []string{manifest.ObjectRef, manifest.ObjectMeasurement.SHA256, "complete=true", "recoverable=true", "recovery_tool=artifact_read"} {
		if !strings.Contains(stub, want) {
			t.Fatalf("canonical stub missing %q: %s", want, stub)
		}
	}

	second, secondRes, err := svc.compactLocalWithWatermark(context.Background(), first, &llm.Usage{TotalTokens: 3200}, tierPrune)
	if err != nil {
		t.Fatalf("prune canonical stub: %v", err)
	}
	if !secondRes.Compacted || !strings.Contains(second[2].Content.PlainText(), "[Tool result pruned:") {
		t.Fatalf("canonical prune result=%#v message=%q", secondRes, second[2].Content.PlainText())
	}
	putsAfterPrune, _ := host.counts()
	if putsAfterPrune != 1 {
		t.Fatalf("prune rewrote canonical source: puts=%d", putsAfterPrune)
	}
	if len(store.ledger.Replacements) != 2 {
		t.Fatalf("ledger replacements after prune = %#v", store.ledger.Replacements)
	}
	for _, got := range store.ledger.Replacements {
		if got.CanonicalArtifact == nil || got.CanonicalArtifact.ObjectRef != manifest.ObjectRef || got.FullArtifact != "" {
			t.Fatalf("replacement lost canonical identity: %#v", got)
		}
	}

	third, thirdRes, err := svc.compactLocalWithWatermark(context.Background(), second, &llm.Usage{TotalTokens: 3200}, tierPrune)
	if err != nil {
		t.Fatalf("repeat canonical prune: %v", err)
	}
	if thirdRes.Compacted || !reflect.DeepEqual(third, second) {
		t.Fatalf("repeat prune was not a fixed point: result=%#v", thirdRes)
	}
	putsAfterRepeat, _ := host.counts()
	if putsAfterRepeat != 1 || len(store.ledger.Replacements) != 2 {
		t.Fatalf("repeat prune churned canonical state: puts=%d replacements=%d", putsAfterRepeat, len(store.ledger.Replacements))
	}
}

func TestCanonicalLocalPruneRejectsLedgerObjectDifferentFromSnipParent(t *testing.T) {
	const sessionID = "sess-canonical-prune-parent-mismatch"
	host := newCanonicalCompactionHost()
	store := &memoryLedgerStore{ledger: NewLedger(sessionID)}
	svc := NewService(canonicalCompactionConfig(sessionID, store, host))
	original := strings.Repeat("canonical source retained once\n", 500)

	snipped, snipRes, err := svc.CompactLocal(context.Background(), snipTestMessages(original), &llm.Usage{TotalTokens: 3000})
	if err != nil {
		t.Fatalf("canonical snip: %v", err)
	}
	if !snipRes.Compacted || len(store.ledger.Replacements) != 1 {
		t.Fatalf("canonical snip result=%#v ledger=%#v", snipRes, store.ledger)
	}
	parent := store.ledger.Replacements[0]
	owner := canonicalCompactionOwner(sessionID, "grep", "call-grep")
	other := putCanonicalToolObject(t, host, owner, strings.Repeat("different canonical object\n", 500), artifact.Retention{
		Class:     artifact.RetentionDurable,
		CreatedAt: time.Date(2026, 7, 20, 2, 3, 4, 0, time.UTC),
	})
	wrongText := canonicalToolPruneReplacementText(snipped[2], other)
	store.ledger.Replacements = append(store.ledger.Replacements, LedgerReplacement{
		MessageKey:            parent.MessageKey + "/tier:prune",
		PartKey:               parent.PartKey,
		Role:                  string(llm.RoleTool),
		ToolName:              "grep",
		Tier:                  tierPrune,
		OriginalHash:          parent.ReplacementHash,
		ReplacementHash:       ContentHash(wrongText),
		ReplacementText:       wrongText,
		CanonicalArtifact:     cloneCanonicalManifestPointer(other),
		ParentReplacementHash: parent.ReplacementHash,
		CreatedAt:             time.Now().UTC(),
	})
	store.saves = 0

	got, res, err := svc.compactLocalWithWatermark(context.Background(), snipped, &llm.Usage{TotalTokens: 3200}, tierPrune)
	if err != nil {
		t.Fatalf("canonical prune: %v", err)
	}
	if res.Compacted || !reflect.DeepEqual(got, snipped) {
		t.Fatalf("mismatched prune record changed snip history: result=%#v", res)
	}
	joined := strings.Join(res.Warnings, "\n")
	if !strings.Contains(joined, "object_ref does not match its snip parent") || !strings.Contains(joined, "action=leaving original") {
		t.Fatalf("mismatched prune warning = %q", joined)
	}
	if store.saves != 0 || len(store.ledger.Replacements) != 2 {
		t.Fatalf("mismatched prune record caused ledger churn: saves=%d ledger=%#v", store.saves, store.ledger)
	}
}

func TestCanonicalLocalCompactionRejectsInvalidEvidenceAndPreservesPreview(t *testing.T) {
	tests := []struct {
		name      string
		build     func(*testing.T, *canonicalCompactionHost, artifact.Owner) string
		configure func(*canonicalCompactionHost)
		want      string
	}{
		{
			name: "partial envelope",
			build: func(t *testing.T, _ *canonicalCompactionHost, owner artifact.Owner) string {
				created := time.Date(2026, 7, 20, 1, 2, 3, 0, time.UTC)
				manifest := artifact.Manifest{
					SchemaVersion: artifact.SchemaVersion,
					ObjectRef:     "obj:v1:partial",
					ObjectKind:    artifact.ObjectKindProviderVisibleView,
					Owner:         owner,
					Complete:      false,
					Recoverable:   false,
					Preview:       artifact.Preview{Kind: artifact.PreviewKindPrefix, Truncated: true},
					Retention:     artifact.Retention{Class: artifact.RetentionEphemeral, CreatedAt: created},
					ContentType:   "text/plain",
					Encoding:      artifact.EncodingUTF8,
				}
				return encodeCanonicalToolEnvelope(t, manifest, strings.Repeat("partial-preview\n", 80))
			},
			want: "complete",
		},
		{
			name: "owner mismatch",
			build: func(t *testing.T, host *canonicalCompactionHost, _ artifact.Owner) string {
				manifest := putCanonicalToolObject(t, host, canonicalCompactionOwner("another-session", "grep", "call-grep"), strings.Repeat("owner mismatch\n", 200), artifact.Retention{Class: artifact.RetentionDurable, CreatedAt: time.Now().UTC()})
				return encodeCanonicalToolEnvelope(t, manifest, strings.Repeat("owner-preview\n", 80))
			},
			want: "owner",
		},
		{
			name: "hash mismatch",
			build: func(t *testing.T, host *canonicalCompactionHost, owner artifact.Owner) string {
				manifest := putCanonicalToolObject(t, host, owner, strings.Repeat("hash mismatch\n", 200), artifact.Retention{Class: artifact.RetentionDurable, CreatedAt: time.Now().UTC()})
				manifest.ObjectMeasurement.SHA256 = strings.Repeat("0", 64)
				return encodeCanonicalToolEnvelope(t, manifest, strings.Repeat("hash-preview\n", 80))
			},
			want: "manifest",
		},
		{
			name: "ephemeral retention",
			build: func(t *testing.T, host *canonicalCompactionHost, owner artifact.Owner) string {
				manifest := putCanonicalToolObject(t, host, owner, strings.Repeat("ephemeral\n", 200), artifact.Retention{Class: artifact.RetentionEphemeral, CreatedAt: time.Now().UTC()})
				return encodeCanonicalToolEnvelope(t, manifest, strings.Repeat("ephemeral-preview\n", 80))
			},
			want: "durable",
		},
		{
			name: "resolver failure",
			build: func(t *testing.T, host *canonicalCompactionHost, owner artifact.Owner) string {
				manifest := putCanonicalToolObject(t, host, owner, strings.Repeat("resolver failure\n", 200), artifact.Retention{Class: artifact.RetentionDurable, CreatedAt: time.Now().UTC()})
				return encodeCanonicalToolEnvelope(t, manifest, strings.Repeat("resolver-preview\n", 80))
			},
			configure: func(host *canonicalCompactionHost) { host.resolveErr = errors.New("resolver unavailable") },
			want:      "resolver unavailable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const sessionID = "sess-invalid-canonical"
			host := newCanonicalCompactionHost()
			owner := canonicalCompactionOwner(sessionID, "grep", "call-grep")
			original := tt.build(t, host, owner)
			if tt.configure != nil {
				tt.configure(host)
			}
			putsBefore, _ := host.counts()
			store := &memoryLedgerStore{ledger: NewLedger(sessionID)}
			svc := NewService(canonicalCompactionConfig(sessionID, store, host))
			messages := snipTestMessages(original)

			got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3000})
			if err != nil {
				t.Fatalf("CompactLocal: %v", err)
			}
			if res.Compacted || got[2].Content.PlainText() != original {
				t.Fatalf("invalid canonical evidence changed history: result=%#v", res)
			}
			joined := strings.Join(res.Warnings, "\n")
			for _, want := range []string{"stage=compaction_artifact_validate", "action=leaving original", tt.want} {
				if !strings.Contains(joined, want) {
					t.Fatalf("warning missing %q: %q", want, joined)
				}
			}
			putsAfter, _ := host.counts()
			if putsAfter != putsBefore || store.saves != 0 || len(store.ledger.Replacements) != 0 {
				t.Fatalf("invalid evidence caused persistence: puts=%d/%d saves=%d ledger=%#v", putsBefore, putsAfter, store.saves, store.ledger)
			}
		})
	}
}

func TestCanonicalLocalPruneRejectsLegacyLedgerStubWithoutSourceBytes(t *testing.T) {
	const sessionID = "sess-legacy-stub"
	stub := "[Tool result snipped: grep tool_call_id=call-grep lines=100 bytes=5000 full_output=.goode/truncated/legacy.txt]"
	ledger := NewLedger(sessionID)
	ledger.Replacements = []LedgerReplacement{{
		MessageKey:      "legacy-message",
		PartKey:         "content-0",
		Role:            string(llm.RoleTool),
		ToolName:        "grep",
		Tier:            tierSnip,
		ReplacementHash: ContentHash(stub),
		ReplacementText: stub,
		FullArtifact:    ".goode/truncated/legacy.txt",
	}}
	store := &memoryLedgerStore{ledger: ledger}
	host := newCanonicalCompactionHost()
	svc := NewService(canonicalCompactionConfig(sessionID, store, host))
	messages := snipTestMessages(stub)

	got, res, err := svc.compactLocalWithWatermark(context.Background(), messages, &llm.Usage{TotalTokens: 3200}, tierPrune)
	if err != nil {
		t.Fatalf("prune legacy stub: %v", err)
	}
	if res.Compacted || !reflect.DeepEqual(got, messages) {
		t.Fatalf("legacy stub was promoted/pruned: result=%#v", res)
	}
	joined := strings.Join(res.Warnings, "\n")
	if !strings.Contains(joined, "legacy/unverified") || !strings.Contains(joined, "action=leaving original") {
		t.Fatalf("legacy rejection warning = %q", joined)
	}
	puts, _ := host.counts()
	if puts != 0 || store.saves != 0 || len(store.ledger.Replacements) != 1 {
		t.Fatalf("legacy stub caused canonical churn: puts=%d saves=%d ledger=%#v", puts, store.saves, store.ledger)
	}
}

func TestCanonicalLocalMigratesLegacyLedgerOnlyWhileExactSourceIsPresent(t *testing.T) {
	const sessionID = "sess-legacy-source-migration"
	original := strings.Repeat("legacy source still present\n", 300)
	messages := snipTestMessages(original)
	key := StableMessageKey(MessageKeyInput{
		Role:           string(messages[2].Role),
		ToolCallID:     messages[2].ToolCallID,
		ToolName:       messages[2].ToolName,
		OriginalText:   original,
		FirstSeenIndex: 2,
	})
	legacyText := "[Tool result snipped: grep tool_call_id=call-grep lines=300 bytes=8400 full_output=.goode/truncated/legacy-source.txt]"
	ledger := NewLedger(sessionID)
	ledger.Replacements = []LedgerReplacement{{
		MessageKey:      key,
		PartKey:         "content-0",
		Role:            string(llm.RoleTool),
		ToolName:        "grep",
		Tier:            tierSnip,
		OriginalHash:    ContentHash(original),
		ReplacementHash: ContentHash(legacyText),
		ReplacementText: legacyText,
		FullArtifact:    ".goode/truncated/legacy-source.txt",
	}}
	store := &memoryLedgerStore{ledger: ledger}
	host := newCanonicalCompactionHost()
	svc := NewService(canonicalCompactionConfig(sessionID, store, host))

	got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3000})
	if err != nil {
		t.Fatalf("migrate legacy ledger: %v", err)
	}
	if !res.Compacted || len(store.ledger.Replacements) != 1 {
		t.Fatalf("migration result=%#v ledger=%#v", res, store.ledger)
	}
	repl := store.ledger.Replacements[0]
	if repl.MessageKey != key || repl.PartKey != "content-0" || repl.CanonicalArtifact == nil || repl.FullArtifact != "" {
		t.Fatalf("legacy replacement was not migrated in place: %#v", repl)
	}
	if strings.Contains(got[2].Content.PlainText(), ".goode/truncated/legacy-source.txt") || !strings.Contains(got[2].Content.PlainText(), repl.CanonicalArtifact.ObjectRef) {
		t.Fatalf("migrated provider stub = %q", got[2].Content.PlainText())
	}
	puts, resolves := host.counts()
	if puts != 1 || resolves != 1 || store.saves != 1 {
		t.Fatalf("migration persistence counts: puts=%d resolves=%d saves=%d", puts, resolves, store.saves)
	}
}

func TestCanonicalLocalCheckpointAndLedgerRoundTripRecoverSameObject(t *testing.T) {
	const sessionID = "sess-canonical-checkpoint"
	host := newCanonicalCompactionHost()
	store := &memoryLedgerStore{ledger: NewLedger(sessionID)}
	svc := NewService(canonicalCompactionConfig(sessionID, store, host))
	original := strings.Repeat("checkpoint-source-tail\n", 500)
	messages := snipTestMessages(original)

	compacted, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3000})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if !res.Compacted || len(store.ledger.Replacements) != 1 || store.ledger.Replacements[0].CanonicalArtifact == nil {
		t.Fatalf("canonical compaction did not create descriptor: result=%#v ledger=%#v", res, store.ledger)
	}
	ref := store.ledger.Replacements[0].CanonicalArtifact.ObjectRef

	ledgerJSON, err := json.Marshal(store.ledger)
	if err != nil {
		t.Fatalf("marshal ledger: %v", err)
	}
	var resumedLedger Ledger
	if err := json.Unmarshal(ledgerJSON, &resumedLedger); err != nil {
		t.Fatalf("unmarshal ledger: %v", err)
	}
	if err := resumedLedger.Validate(sessionID); err != nil {
		t.Fatalf("validate resumed ledger: %v", err)
	}
	if got := resumedLedger.Replacements[0].CanonicalArtifact; got == nil || got.ObjectRef != ref {
		t.Fatalf("resumed ledger descriptor = %#v, want ref %q", got, ref)
	}

	checkpoint, err := NewCompactionCheckpoint(compacted, res)
	if err != nil {
		t.Fatalf("NewCompactionCheckpoint: %v", err)
	}
	checkpointJSON, err := json.Marshal(checkpoint)
	if err != nil {
		t.Fatalf("marshal checkpoint: %v", err)
	}
	var resumedCheckpoint CompactionCheckpoint
	if err := json.Unmarshal(checkpointJSON, &resumedCheckpoint); err != nil {
		t.Fatalf("unmarshal checkpoint: %v", err)
	}
	if err := resumedCheckpoint.Validate(); err != nil {
		t.Fatalf("validate resumed checkpoint: %v", err)
	}
	if !strings.Contains(resumedCheckpoint.Messages[2].Content.PlainText(), ref) {
		t.Fatalf("checkpoint seed lost opaque ref: %#v", resumedCheckpoint.Messages[2])
	}

	putsBeforeResume, _ := host.counts()
	resumedStore := &memoryLedgerStore{ledger: resumedLedger.Clone()}
	resumedService := NewService(canonicalCompactionConfig(sessionID, resumedStore, host))
	pruned, pruneRes, err := resumedService.compactLocalWithWatermark(context.Background(), resumedCheckpoint.Messages, &llm.Usage{TotalTokens: 3200}, tierPrune)
	if err != nil {
		t.Fatalf("prune resumed checkpoint: %v", err)
	}
	if !pruneRes.Compacted || !strings.Contains(pruned[2].Content.PlainText(), ref) {
		t.Fatalf("resumed prune lost canonical ref: result=%#v message=%q", pruneRes, pruned[2].Content.PlainText())
	}
	putsAfterResume, _ := host.counts()
	if putsAfterResume != putsBeforeResume {
		t.Fatalf("resumed compaction duplicated object: puts=%d before=%d", putsAfterResume, putsBeforeResume)
	}
}

func TestCanonicalLocalSinkContractFailurePreservesOriginal(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*artifact.Manifest)
		want   string
	}{
		{name: "short byte count", mutate: func(manifest *artifact.Manifest) { value := int64(1); manifest.ObjectMeasurement.Bytes = &value }, want: "byte count"},
		{name: "hash mismatch", mutate: func(manifest *artifact.Manifest) { manifest.ObjectMeasurement.SHA256 = strings.Repeat("0", 64) }, want: "sha256"},
		{name: "owner mismatch", mutate: func(manifest *artifact.Manifest) { manifest.Owner.SubjectID = "other" }, want: "owner"},
		{name: "ephemeral result", mutate: func(manifest *artifact.Manifest) { manifest.Retention.Class = artifact.RetentionEphemeral }, want: "durable"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const sessionID = "sess-sink-contract"
			host := newCanonicalCompactionHost()
			host.mutatePut = tt.mutate
			store := &memoryLedgerStore{ledger: NewLedger(sessionID)}
			svc := NewService(canonicalCompactionConfig(sessionID, store, host))
			messages := snipTestMessages(strings.Repeat("sink-contract\n", 300))
			got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3000})
			if err != nil {
				t.Fatalf("CompactLocal: %v", err)
			}
			if res.Compacted || !reflect.DeepEqual(got, messages) {
				t.Fatalf("invalid sink result changed history: result=%#v", res)
			}
			joined := strings.Join(res.Warnings, "\n")
			if !strings.Contains(joined, "stage=compaction_artifact_write") || !strings.Contains(joined, tt.want) {
				t.Fatalf("sink contract warning missing %q: %q", tt.want, joined)
			}
			if store.saves != 0 || len(store.ledger.Replacements) != 0 {
				t.Fatalf("invalid sink result reached ledger: saves=%d ledger=%#v", store.saves, store.ledger)
			}
		})
	}
}
