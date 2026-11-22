package compaction

import (
	"bytes"
	"context"
	"encoding/json"
	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"strings"
	"testing"
	"time"
)

func TestD3CanonicalEnvelopeWithJSONEquivalentTime(t *testing.T) {
	const session = "review-time"
	host := newCanonicalCompactionHost()
	store := &memoryLedgerStore{ledger: NewLedger(session)}
	cfg := canonicalCompactionConfig(session, store, host)
	content := strings.Repeat("sample line\n", 200)
	when := time.Date(2024, 1, 3, 8, 0, 0, 0, time.FixedZone("store_clock", 3600))
	manifest := putCanonicalToolObject(t, host, canonicalCompactionOwner(session, "grep", "call-grep"), content, artifact.Retention{Class: artifact.RetentionDurable, CreatedAt: when})
	encoded := encodeCanonicalToolEnvelope(t, manifest, content[:100])
	decoded, ok, err := (artifact.JSONEnvelopeCodec{}).Decode(encoded)
	if !ok || err != nil {
		t.Fatal(err)
	}
	a, _ := json.Marshal(manifest.Retention)
	b, _ := json.Marshal(decoded.Manifest.Retention)
	if !bytes.Equal(a, b) || !decoded.Manifest.Retention.CreatedAt.Equal(manifest.Retention.CreatedAt) {
		t.Fatal("fixture changed retention")
	}
	out, res, err := NewService(cfg).CompactPipeline(context.Background(), nil, snipTestMessages(encoded), PipelineRequest{TargetWatermark: "snip", EstimatedTokens: 3000})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Compacted || out[2].PlainText() == encoded {
		t.Fatalf("same serialized retention on both sides %s, valid canonical envelope refused: %v", a, res.Warnings)
	}
}

func TestCanonicalManifestRetentionIdentity(t *testing.T) {
	when := time.Date(2024, 1, 3, 8, 0, 0, 0, time.FixedZone("host", 3600))
	gc, expiry := when.Add(time.Hour), when.Add(2*time.Hour)
	left := artifact.Manifest{ObjectRef: "original", Retention: artifact.Retention{Class: artifact.RetentionDurable, CreatedAt: when, GCEligibleAfter: &gc, ExpiresAt: &expiry}}
	right := left.Clone()
	right.Retention.CreatedAt = when.UTC()
	*right.Retention.GCEligibleAfter = gc.UTC()
	*right.Retention.ExpiresAt = expiry.UTC()
	if !canonicalManifestIdentityEqual(left, right) {
		t.Fatal("equivalent retention instants differ")
	}
	if left.Retention.CreatedAt.Location().String() != "host" || left.Retention.GCEligibleAfter.Location().String() != "host" {
		t.Fatal("comparison mutated caller retention")
	}
	*right.Retention.ExpiresAt = expiry.Add(time.Nanosecond)
	if canonicalManifestIdentityEqual(left, right) {
		t.Fatal("changed expiry was accepted")
	}
	right = left.Clone()
	right.ObjectRef = "other"
	if canonicalManifestIdentityEqual(left, right) {
		t.Fatal("changed object identity was accepted")
	}
}
