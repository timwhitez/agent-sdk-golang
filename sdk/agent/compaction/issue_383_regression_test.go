package compaction

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestCanonicalSnipAcceptsLiteralEnvelopeMarkers(t *testing.T) {
	for _, codec := range []artifact.EnvelopeCodec{nil, artifact.JSONEnvelopeCodec{}, &artifact.JSONEnvelopeCodec{}} {
		for _, marker := range []string{artifact.EnvelopeStartMarker, artifact.EnvelopeEndMarker, artifact.EnvelopeStartMarker + " text " + artifact.EnvelopeEndMarker} {
			t.Run(marker, func(t *testing.T) {
				const session = "literal-marker"
				host := newCanonicalCompactionHost()
				store := &memoryLedgerStore{ledger: NewLedger(session)}
				cfg := canonicalCompactionConfig(session, store, host)
				cfg.ArtifactEnvelopeCodec = codec
				svc := NewService(cfg)
				content := "// Example marker: " + marker + "\n" + strings.Repeat("ordinary source content\n", 300)
				messages := snipTestMessages(content)
				out, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3000})
				if err != nil || !res.Compacted || out[2].PlainText() == content {
					t.Fatalf("ordinary source mentioning marker could not compact: compacted=%v err=%v warnings=%v", res.Compacted, err, res.Warnings)
				}
				if len(store.ledger.Replacements) != 1 {
					t.Fatalf("replacements=%d", len(store.ledger.Replacements))
				}
				_, repeated, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3000})
				if err != nil || !repeated.Compacted {
					t.Fatalf("reuse of ordinary marker text failed: %+v err=%v", repeated, err)
				}
				puts, _ := host.counts()
				if puts != 1 {
					t.Fatalf("same source generated %d objects, want one", puts)
				}
			})
		}
	}
}

func TestCanonicalSnipRejectsMalformedFramedEnvelope(t *testing.T) {
	const session = "malformed-frame"
	host := newCanonicalCompactionHost()
	store := &memoryLedgerStore{ledger: NewLedger(session)}
	svc := NewService(canonicalCompactionConfig(session, store, host))
	content := artifact.EnvelopeStartMarker + "\n" + strings.Repeat("invalid envelope\n", 300) + artifact.EnvelopeEndMarker
	messages := snipTestMessages(content)
	out, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3000})
	puts, _ := host.counts()
	if err != nil || res.Compacted || out[2].PlainText() != content || puts != 0 || len(store.ledger.Replacements) != 0 || !strings.Contains(strings.Join(res.Warnings, "\n"), "artifact envelope decode") {
		t.Fatalf("malformed framed envelope accepted: result=%+v err=%v puts=%d", res, err, puts)
	}
}

type literalMarkerTestCodec struct {
	artifact.JSONEnvelopeCodec
	calls int
	err   error
}

func (c *literalMarkerTestCodec) Decode(string) (artifact.Envelope, bool, error) {
	c.calls++
	return artifact.Envelope{}, c.err != nil, c.err
}

func TestCanonicalSnipDefersRecognitionToCustomCodec(t *testing.T) {
	for _, reject := range []bool{false, true} {
		const session = "custom-frame"
		host := newCanonicalCompactionHost()
		store := &memoryLedgerStore{ledger: NewLedger(session)}
		cfg := canonicalCompactionConfig(session, store, host)
		codec := &literalMarkerTestCodec{}
		if reject {
			codec.err = errors.New("custom envelope rejected")
		}
		cfg.ArtifactEnvelopeCodec = codec
		svc := NewService(cfg)
		content := "// quoted " + artifact.EnvelopeStartMarker + strings.Repeat("ordinary source\n", 300)
		messages := snipTestMessages(content)
		for attempt := 0; attempt < 2; attempt++ {
			_, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3000})
			if err != nil || res.Compacted == reject {
				t.Fatalf("custom codec recognition ignored: reject=%v result=%+v err=%v", reject, res, err)
			}
		}
		if codec.calls != 2 {
			t.Fatalf("custom codec invoked %d times for archive and reuse, want 2", codec.calls)
		}
	}
}
