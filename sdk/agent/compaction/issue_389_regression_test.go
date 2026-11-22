package compaction

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestR2CanonicalSnipDoesNotDiscardTrailingFramedSource(t *testing.T) {
	const session = "r2-trailing-frame"
	for _, suffix := range []string{"second_envelope", "ordinary_tail_and_marker"} {
		t.Run(suffix, func(t *testing.T) {
			host := newCanonicalCompactionHost()
			owner := canonicalCompactionOwner(session, "grep", "call-grep")
			retention := artifact.Retention{Class: artifact.RetentionDurable, CreatedAt: time.Now().UTC()}
			first := putCanonicalToolObject(t, host, owner, strings.Repeat("first source\n", 300), retention)
			content := encodeCanonicalToolEnvelope(t, first, strings.Repeat("first preview\n", 100))
			if suffix == "second_envelope" {
				second := putCanonicalToolObject(t, host, owner, strings.Repeat("second source must remain recoverable\n", 300), retention)
				content += "\n" + encodeCanonicalToolEnvelope(t, second, strings.Repeat("second preview\n", 100))
			} else {
				content += "\n" + strings.Repeat("trailing source must not disappear\n", 300) + artifact.EnvelopeEndMarker
			}
			if !artifact.IsFramedEnvelope(content) {
				t.Fatal("fixture must reach default framed-envelope handling")
			}
			for _, reuse := range []bool{false, true} {
				store := &memoryLedgerStore{ledger: NewLedger(session)}
				messages := snipTestMessages(content)
				if reuse {
					// An old ledger may contain a replacement accepted by the
					// earlier prefix-only decoder. Revalidation must reject it.
					msg := messages[2]
					replacement := canonicalSnipReplacementText(msg, content, first)
					store.ledger.Replacements = []LedgerReplacement{{
						MessageKey: StableMessageKey(MessageKeyInput{Role: string(msg.Role), ToolCallID: msg.ToolCallID, ToolName: msg.ToolName, OriginalText: content, FirstSeenIndex: 2}),
						PartKey:    "content-0", Role: string(msg.Role), ToolName: msg.ToolName, Tier: tierSnip,
						OriginalHash: ContentHash(content), ReplacementHash: ContentHash(replacement), ReplacementText: replacement,
						CanonicalArtifact: cloneCanonicalManifestPointer(first), CreatedAt: time.Now().UTC(),
					}}
				}
				before := store.ledger.Clone()
				svc := NewService(canonicalCompactionConfig(session, store, host))
				out, res, err := svc.CompactLocal(t.Context(), messages, &llm.Usage{TotalTokens: 3000})
				if err != nil {
					t.Fatal(err)
				}
				if res.Compacted || !reflect.DeepEqual(out, messages) || !reflect.DeepEqual(store.ledger, before) {
					t.Fatalf("trailing framed source discarded (reuse=%v): compacted=%v replacements=%d result=%q", reuse, res.Compacted, len(store.ledger.Replacements), out[2].PlainText())
				}
			}
		})
	}
}
