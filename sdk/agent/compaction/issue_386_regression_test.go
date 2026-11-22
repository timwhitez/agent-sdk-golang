package compaction

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestLocalCompactionPreservesToolAndAssistantAttachments(t *testing.T) {
	for _, tc := range []struct {
		name   string
		role   llm.Role
		tokens int
		tier   string
	}{
		{"tool snip", llm.RoleTool, 3000, tierSnip},
		{"tool prune", llm.RoleTool, 3250, tierPrune},
		{"assistant prune", llm.RoleAssistant, 3250, tierPrune},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const session = "review-local-attachments"
			const diagram = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGOoBwAAgQCA05RTSgAAAABJRU5ErkJggg=="
			host := newCanonicalCompactionHost()
			store := &memoryLedgerStore{ledger: NewLedger(session)}
			cfg := canonicalCompactionConfig(session, store, host)
			text := strings.Repeat("report with reference to the attached diagram\n", 200)
			messages := snipTestMessages(text)
			index := 2
			if tc.role == llm.RoleAssistant {
				messages = []llm.Message{llm.NewUserMessage("Explain the attached result."), llm.NewAssistantMessage(text, nil), llm.NewUserMessage("Continue.")}
				index = 1
			}
			messages[index].Content = llm.Content{Blocks: []llm.ContentBlock{
				{Type: "image_url", ImageURL: &llm.ImageURL{URL: diagram}},
				{Type: "text", Text: text},
				{Type: "document", Source: &llm.DocSrc{Data: "c3VwcG9ydGluZyBldmlkZW5jZQ==", MediaType: "application/pdf"}},
			}}
			original := llm.CloneMessages(messages)
			svc := NewService(cfg)
			for attempt := 0; attempt < 2; attempt++ {
				out, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: tc.tokens})
				if err != nil || !res.Compacted || out[index].Content.PlainText() == text || !reflect.DeepEqual(res.TiersApplied, []string{tc.tier}) {
					t.Fatalf("fixture did not exercise local %s compaction: %+v err=%v", tc.name, res, err)
				}
				blocks := out[index].Content.Blocks
				if len(blocks) != 3 || blocks[1].Type != "text" || !reflect.DeepEqual(blocks[0], original[index].Content.Blocks[0]) || !reflect.DeepEqual(blocks[2], original[index].Content.Blocks[2]) {
					t.Fatalf("attachments changed order or data: %#v", blocks)
				}
				blocks[0].ImageURL.URL = "caller mutation"
				blocks[2].Source.Data = "caller mutation"
				if !reflect.DeepEqual(messages, original) {
					t.Fatal("compaction result aliases or changed the input")
				}
			}
			puts, _ := host.counts()
			if puts != 1 {
				t.Fatalf("ledger reuse wrote %d source objects, want 1", puts)
			}
		})
	}
}

func TestLocalCompactionPreservesInterleavedTextAndAttachments(t *testing.T) {
	for _, role := range []llm.Role{llm.RoleTool, llm.RoleAssistant} {
		t.Run(string(role), func(t *testing.T) {
			const session = "review-interleaved-attachments"
			host := newCanonicalCompactionHost()
			store := &memoryLedgerStore{ledger: NewLedger(session)}
			cfg := canonicalCompactionConfig(session, store, host)
			text := strings.Repeat("evidence paragraph\n", 300)
			messages := snipTestMessages(text)
			index := 2
			if role == llm.RoleAssistant {
				messages = []llm.Message{llm.NewUserMessage("Explain."), llm.NewAssistantMessage(text, nil), llm.NewUserMessage("Continue.")}
				index = 1
			}
			messages[index].Content = llm.Content{Blocks: []llm.ContentBlock{
				{Type: "text", Text: text}, {Type: "image_url", ImageURL: &llm.ImageURL{URL: "image"}}, {Type: "text", Text: "This refers to the image above."},
			}}
			original := llm.CloneMessages(messages)
			out, res, err := NewService(cfg).CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3250})
			puts, _ := host.counts()
			if err != nil || res.Compacted || !reflect.DeepEqual(out, original) || !reflect.DeepEqual(messages, original) || puts != 0 || len(store.ledger.Replacements) != 0 {
				t.Fatalf("ambiguous source changed: result=%+v err=%v puts=%d", res, err, puts)
			}
		})
	}
}

func TestToolPruneFromSnipPreservesAttachmentsAndRejectsAmbiguousReuse(t *testing.T) {
	const session = "review-attachment-prune-reuse"
	host := newCanonicalCompactionHost()
	store := &memoryLedgerStore{ledger: NewLedger(session)}
	cfg := canonicalCompactionConfig(session, store, host)
	messages := snipTestMessages(strings.Repeat("source material\n", 300))
	messages[2].Content = llm.Content{Blocks: []llm.ContentBlock{
		{Type: "text", Text: messages[2].PlainText()}, {Type: "image_url", ImageURL: &llm.ImageURL{URL: "image"}},
	}}
	svc := NewService(cfg)
	snipped, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3000})
	if err != nil || !res.Compacted {
		t.Fatalf("snip failed: %+v err=%v", res, err)
	}
	savedSnip := llm.CloneMessages(snipped)
	pruned, res, err := svc.CompactLocal(context.Background(), snipped, &llm.Usage{TotalTokens: 3250})
	if err != nil || !res.Compacted || !reflect.DeepEqual(res.TiersApplied, []string{tierPrune}) || len(pruned[2].Content.Blocks) != 2 || !reflect.DeepEqual(pruned[2].Content.Blocks[1], messages[2].Content.Blocks[1]) || !reflect.DeepEqual(snipped, savedSnip) {
		t.Fatalf("snip-to-prune lost attachment: %+v err=%v", res, err)
	}
	// Empty text still has a position and must not let ledger-based pruning
	// bypass the single-block mapping rule.
	snipped[2].Content.Blocks = append(snipped[2].Content.Blocks, llm.ContentBlock{Type: "text"})
	ambiguous := llm.CloneMessages(snipped)
	out, res, err := svc.CompactLocal(context.Background(), snipped, &llm.Usage{TotalTokens: 3250})
	puts, _ := host.counts()
	if err != nil || res.Compacted || !reflect.DeepEqual(out, ambiguous) || puts != 1 {
		t.Fatalf("ledger reuse bypassed source mapping: %+v err=%v puts=%d", res, err, puts)
	}
}

func TestLocalAssistantRewriteClearsOpaqueProviderState(t *testing.T) {
	for _, blockText := range []bool{false, true} {
		const session = "review-rewrite-provider-state"
		host := newCanonicalCompactionHost()
		cfg := canonicalCompactionConfig(session, &memoryLedgerStore{ledger: NewLedger(session)}, host)
		content := llm.TextContent(strings.Repeat("assistant source\n", 300))
		if blockText {
			content = llm.Content{Blocks: []llm.ContentBlock{{Type: "text", Text: content.Text}, {Type: "image_url", ImageURL: &llm.ImageURL{URL: "image"}}}}
		}
		content, err := llm.WithProviderState(content, []llm.ProviderState{{Provider: "test", Kind: "response", Data: json.RawMessage(`{"id":"original"}`)}})
		if err != nil {
			t.Fatal(err)
		}
		messages := []llm.Message{llm.NewUserMessage("Explain."), {Role: llm.RoleAssistant, Content: content}, llm.NewUserMessage("Continue.")}
		out, res, err := NewService(cfg).CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3250})
		if err != nil || !res.Compacted || llm.HasProviderState(out[1].Content) || !llm.HasProviderState(messages[1].Content) {
			t.Fatalf("rewritten assistant retained stale state or changed input: %+v err=%v", res, err)
		}
	}
}
