package compaction

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestR2LegacyAssistantReductionKeepsAttachmentsAndClearsState(t *testing.T) {
	text := strings.Repeat("Long assistant analysis alongside the attachment.\n", 300)
	content := llm.Content{Blocks: []llm.ContentBlock{
		{Type: "document", Source: &llm.DocSrc{Data: "fixture", MediaType: "application/pdf"}},
		{Type: "text", Text: text},
		{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://example.invalid/image"}},
	}}
	content, err := llm.WithProviderState(content, []llm.ProviderState{{Provider: "test", Kind: "history", Data: json.RawMessage(`{"id":"opaque-original"}`)}})
	if err != nil {
		t.Fatal(err)
	}
	messages := []llm.Message{llm.NewUserMessage("Analyze"), {Role: llm.RoleAssistant, Content: content}, llm.NewUserMessage("Continue")}
	original := llm.CloneMessages(messages)
	store := &memoryLedgerStore{ledger: NewLedger("r2-legacy-attachments")}
	writes := 0
	svc := NewService(&Config{
		Enabled: true, ContextWindow: 4000, ProtectedRecentMessages: 1,
		SessionID: "r2-legacy-attachments", LedgerStore: store,
		ToolArtifactWriter: ArtifactWriterFunc(func(_ context.Context, req ArtifactRequest) (ArtifactResult, error) {
			writes++
			if req.Content != text {
				t.Fatal("archive did not contain exactly the rewritten text block")
			}
			return ArtifactResult{Path: "compat/archive.txt"}, nil
		}),
	})
	for attempt := 0; attempt < 2; attempt++ {
		out, res, err := svc.CompactLocal(t.Context(), messages, &llm.Usage{TotalTokens: 3250})
		if err != nil || !res.Compacted || llm.HasProviderState(out[1].Content) {
			t.Fatalf("legacy reduction failed: %+v err=%v", res, err)
		}
		blocks := out[1].Content.Blocks
		if len(blocks) != 3 || blocks[1].Text == text || !reflect.DeepEqual(blocks[0], original[1].Content.Blocks[0]) || !reflect.DeepEqual(blocks[2], original[1].Content.Blocks[2]) {
			t.Fatalf("legacy attachment boundary changed: %+v", blocks)
		}
		blocks[0].Source.Data = "changed by output consumer"
		blocks[2].ImageURL.URL = "changed by output consumer"
		if !reflect.DeepEqual(messages, original) {
			t.Fatal("output attachments alias source history")
		}
	}
	if writes != 1 {
		t.Fatalf("legacy ledger reuse archived %d times", writes)
	}
}
