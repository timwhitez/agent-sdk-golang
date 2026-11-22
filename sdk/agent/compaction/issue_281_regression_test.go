package compaction

import (
	"bytes"
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"strings"
	"testing"
)

func TestD3MicrocompactPreservesUserAttachment(t *testing.T) {
	const session = "review-attachments"
	host := newCanonicalCompactionHost()
	store := &memoryLedgerStore{ledger: NewLedger(session)}
	cfg := canonicalCompactionConfig(session, store, host)
	cfg.EnableUserCodeMicrocompact = true
	code := "Trace this code against the attached diagram.\n" + strings.Repeat("`", 3) + "go\n" + numberedLines("code line ", 140) + strings.Repeat("`", 3)
	const diagram = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGOoBwAAgQCA05RTSgAAAABJRU5ErkJggg=="
	messages := []llm.Message{
		{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{
			{Type: "text", Text: code}, {Type: "image_url", ImageURL: &llm.ImageURL{URL: diagram}},
		}}},
		llm.NewAssistantMessage("I will refer back to the diagram.", nil),
		llm.NewUserMessage("Continue."),
	}
	out, res, err := NewService(cfg).CompactPipeline(context.Background(), nil, messages, PipelineRequest{TargetWatermark: "prune", EstimatedTokens: 3250})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Compacted || out[0].Content.Blocks[0].Text == code || messages[0].Content.Blocks[0].Text != code {
		t.Fatal("single text block was not independently compacted")
	}
	for _, block := range out[0].Content.Blocks {
		if block.ImageURL != nil && block.ImageURL.URL == diagram {
			return
		}
	}
	for _, obj := range host.objects {
		if bytes.Contains(obj.content, []byte(diagram)) {
			return
		}
	}
	t.Fatalf("user attachment absent from history and all %d durable objects after tiers=%v, remaining content blocks=%d", len(host.objects), res.TiersApplied, len(out[0].Content.Blocks))
}

func TestMicrocompactKeepsMultipleTextBlockBoundaries(t *testing.T) {
	const session = "review-multiple-blocks"
	host := newCanonicalCompactionHost()
	cfg := canonicalCompactionConfig(session, &memoryLedgerStore{ledger: NewLedger(session)}, host)
	cfg.EnableUserCodeMicrocompact = true
	code := "```go\n" + numberedLines("code ", 140) + "```"
	messages := []llm.Message{
		{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{
			{Type: "text", Text: code}, {Type: "image_url", ImageURL: &llm.ImageURL{URL: "image"}}, {Type: "text", Text: "The image above is the expected result."},
		}}},
		llm.NewUserMessage("Continue."),
	}
	out, res, err := NewService(cfg).CompactPipeline(context.Background(), nil, messages, PipelineRequest{TargetWatermark: "prune", EstimatedTokens: 3250})
	if err != nil || res.Compacted || len(out[0].Content.Blocks) != 3 || out[0].Content.Blocks[0].Text != code || host.puts != 0 {
		t.Fatalf("unrepresentable text boundaries were changed: %+v err=%v writes=%d", res, err, host.puts)
	}
}
