package compaction

import (
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestNewCompactionCheckpointOwnsDeepMessageSnapshot(t *testing.T) {
	messages := []llm.Message{
		{
			Role: llm.RoleAssistant,
			Content: llm.Content{Blocks: []llm.ContentBlock{
				{
					Type:     "image_url",
					Text:     "original",
					ImageURL: &llm.ImageURL{URL: "https://example.test/image.png"},
					Source:   &llm.DocSrc{Data: "document", MediaType: "text/plain"},
				},
			}},
			ToolCalls: []llm.ToolCall{
				{
					ID:         "call_1",
					Type:       "function",
					ThoughtSig: []byte{1, 2, 3},
					Function:   llm.FunctionCall{Name: "read", Arguments: `{}`},
				},
			},
		},
	}
	checkpoint, err := NewCompactionCheckpoint(messages, Result{Compacted: true})
	if err != nil {
		t.Fatal(err)
	}
	originalID := checkpoint.CheckpointID

	messages[0].Content.Blocks[0].Text = "mutated"
	messages[0].Content.Blocks[0].ImageURL.URL = "https://mutated.test/image.png"
	messages[0].Content.Blocks[0].Source.Data = "mutated document"
	messages[0].ToolCalls[0].ThoughtSig[0] = 9
	messages[0].ToolCalls[0].Function.Name = "write"

	got := checkpoint.Messages[0]
	if got.Content.Blocks[0].Text != "original" {
		t.Fatalf("checkpoint block text = %q", got.Content.Blocks[0].Text)
	}
	if got.Content.Blocks[0].ImageURL.URL != "https://example.test/image.png" {
		t.Fatalf("checkpoint image URL = %q", got.Content.Blocks[0].ImageURL.URL)
	}
	if got.Content.Blocks[0].Source.Data != "document" {
		t.Fatalf("checkpoint document data = %q", got.Content.Blocks[0].Source.Data)
	}
	if got.ToolCalls[0].ThoughtSig[0] != 1 {
		t.Fatalf("checkpoint thought signature = %v", got.ToolCalls[0].ThoughtSig)
	}
	if got.ToolCalls[0].Function.Name != "read" {
		t.Fatalf("checkpoint tool name = %q", got.ToolCalls[0].Function.Name)
	}
	if checkpoint.CheckpointID != originalID {
		t.Fatalf("checkpoint id changed from %q to %q", originalID, checkpoint.CheckpointID)
	}
	if err := checkpoint.Validate(); err != nil {
		t.Fatalf("checkpoint no longer validates after source mutation: %v", err)
	}
}
