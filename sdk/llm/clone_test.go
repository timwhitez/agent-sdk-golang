package llm

import "testing"

func TestCloneMessagesDeepCopiesMutableNestedValues(t *testing.T) {
	original := []Message{
		{
			Role: RoleUser,
			Content: Content{Blocks: []ContentBlock{
				{
					Type:     "image_url",
					Text:     "original text",
					ImageURL: &ImageURL{URL: "https://example.test/original.png", Detail: "high"},
					Source:   &DocSrc{Data: "ZG9j", MediaType: "application/pdf"},
				},
			}},
		},
		{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{
				{
					ID:         "call-1",
					Type:       "function",
					Function:   FunctionCall{Name: "read", Arguments: `{"path":"a"}`},
					ThoughtSig: []byte{1, 2, 3},
				},
			},
		},
	}

	cloned := CloneMessages(original)
	cloned[0].Content.Blocks[0].Text = "clone text"
	cloned[0].Content.Blocks[0].ImageURL.URL = "https://example.test/clone.png"
	cloned[0].Content.Blocks[0].Source.Data = "Y2xvbmU="
	cloned[1].ToolCalls[0].ThoughtSig[0] = 9

	if got := original[0].Content.Blocks[0].Text; got != "original text" {
		t.Fatalf("original text mutated through clone: %q", got)
	}
	if got := original[0].Content.Blocks[0].ImageURL.URL; got != "https://example.test/original.png" {
		t.Fatalf("original image URL mutated through clone: %q", got)
	}
	if got := original[0].Content.Blocks[0].Source.Data; got != "ZG9j" {
		t.Fatalf("original document source mutated through clone: %q", got)
	}
	if got := original[1].ToolCalls[0].ThoughtSig[0]; got != 1 {
		t.Fatalf("original thought signature mutated through clone: %d", got)
	}

	original[0].Content.Blocks[0].ImageURL.Detail = "low"
	original[1].ToolCalls[0].ThoughtSig[1] = 8
	if got := cloned[0].Content.Blocks[0].ImageURL.Detail; got != "high" {
		t.Fatalf("clone image detail mutated through original: %q", got)
	}
	if got := cloned[1].ToolCalls[0].ThoughtSig[1]; got != 2 {
		t.Fatalf("clone thought signature mutated through original: %d", got)
	}
}

func TestCloneMessagesPreservesNilAndEmptySlices(t *testing.T) {
	if CloneMessages(nil) != nil {
		t.Fatal("nil message slice must remain nil")
	}
	empty := CloneMessages([]Message{})
	if empty == nil || len(empty) != 0 {
		t.Fatalf("non-nil empty slice changed: %#v", empty)
	}
}
