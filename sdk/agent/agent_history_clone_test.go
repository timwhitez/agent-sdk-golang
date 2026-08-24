package agent

import (
	"context"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type historyCloneModel struct{}

func (historyCloneModel) Provider() string { return "stub" }
func (historyCloneModel) Model() string    { return "stub" }
func (historyCloneModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("ok")}, nil
}

func mutableHistoryMessage() llm.Message {
	return llm.Message{
		Role: llm.RoleAssistant,
		Content: llm.Content{
			Text: "top-level",
			Blocks: []llm.ContentBlock{
				{
					Type: "image_url",
					Text: "block",
					ImageURL: &llm.ImageURL{
						URL:       "https://example.test/image.png",
						Detail:    "high",
						MediaType: "image/png",
					},
					Source: &llm.DocSrc{Data: "document", MediaType: "text/plain"},
				},
			},
		},
		ToolCalls: []llm.ToolCall{
			{
				ID:         "call_1",
				Type:       "function",
				ThoughtSig: []byte{1, 2, 3},
				Function: llm.FunctionCall{
					Name:      "read",
					Arguments: `{"path":"README.md"}`,
				},
			},
		},
	}
}

func mutateHistoryMessage(message *llm.Message) {
	message.Content.Blocks[0].Text = "mutated block"
	message.Content.Blocks[0].ImageURL.URL = "https://mutated.test/image.png"
	message.Content.Blocks[0].Source.Data = "mutated document"
	message.ToolCalls[0].ThoughtSig[0] = 9
	message.ToolCalls[0].Function.Name = "write"
}

func assertOriginalHistoryMessage(t *testing.T, message llm.Message) {
	t.Helper()
	if got := message.Content.Blocks[0].Text; got != "block" {
		t.Fatalf("block text = %q, want block", got)
	}
	if got := message.Content.Blocks[0].ImageURL.URL; got != "https://example.test/image.png" {
		t.Fatalf("image URL = %q", got)
	}
	if got := message.Content.Blocks[0].Source.Data; got != "document" {
		t.Fatalf("document data = %q", got)
	}
	if got := message.ToolCalls[0].ThoughtSig[0]; got != 1 {
		t.Fatalf("thought signature[0] = %d, want 1", got)
	}
	if got := message.ToolCalls[0].Function.Name; got != "read" {
		t.Fatalf("tool name = %q, want read", got)
	}
}

func TestNewTakesOwnershipOfInitialMessages(t *testing.T) {
	initial := []llm.Message{mutableHistoryMessage()}
	agent, err := New(Config{LLM: historyCloneModel{}, InitialMessages: initial})
	if err != nil {
		t.Fatal(err)
	}
	mutateHistoryMessage(&initial[0])
	assertOriginalHistoryMessage(t, agent.Messages()[0])
}

func TestMessagesReturnsDeepSnapshot(t *testing.T) {
	agent, err := New(Config{LLM: historyCloneModel{}, InitialMessages: []llm.Message{mutableHistoryMessage()}})
	if err != nil {
		t.Fatal(err)
	}
	snapshot := agent.Messages()
	mutateHistoryMessage(&snapshot[0])
	assertOriginalHistoryMessage(t, agent.Messages()[0])
}

func TestReplaceHistoryTakesOwnership(t *testing.T) {
	agent, err := New(Config{LLM: historyCloneModel{}})
	if err != nil {
		t.Fatal(err)
	}
	replacement := []llm.Message{mutableHistoryMessage()}
	agent.ReplaceHistory(replacement)
	mutateHistoryMessage(&replacement[0])
	assertOriginalHistoryMessage(t, agent.Messages()[0])
}
