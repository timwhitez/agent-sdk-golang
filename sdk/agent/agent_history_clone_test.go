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

func cloneHistoryFixture() []llm.Message {
	return []llm.Message{
		{
			Role: llm.RoleUser,
			Content: llm.Content{Blocks: []llm.ContentBlock{
				{
					Type:     "image_url",
					Text:     "seed",
					ImageURL: &llm.ImageURL{URL: "https://example.test/seed.png"},
					Source:   &llm.DocSrc{Data: "c2VlZA==", MediaType: "application/pdf"},
				},
			}},
		},
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{
				{ID: "call-1", Type: "function", ThoughtSig: []byte{1, 2, 3}},
			},
		},
	}
}

func assertHistoryFixtureUnchanged(t *testing.T, got []llm.Message) {
	t.Helper()
	if text := got[0].Content.Blocks[0].Text; text != "seed" {
		t.Fatalf("history text = %q, want seed", text)
	}
	if url := got[0].Content.Blocks[0].ImageURL.URL; url != "https://example.test/seed.png" {
		t.Fatalf("history image URL = %q", url)
	}
	if data := got[0].Content.Blocks[0].Source.Data; data != "c2VlZA==" {
		t.Fatalf("history document data = %q", data)
	}
	if sig := got[1].ToolCalls[0].ThoughtSig[0]; sig != 1 {
		t.Fatalf("history thought signature[0] = %d, want 1", sig)
	}
}

func mutateHistoryFixture(messages []llm.Message) {
	messages[0].Content.Blocks[0].Text = "mutated"
	messages[0].Content.Blocks[0].ImageURL.URL = "https://example.test/mutated.png"
	messages[0].Content.Blocks[0].Source.Data = "bXV0YXRlZA=="
	messages[1].ToolCalls[0].ThoughtSig[0] = 9
}

func TestAgentInitialMessagesOwnTheirNestedState(t *testing.T) {
	seed := cloneHistoryFixture()
	ag, err := New(Config{LLM: historyCloneModel{}, InitialMessages: seed})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	mutateHistoryFixture(seed)
	assertHistoryFixtureUnchanged(t, ag.Messages())
}

func TestAgentMessagesReturnsIndependentNestedState(t *testing.T) {
	ag, err := New(Config{LLM: historyCloneModel{}, InitialMessages: cloneHistoryFixture()})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	snapshot := ag.Messages()
	mutateHistoryFixture(snapshot)
	assertHistoryFixtureUnchanged(t, ag.Messages())
}

func TestAgentReplaceHistoryOwnsNestedState(t *testing.T) {
	ag, err := New(Config{LLM: historyCloneModel{}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	replacement := cloneHistoryFixture()
	ag.ReplaceHistory(replacement)
	mutateHistoryFixture(replacement)
	assertHistoryFixtureUnchanged(t, ag.Messages())
}
