from pathlib import Path

agent = Path("sdk/agent/agent.go")
text = agent.read_text()
replacements = {
    '''\tif len(cfg.InitialMessages) > 0 {
\t\tag.messages = append([]llm.Message(nil), cfg.InitialMessages...)
\t}
''': '''\tif len(cfg.InitialMessages) > 0 {
\t\tag.messages = llm.CloneMessages(cfg.InitialMessages)
\t}
''',
    '''func (a *Agent) Messages() []llm.Message {
\ta.mu.Lock()
\tdefer a.mu.Unlock()
\tcpy := make([]llm.Message, len(a.messages))
\tcopy(cpy, a.messages)
\treturn cpy
}
''': '''func (a *Agent) Messages() []llm.Message {
\ta.mu.Lock()
\tdefer a.mu.Unlock()
\treturn llm.CloneMessages(a.messages)
}
''',
    '''func (a *Agent) ReplaceHistory(messages []llm.Message) {
\ta.mu.Lock()
\ta.messages = append([]llm.Message(nil), messages...)
\ta.resetEphemeralTrackingLocked()
''': '''func (a *Agent) ReplaceHistory(messages []llm.Message) {
\ta.mu.Lock()
\ta.messages = llm.CloneMessages(messages)
\ta.resetEphemeralTrackingLocked()
''',
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"history replacement count={count} for {old[:40]!r}")
    text = text.replace(old, new)
agent.write_text(text)

Path("sdk/llm/clone.go").write_text(r'''package llm

// CloneMessages returns a deep copy of a conversation history. The returned
// graph owns every mutable slice and pointee carried by Message, so callers may
// modify it without changing the source history.
func CloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	out := make([]Message, len(messages))
	for i := range messages {
		out[i] = CloneMessage(messages[i])
	}
	return out
}

// CloneMessage returns a deep copy of one provider-neutral message.
func CloneMessage(message Message) Message {
	out := message
	out.Content = CloneContent(message.Content)
	if message.ToolCalls != nil {
		out.ToolCalls = make([]ToolCall, len(message.ToolCalls))
		for i := range message.ToolCalls {
			out.ToolCalls[i] = CloneToolCall(message.ToolCalls[i])
		}
	}
	return out
}

// CloneContent returns a deep copy of message content and its pointer-bearing
// blocks.
func CloneContent(content Content) Content {
	out := content
	if content.Blocks == nil {
		return out
	}
	out.Blocks = make([]ContentBlock, len(content.Blocks))
	for i, block := range content.Blocks {
		out.Blocks[i] = block
		if block.ImageURL != nil {
			image := *block.ImageURL
			out.Blocks[i].ImageURL = &image
		}
		if block.Source != nil {
			source := *block.Source
			out.Blocks[i].Source = &source
		}
	}
	return out
}

// CloneToolCall returns a deep copy of one tool call, including provider-owned
// opaque byte signatures.
func CloneToolCall(call ToolCall) ToolCall {
	out := call
	if call.ThoughtSig != nil {
		out.ThoughtSig = append([]byte(nil), call.ThoughtSig...)
	}
	return out
}
''')

Path("sdk/agent/agent_history_clone_test.go").write_text(r'''package agent

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
''')
