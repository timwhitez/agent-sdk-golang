package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// With adaptive thinking the provider returns thinking blocks whose text is
// empty but whose signature must be replayed. Anthropic rejects a replayed
// thinking block without its "thinking" field ("thinking: Field required").
func TestReplayedAssistantKeepsEmptyThinking(t *testing.T) {
	c := &Client{ModelName: "claude-opus-5-5"}
	assistant := llm.Message{
		Role: llm.RoleAssistant,
		Content: llm.Content{Blocks: []llm.ContentBlock{
			{Type: "thinking", Thinking: "", Signature: "sig-adaptive"},
		}},
		ToolCalls: []llm.ToolCall{{ID: "toolu_1", Type: "function", Function: llm.FunctionCall{Name: "read_file", Arguments: "{\"path\":\"README.md\"}"}}},
	}
	req := llm.InvokeRequest{Messages: []llm.Message{
		llm.NewUserMessage("read the readme"),
		assistant,
		{Role: llm.RoleTool, ToolCallID: "toolu_1", Content: llm.TextContent("# Demo")},
	}}
	payload, err := c.buildRequest(req, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	var assistantBlocks []map[string]any
	for _, m := range decoded.Messages {
		if m.Role == "assistant" {
			assistantBlocks = m.Content
		}
	}
	if len(assistantBlocks) != 2 {
		t.Fatalf("assistant blocks = %v, want thinking + tool_use", assistantBlocks)
	}
	thinking := assistantBlocks[0]
	if thinking["type"] != "thinking" || thinking["signature"] != "sig-adaptive" {
		t.Fatalf("thinking block = %v", thinking)
	}
	if text, ok := thinking["thinking"].(string); !ok || text != "" {
		t.Fatalf("thinking block must carry an empty thinking string, got %v in %s", thinking["thinking"], raw)
	}
	if assistantBlocks[1]["type"] != "tool_use" {
		t.Fatalf("second block = %v, want tool_use", assistantBlocks[1])
	}
}
