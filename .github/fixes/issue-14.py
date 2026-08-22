from pathlib import Path

client = Path("sdk/llm/anthropic/client.go")
text = client.read_text()
old = '''func validateAnthropicToolHistory(messages []llm.Message) error {
\tfor i := 0; i < len(messages); i++ {
\t\tm := messages[i]
\t\tif m.Role == llm.RoleTool {
\t\t\treturn fmt.Errorf("anthropic: invalid tool history: tool message at index %d has no preceding assistant tool call", i)
\t\t}
\t\tif m.Role != llm.RoleAssistant || len(m.ToolCalls) == 0 {
\t\t\tcontinue
\t\t}
\t\texpected := make(map[string]bool, len(m.ToolCalls))
\t\tfor _, call := range m.ToolCalls {
\t\t\tid := normalizeToolCallIDWithWarning(call.ID, nil)
\t\t\tif id == "" {
\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: assistant tool call at index %d has empty id", i)
\t\t\t}
\t\t\texpected[id] = false
\t\t}
\t\tj := i + 1
\t\tfor j < len(messages) && messages[j].Role == llm.RoleTool {
\t\t\tid := normalizeToolCallIDWithWarning(messages[j].ToolCallID, nil)
\t\t\tif id == "" {
\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: tool message at index %d has empty tool_call_id", j)
\t\t\t}
\t\t\tif _, ok := expected[id]; !ok {
\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: tool message at index %d references unknown tool_use id %q", j, id)
\t\t\t}
\t\t\texpected[id] = true
\t\t\tj++
\t\t}
\t\tfor _, call := range m.ToolCalls {
\t\t\tid := normalizeToolCallIDWithWarning(call.ID, nil)
\t\t\tif !expected[id] {
\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: assistant tool call %q at index %d is missing a contiguous tool result", id, i)
\t\t\t}
\t\t}
\t\ti = j - 1
\t}
\treturn nil
}
'''
new = '''func validateAnthropicToolHistory(messages []llm.Message) error {
\tfor i := 0; i < len(messages); i++ {
\t\tm := messages[i]
\t\tif m.Role == llm.RoleTool {
\t\t\treturn fmt.Errorf("anthropic: invalid tool history: tool message at index %d has no preceding assistant tool call", i)
\t\t}
\t\tif m.Role != llm.RoleAssistant || len(m.ToolCalls) == 0 {
\t\t\tcontinue
\t\t}

\t\t// Preserve the source identity separately from Anthropic's wire-safe ID.
\t\t// The normalization is lossy (for example call/a and call:a both become
\t\t// call_a), so keying only by the normalized value can silently merge two
\t\t// distinct calls and let one tool_result satisfy both.
\t\texpected := make(map[string]bool, len(m.ToolCalls))
\t\twireOwner := make(map[string]string, len(m.ToolCalls))
\t\tfor _, call := range m.ToolCalls {
\t\t\toriginalID := strings.TrimSpace(call.ID)
\t\t\tif originalID == "" {
\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: assistant tool call at index %d has empty id", i)
\t\t\t}
\t\t\twireID := normalizeToolCallIDWithWarning(originalID, nil)
\t\t\tif previous, exists := wireOwner[wireID]; exists {
\t\t\t\tif previous == originalID {
\t\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: assistant tool call at index %d repeats id %q", i, originalID)
\t\t\t\t}
\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: assistant tool call ids %q and %q at index %d both normalize to %q", previous, originalID, i, wireID)
\t\t\t}
\t\twireOwner[wireID] = originalID
\t\texpected[originalID] = false
\t\t}

\t\tj := i + 1
\t\tfor j < len(messages) && messages[j].Role == llm.RoleTool {
\t\t\toriginalID := strings.TrimSpace(messages[j].ToolCallID)
\t\t\tif originalID == "" {
\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: tool message at index %d has empty tool_call_id", j)
\t\t\t}
\t\t\tseen, ok := expected[originalID]
\t\t\tif !ok {
\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: tool message at index %d references unknown tool_use id %q", j, originalID)
\t\t\t}
\t\t\tif seen {
\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: tool message at index %d repeats tool_use id %q", j, originalID)
\t\t\t}
\t\t\texpected[originalID] = true
\t\t\tj++
\t\t}
\t\tfor originalID, seen := range expected {
\t\t\tif !seen {
\t\t\t\treturn fmt.Errorf("anthropic: invalid tool history: assistant tool call %q at index %d is missing a contiguous tool result", originalID, i)
\t\t\t}
\t\t}
\t\ti = j - 1
\t}
\treturn nil
}
'''
if text.count(old) != 1:
    raise SystemExit(f"expected one validator block, found {text.count(old)}")
client.write_text(text.replace(old, new))

test = Path("sdk/llm/anthropic/tool_history_collision_test.go")
test.write_text(r'''package anthropic

import (
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func anthropicHistoryCall(id, name string) llm.ToolCall {
	return llm.ToolCall{
		ID:   id,
		Type: "function",
		Function: llm.FunctionCall{
			Name:      name,
			Arguments: `{}`,
		},
	}
}

func anthropicHistoryResult(id string) llm.Message {
	return llm.Message{
		Role:       llm.RoleTool,
		ToolCallID: id,
		Content:    llm.TextContent("ok"),
	}
}

func TestValidateAnthropicToolHistoryRejectsNormalizedIDCollision(t *testing.T) {
	messages := []llm.Message{
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{
				anthropicHistoryCall("call/a", "first"),
				anthropicHistoryCall("call:a", "second"),
			},
		},
		anthropicHistoryResult("call/a"),
		anthropicHistoryResult("call:a"),
	}

	err := validateAnthropicToolHistory(messages)
	if err == nil || !strings.Contains(err.Error(), "both normalize to") {
		t.Fatalf("validateAnthropicToolHistory() error = %v, want normalization collision", err)
	}
}

func TestValidateAnthropicToolHistoryRejectsDuplicateOriginalID(t *testing.T) {
	messages := []llm.Message{
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{
				anthropicHistoryCall("call_dup", "first"),
				anthropicHistoryCall("call_dup", "second"),
			},
		},
		anthropicHistoryResult("call_dup"),
	}

	err := validateAnthropicToolHistory(messages)
	if err == nil || !strings.Contains(err.Error(), "repeats id") {
		t.Fatalf("validateAnthropicToolHistory() error = %v, want duplicate call id", err)
	}
}

func TestValidateAnthropicToolHistoryRejectsDuplicateResult(t *testing.T) {
	messages := []llm.Message{
		{
			Role:      llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{anthropicHistoryCall("call_1", "first")},
		},
		anthropicHistoryResult("call_1"),
		anthropicHistoryResult("call_1"),
	}

	err := validateAnthropicToolHistory(messages)
	if err == nil || !strings.Contains(err.Error(), "repeats tool_use id") {
		t.Fatalf("validateAnthropicToolHistory() error = %v, want duplicate result", err)
	}
}

func TestSerializeAnthropicToolHistoryKeepsMatchingNormalizedWireID(t *testing.T) {
	messages := []llm.Message{
		{
			Role:      llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{anthropicHistoryCall("call/a", "first")},
		},
		anthropicHistoryResult("call/a"),
	}

	_, wire, err := serializeMessagesWithWarning(messages, nil)
	if err != nil {
		t.Fatalf("serializeMessagesWithWarning() error = %v", err)
	}
	if len(wire) != 2 || len(wire[0].Content) != 1 || len(wire[1].Content) != 1 {
		t.Fatalf("unexpected serialized history: %#v", wire)
	}
	if got := wire[0].Content[0].ID; got != "call_a" {
		t.Fatalf("tool_use id = %q, want call_a", got)
	}
	if got := wire[1].Content[0].ToolUseID; got != "call_a" {
		t.Fatalf("tool_result id = %q, want call_a", got)
	}
}
''')
