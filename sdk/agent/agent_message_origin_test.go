package agent

import (
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func assertHistoryContainsNamedUserMessage(t *testing.T, messages []llm.Message, text, name string) {
	t.Helper()
	for _, message := range messages {
		if message.Role == llm.RoleUser && message.Content.PlainText() == text {
			if message.Name != name {
				t.Fatalf("message %q name = %q, want %q", text, message.Name, name)
			}
			return
		}
	}
	t.Fatalf("history is missing user-role message %q: %#v", text, messages)
}
