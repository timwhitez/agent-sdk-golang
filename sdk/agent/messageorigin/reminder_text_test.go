package messageorigin

import (
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestDoneRemindersAskForTheFinalAnswerAndKeepLegacyTexts(t *testing.T) {
	for _, text := range []string{RequireDoneReminderText, EarlyStopReminderText} {
		if !strings.Contains(text, "final answer for the user") || strings.Contains(text, "concise completion message") {
			t.Fatalf("reminder %q should ask for the final answer, not a completion message", text)
		}
		if !IsInternalMessage(llm.Message{Role: llm.RoleUser, Content: llm.TextContent(text)}) {
			t.Fatalf("current reminder %q is not recognized as internal", text)
		}
	}
	for _, text := range []string{legacyRequireDoneReminderText, legacyRequireDoneReminderTextV2, legacyEarlyStopReminderText} {
		if !IsInternalMessage(llm.Message{Role: llm.RoleUser, Content: llm.TextContent(text)}) {
			t.Fatalf("persisted legacy reminder %q is no longer recognized as internal", text)
		}
	}
}
