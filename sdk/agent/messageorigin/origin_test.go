package messageorigin

import (
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestNewInternalUserMessageCarriesStableName(t *testing.T) {
	message := NewInternalUserMessage(KindRequireDone, "reminder")
	if message.Role != llm.RoleUser || message.Name != "sdk_internal_require_done" || message.Content.PlainText() != "reminder" {
		t.Fatalf("internal message = %#v", message)
	}
	if !IsInternalMessage(message) || IsRealUserMessage(message) {
		t.Fatalf("internal message classification = internal:%t real:%t", IsInternalMessage(message), IsRealUserMessage(message))
	}
}

func TestIsRealUserMessagePreservesUnknownNamedUsers(t *testing.T) {
	message := llm.Message{Role: llm.RoleUser, Name: "customer_alias", Content: llm.TextContent(RequireDoneReminderText + " quoted")}
	if !IsRealUserMessage(message) {
		t.Fatalf("unknown named user was classified as internal: %#v", message)
	}
}

func TestIsRealUserMessageRejectsDestroyedUser(t *testing.T) {
	message := llm.NewUserMessage("released")
	message.Destroyed = true
	if IsRealUserMessage(message) {
		t.Fatalf("destroyed user was classified as real: %#v", message)
	}
}
