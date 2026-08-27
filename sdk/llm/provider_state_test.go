package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestProviderStateIsOpaqueClonedAndTokenAccounted(t *testing.T) {
	const sentinel = "encrypted-provider-state-sentinel"
	message := Message{
		Role:    RoleAssistant,
		Content: TextContent("visible answer"),
		ProviderState: []ProviderState{{
			Provider: "test-provider",
			Kind:     "opaque.v1",
			Data:     json.RawMessage(`{"encrypted":"` + sentinel + `"}`),
		}},
	}
	if strings.Contains(message.PlainText(), sentinel) || message.PlainText() != "visible answer" {
		t.Fatalf("opaque provider state leaked into plain text: %q", message.PlainText())
	}
	clone := CloneMessage(message)
	clone.ProviderState[0].Data[2] = 'X'
	if string(message.ProviderState[0].Data) == string(clone.ProviderState[0].Data) {
		t.Fatal("CloneMessage aliased provider state bytes")
	}
	withoutState := message
	withoutState.ProviderState = nil
	if EstimateMessagesTokens([]Message{message}) <= EstimateMessagesTokens([]Message{withoutState}) {
		t.Fatal("opaque provider state was omitted from token estimation")
	}
	encoded, err := json.Marshal(message)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Message
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.ProviderState) != 1 || !strings.Contains(string(decoded.ProviderState[0].Data), sentinel) {
		t.Fatalf("provider state did not survive history serialization: %#v", decoded.ProviderState)
	}
}
