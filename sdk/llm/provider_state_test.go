package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestProviderStateIsOpaqueClonedAndTokenAccounted(t *testing.T) {
	const sentinel = "encrypted-provider-state-sentinel"
	content, err := WithProviderState(
		TextContent("visible answer"),
		[]ProviderState{{
			Provider: "test-provider",
			Kind:     "opaque.v1",
			Data:     json.RawMessage(`{"encrypted":"` + sentinel + `"}`),
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	message := Message{Role: RoleAssistant, Content: content}
	if strings.Contains(message.PlainText(), sentinel) || message.PlainText() != "visible answer" {
		t.Fatalf("opaque provider state leaked into plain text: %q", message.PlainText())
	}
	clone := CloneMessage(message)
	clone.Content.Blocks[len(clone.Content.Blocks)-1].Data = "changed"
	if message.Content.Blocks[len(message.Content.Blocks)-1].Data == clone.Content.Blocks[len(clone.Content.Blocks)-1].Data {
		t.Fatal("CloneMessage aliased provider-state content blocks")
	}
	withoutState := message
	withoutState.Content = WithoutProviderState(withoutState.Content)
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
	state, err := ProviderStateFromContent(decoded.Content)
	if err != nil {
		t.Fatal(err)
	}
	if len(state) != 1 || !strings.Contains(string(state[0].Data), sentinel) {
		t.Fatalf("provider state did not survive history serialization: %#v", state)
	}
}
