package anthropic

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestAnthropicSerializationIgnoresProviderStateContentBlock(t *testing.T) {
	baseline := llm.Message{Role: llm.RoleAssistant, Content: llm.TextContent("visible")}
	withState := baseline
	var err error
	withState.Content, err = llm.WithProviderState(withState.Content, []llm.ProviderState{{
		Provider: "openai-responses",
		Kind:     "response.output_item.v1",
		Data:     json.RawMessage(`{"encrypted_content":"must-not-leak"}`),
	}})
	if err != nil {
		t.Fatal(err)
	}
	_, got, err := serializeMessages([]llm.Message{withState})
	if err != nil {
		t.Fatal(err)
	}
	_, want, err := serializeMessages([]llm.Message{baseline})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Anthropic message changed by provider state: got=%#v want=%#v", got, want)
	}
}
