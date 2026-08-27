package openai

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestChatSerializationIgnoresProviderStateContentBlock(t *testing.T) {
	baseline := llm.TextContent("visible")
	withState, err := llm.WithProviderState(baseline, []llm.ProviderState{{
		Provider: "openai-responses",
		Kind:     "response.output_item.v1",
		Data:     json.RawMessage(`{"encrypted_content":"must-not-leak"}`),
	}})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := contentToOpenAI(withState), contentToOpenAI(baseline); !reflect.DeepEqual(got, want) {
		t.Fatalf("chat content changed by provider state: got=%#v want=%#v", got, want)
	}
	got, err := toChatMessage(llm.Message{Role: llm.RoleAssistant, Content: withState})
	if err != nil {
		t.Fatal(err)
	}
	want, err := toChatMessage(llm.Message{Role: llm.RoleAssistant, Content: baseline})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chat message changed by provider state: got=%#v want=%#v", got, want)
	}
}
