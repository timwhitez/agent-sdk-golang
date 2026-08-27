package llm_test

import (
	"encoding/json"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestLegacyUnkeyedMessageAndCompletionLiteralsStillCompile(t *testing.T) {
	message := llm.Message{
		llm.RoleUser,
		llm.TextContent("hello"),
		"",
		false,
		nil,
		"",
		"",
		false,
		false,
		false,
	}
	completion := llm.Completion{
		llm.TextContent("answer"),
		"",
		nil,
		nil,
		"end_turn",
		"resp_1",
		nil,
		json.RawMessage(`{"ok":true}`),
	}
	options := llm.ResponsesOptions{
		nil,
		nil,
		"",
		"",
		"",
		nil,
		nil,
		nil,
		nil,
		nil,
		"",
		nil,
	}
	if message.PlainText() != "hello" || completion.PlainText() != "answer" {
		t.Fatalf("legacy literals changed behavior: message=%#v completion=%#v", message, completion)
	}
	_ = options
}
