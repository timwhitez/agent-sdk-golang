package openai

import (
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestResponsesManagedToolOutputValidation(t *testing.T) {
	call := llm.NewAssistantMessage("", []llm.ToolCall{historyToolCall("local", "work")})
	for _, tc := range []struct {
		name      string
		messages  []llm.Message
		managed   bool
		wantError bool
	}{
		{"server_output", []llm.Message{historyToolResult("external")}, true, false},
		{"stateless_orphan", []llm.Message{historyToolResult("external")}, false, true},
		{"empty_id", []llm.Message{historyToolResult("")}, true, true},
		{"duplicate_external", []llm.Message{historyToolResult("external"), historyToolResult("external")}, true, true},
		{"local_missing_result", []llm.Message{call, historyToolResult("external")}, true, true},
		{"local_result_before_call", []llm.Message{historyToolResult("local"), call}, true, true},
		{"mixed_valid", []llm.Message{historyToolResult("external"), call, historyToolResult("local")}, true, false},
		{"duplicate_local", []llm.Message{call, historyToolResult("local"), historyToolResult("local")}, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateResponsesToolHistory(tc.messages, tc.managed); (err != nil) != tc.wantError {
				t.Fatalf("err=%v wantError=%v", err, tc.wantError)
			}
		})
	}
}

func TestResponsesManagedOutputsRequireItemMode(t *testing.T) {
	client := &ResponsesClient{ModelName: "fixture", Extra: map[string]any{"previous_response_id": "resp_previous"}}
	legacy := false
	_, err := client.buildRequest(llm.InvokeRequest{Messages: []llm.Message{historyToolResult("external")}, Responses: &llm.ResponsesOptions{UseResponseItems: &legacy}})
	if err == nil {
		t.Fatal("legacy input silently converted a server-managed function output into user text")
	}
}
