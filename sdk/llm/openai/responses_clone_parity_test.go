package openai

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestCloneInvokeRequestPreservesResponsesPayload(t *testing.T) {
	request := llm.InvokeRequest{
		Messages: []llm.Message{llm.NewUserMessage("hello")},
		Tools: []llm.ToolDefinition{{
			Name: "bounded",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"value": map[string]any{"type": "integer", "maximum": uint64(9007199254740993)},
				},
				"required": []string{"value"},
			},
		}},
	}
	cloned, err := llm.CloneInvokeRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	client := &ResponsesClient{ModelName: "test-model"}
	before, err := client.buildRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	after, err := client.buildRequest(cloned)
	if err != nil {
		t.Fatal(err)
	}
	beforeJSON, err := json.Marshal(before)
	if err != nil {
		t.Fatal(err)
	}
	afterJSON, err := json.Marshal(after)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(beforeJSON, afterJSON) {
		t.Fatalf("Responses payload changed after clone:\nbefore=%s\nafter=%s", beforeJSON, afterJSON)
	}
}
