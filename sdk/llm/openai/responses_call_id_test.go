package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestResponsesFunctionCallRoundTripUsesCallID(t *testing.T) {
	t.Parallel()
	requestCount := 0
	var followup map[string]any
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		requestCount++
		if requestCount == 2 {
			if err := json.NewDecoder(r.Body).Decode(&followup); err != nil {
				t.Fatalf("decode follow-up request: %v", err)
			}
		}
		responseBody := `{"id":"resp_first","status":"completed","output":[{"id":"fc_roundtrip","call_id":"call_roundtrip","type":"function_call","name":"lookup","arguments":"{\"query\":\"go\"}"}]}`
		if requestCount > 1 {
			responseBody = `{"id":"resp_second","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}`
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(responseBody)),
			Request:    r,
		}, nil
	})}
	client := &ResponsesClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}

	first, err := client.Invoke(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("look it up")}},
	})
	if err != nil {
		t.Fatalf("first response: %v", err)
	}
	if len(first.ToolCalls) != 1 || first.ToolCalls[0].ID != "call_roundtrip" {
		t.Fatalf("first tool calls = %#v", first.ToolCalls)
	}

	_, err = client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: llm.TextContent("look it up")},
		{Role: llm.RoleAssistant, ToolCalls: first.ToolCalls},
		{Role: llm.RoleTool, ToolCallID: first.ToolCalls[0].ID, Content: llm.TextContent(`{"result":"ok"}`)},
	}})
	if err != nil {
		t.Fatalf("follow-up response: %v", err)
	}

	input, ok := followup["input"].([]any)
	if !ok {
		t.Fatalf("follow-up input = %#v", followup["input"])
	}
	foundOutput := false
	for _, raw := range input {
		item, _ := raw.(map[string]any)
		if item["type"] != "function_call_output" {
			continue
		}
		foundOutput = true
		if got := item["call_id"]; got != "call_roundtrip" {
			t.Fatalf("function_call_output.call_id = %#v, want call_roundtrip", got)
		}
	}
	if !foundOutput {
		t.Fatalf("follow-up omitted function_call_output: %#v", input)
	}
}
