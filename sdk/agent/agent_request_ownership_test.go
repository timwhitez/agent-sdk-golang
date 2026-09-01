package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestNewOwnsNestedToolSchemas(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string"},
		},
	}
	agent, err := New(Config{
		LLM: historyCloneModel{},
		Tools: []tools.Tool{{
			Name:   "read",
			Schema: schema,
			Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				return llm.TextContent("ok"), nil
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	schema["properties"].(map[string]any)["path"].(map[string]any)["type"] = "number"

	got := agent.toolMap["read"].Schema["properties"].(map[string]any)["path"].(map[string]any)["type"]
	if got != "string" {
		t.Fatalf("agent tool schema changed through caller-owned map: %#v", got)
	}
}

type requestMutatingRetryModel struct {
	calls int
	err   error
}

func (m *requestMutatingRetryModel) Provider() string { return "stub" }
func (m *requestMutatingRetryModel) Model() string    { return "stub" }

func (m *requestMutatingRetryModel) Invoke(_ context.Context, request llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls == 1 {
		request.Messages[0].Content.Text = "mutated"
		request.Tools[0].Parameters["type"] = "array"
		request.Responses.OutputSchema["type"] = "array"
		return nil, &net.DNSError{Err: "i/o timeout", IsTimeout: true}
	}
	if got := request.Messages[0].Content.Text; got != "original" {
		m.err = fmt.Errorf("retry message = %q", got)
	}
	if got := request.Tools[0].Parameters["type"]; got != "object" {
		m.err = fmt.Errorf("retry tool schema = %#v", got)
	}
	if got := request.Responses.OutputSchema["type"]; got != "object" {
		m.err = fmt.Errorf("retry response schema = %#v", got)
	}
	return &llm.Completion{Content: llm.TextContent("ok"), StopReason: "stop"}, nil
}

func TestInvokeRetryUsesFreshRequestClone(t *testing.T) {
	model := &requestMutatingRetryModel{}
	agent, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("original")}},
		Tools:    []llm.ToolDefinition{{Name: "read", Parameters: map[string]any{"type": "object"}}},
		Responses: &llm.ResponsesOptions{
			OutputSchema: map[string]any{"type": "object"},
		},
	}
	if _, _, err := agent.invokeCompletionWithRetry(context.Background(), request, make(chan Event, 8)); err != nil {
		t.Fatal(err)
	}
	if model.calls != 2 {
		t.Fatalf("invoke calls = %d, want 2", model.calls)
	}
	if model.err != nil {
		t.Fatal(model.err)
	}
}
