package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestForeignThinkingBlocksNeverBecomeOpenAIInputText(t *testing.T) {
	for _, api := range []string{"chat", "responses"} {
		t.Run(api, func(t *testing.T) {
			var body string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				data, _ := io.ReadAll(r.Body)
				body = string(data)
				w.Header().Set("Content-Type", "application/json")
				if api == "chat" {
					_, _ = io.WriteString(w, `{"id":"fixture","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
				} else {
					_, _ = io.WriteString(w, `{"id":"fixture","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`)
				}
			}))
			defer server.Close()
			var client llm.ChatModel
			if api == "chat" {
				client = &ChatClient{BaseURL: server.URL, ModelName: "fixture", APIKey: "test", HTTPClient: server.Client()}
			} else {
				client = &ResponsesClient{BaseURL: server.URL, ModelName: "fixture", APIKey: "test", HTTPClient: server.Client()}
			}
			request := llm.InvokeRequest{Messages: []llm.Message{
				{Role: llm.RoleAssistant, Content: llm.Content{Blocks: []llm.ContentBlock{
					{Type: "text", Text: "visible history"},
					{Type: "thinking", Thinking: "HIDDEN_REASONING_CANARY"},
					{Type: "redacted_thinking", Text: "REDACTED_REASONING_CANARY", Data: "OPAQUE_REASONING_CANARY"},
					{Type: " THINKING ", Text: "NONCANONICAL_REASONING_CANARY"},
				}}},
				llm.NewUserMessage("continue"),
			}}
			if _, err := client.Invoke(context.Background(), request); err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(body, "visible history") || !strings.Contains(body, "continue") {
				t.Fatalf("visible history was lost: %s", body)
			}
			for _, secret := range []string{"HIDDEN_REASONING_CANARY", "REDACTED_REASONING_CANARY", "OPAQUE_REASONING_CANARY", "NONCANONICAL_REASONING_CANARY"} {
				if strings.Contains(body, secret) {
					t.Fatalf("provider-owned reasoning became %s input text", api)
				}
			}
		})
	}
}
