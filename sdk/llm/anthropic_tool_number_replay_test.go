package llm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func runAnthropicReplay(ctx context.Context, model llm.StreamingChatModel, request llm.InvokeRequest, stream bool) (string, error) {
	if !stream {
		completion, err := model.Invoke(ctx, request)
		if err != nil {
			return "", err
		}
		return completion.PlainText(), nil
	}
	events, err := model.InvokeStream(ctx, request)
	if err != nil {
		return "", err
	}
	var text strings.Builder
	var done bool
	for event := range events {
		switch e := event.(type) {
		case llm.StreamTextDeltaEvent:
			text.WriteString(e.Delta)
		case llm.StreamErrorEvent:
			err = e.AsError()
		case llm.StreamDoneEvent:
			done = true
		}
	}
	if err == nil && !done {
		err = fmt.Errorf("stream ended without a done event")
	}
	return text.String(), err
}
func TestAuditAnthropicToolNumberReplay(t *testing.T) {
	for _, id := range []string{"9007199254740993", "-9007199254740993", "1e400"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
				var sends atomic.Int32
				observed := make(chan string, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if sends.Add(1) == 1 {
						w.Header().Set("Content-Type", "application/json")
						io.WriteString(w, `{"type":"message","id":"msg_1","role":"assistant","stop_reason":"tool_use","content":[{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"id":`+id+`}}],"usage":{"input_tokens":4,"output_tokens":3}}`)
						return
					}
					var body struct {
						Messages []struct {
							Role    string
							Content []struct {
								Type  string
								Input struct{ ID json.Number }
							}
						}
					}
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						http.Error(w, err.Error(), 400)
						return
					}
					var replay string
					for _, m := range body.Messages {
						if m.Role == "assistant" {
							for _, b := range m.Content {
								if b.Type == "tool_use" {
									replay = b.Input.ID.String()
								}
							}
						}
					}
					observed <- replay
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						io.WriteString(w, "data: {\"type\":\"message_stop\"}\n\n")
					} else {
						w.Header().Set("Content-Type", "application/json")
						io.WriteString(w, `{"type":"message","role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}]}`)
					}
				}))
				defer server.Close()
				client := &anthropic.Client{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
				req := llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("Look up this record")}, Tools: []llm.ToolDefinition{{Name: "lookup", Parameters: map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"type": "integer"}}}}}}
				first, err := client.Invoke(t.Context(), req)
				if err != nil || len(first.ToolCalls) != 1 {
					t.Fatalf("first tool call: %v, %v", first, err)
				}
				source := first.ToolCalls[0]
				if source.Function.Arguments != `{"id":`+id+`}` {
					t.Fatalf("source arguments %q", source.Function.Arguments)
				}
				req.Messages = append(req.Messages, llm.NewAssistantMessage("", first.ToolCalls), llm.NewToolMessage(source.ID, "lookup", llm.TextContent("record found"), false))
				if _, err := runAnthropicReplay(t.Context(), client, req, stream); err != nil {
					t.Fatal(err)
				}
				if got := <-observed; got != id {
					t.Fatalf("provider tool input=%s; follow-up tool_use.input.id=%s, want %s", source.Function.Arguments, got, id)
				}
			})
		}
	}
}
