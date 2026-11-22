package llm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func runServerManagedOutput(ctx context.Context, model llm.StreamingChatModel, request llm.InvokeRequest, stream bool) (string, error) {
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
func replyServerManagedOutput(w http.ResponseWriter, stream bool) {
	const answer = `{"id":"resp_next","status":"completed","output":[{"id":"msg_next","type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":"+answer+"}\n\n")
	} else {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, answer)
	}
}
func TestAuditResponsesServerManagedToolOutput(t *testing.T) {
	for _, mode := range []string{"previous_response_id", "conversation"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", mode, stream), func(t *testing.T) {
				var calls atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]json.RawMessage
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						http.Error(w, err.Error(), 400)
						return
					}
					switch calls.Add(1) {
					case 1:
						w.Header().Set("Content-Type", "application/json")
						io.WriteString(w, `{"id":"resp_lookup","status":"completed","output":[{"id":"fc_lookup","type":"function_call","call_id":"call_lookup","name":"lookup","arguments":"{}"}]}`)
					case 2:
						var in []struct {
							Type   string
							CallID string `json:"call_id"`
							Output string
						}
						if err := json.Unmarshal(body["input"], &in); err != nil || len(in) != 1 || in[0].Type != "function_call_output" || in[0].CallID != "call_lookup" || in[0].Output != "ready" {
							http.Error(w, "bad continuation input", 400)
							return
						}
						wantID := `"resp_lookup"`
						if mode == "conversation" {
							wantID = `"conv_example"`
						}
						if string(body[mode]) != wantID {
							http.Error(w, "missing server state selector", 400)
							return
						}
						replyServerManagedOutput(w, stream)
					default:
						http.Error(w, "unexpected request", 400)
					}
				}))
				defer server.Close()
				client := &openai.ResponsesClient{ModelName: "fixture", BaseURL: server.URL, MaxRetries: 1}
				options := &llm.ResponsesOptions{}
				if mode == "conversation" {
					options.ConversationID = "conv_example"
				}
				def := llm.ToolDefinition{Name: "lookup", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}
				first, err := client.Invoke(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("Check status")}, Tools: []llm.ToolDefinition{def}, Responses: options})
				if err != nil || first == nil || len(first.ToolCalls) != 1 {
					t.Fatalf("first call: %#v, %v", first, err)
				}
				continuation := *client
				if mode == "previous_response_id" {
					continuation.Extra = map[string]any{"previous_response_id": first.ResponseID}
				}
				result := llm.NewToolMessage(first.ToolCalls[0].ID, "lookup", llm.TextContent("ready"), false)
				text, err := runServerManagedOutput(t.Context(), &continuation, llm.InvokeRequest{Messages: []llm.Message{result}, Tools: []llm.ToolDefinition{def}, Responses: options}, stream)
				if err != nil || text != "ok" || calls.Load() != 2 {
					t.Fatalf("server returned call_id=%s; continuation HTTP sends=%d, answer=%q, err=%v", first.ToolCalls[0].ID, calls.Load()-1, text, err)
				}
			})
		}
	}
}
