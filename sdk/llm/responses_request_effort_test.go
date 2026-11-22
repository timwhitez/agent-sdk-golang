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
	"sync"
	"testing"
)

func runRequestEffort(ctx context.Context, model llm.StreamingChatModel, request llm.InvokeRequest, stream bool) (string, error) {
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
func replyRequestEffortResponses(w http.ResponseWriter, stream bool) {
	const answer = `{"id":"resp_next","status":"completed","output":[{"id":"msg_next","type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":"+answer+"}\n\n")
	} else {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, answer)
	}
}
func TestAuditResponsesRequestEffortDowngrade(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, configured := range []string{"", "low", "extra"} {
			t.Run(fmt.Sprintf("stream=%t/configured=%q", stream, configured), func(t *testing.T) {
				var mu sync.Mutex
				var sent []string
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body struct {
						Reasoning struct {
							Effort  string
							Summary string
						}
					}
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						http.Error(w, err.Error(), 400)
						return
					}
					mu.Lock()
					sent = append(sent, body.Reasoning.Effort)
					mu.Unlock()
					if body.Reasoning.Effort != "" {
						w.WriteHeader(400)
						io.WriteString(w, `{"error":{"message":"Unsupported parameter: reasoning.effort","param":"reasoning.effort"}}`)
						return
					}
					if body.Reasoning.Summary != "auto" {
						http.Error(w, "unrelated summary option lost", 400)
						return
					}
					replyRequestEffortResponses(w, stream)
				}))
				defer server.Close()
				options := &llm.ResponsesOptions{Reasoning: &llm.ResponsesReasoning{Effort: "high", Summary: "auto"}}
				req := llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("Estimate")}, Responses: options}
				c := &openai.ResponsesClient{ModelName: "fixture", BaseURL: server.URL, ReasoningEffort: configured, MaxRetries: 1, Warningf: func(string, ...any) {}}
				if configured == "extra" {
					c.ReasoningEffort = "low"
					c.Extra = map[string]any{"reasoning": map[string]any{"effort": "medium", "summary": "auto"}}
				}
				text, err := runRequestEffort(t.Context(), c, req, stream)
				if configured == "extra" && c.Extra["reasoning"].(map[string]any)["effort"] != "medium" {
					t.Fatal("call mutated client Extra reasoning")
				}
				mu.Lock()
				defer mu.Unlock()
				if text != "ok" || err != nil || len(sent) != 2 || sent[1] != "" || options.Reasoning.Effort != "high" {
					t.Fatalf("sent efforts=%q; result=%q, err=%v; caller effort=%q", sent, text, err, options.Reasoning.Effort)
				}
			})
		}
	}
}
