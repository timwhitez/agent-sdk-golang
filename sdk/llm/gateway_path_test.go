package llm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAuditResponsesGatewayPathKeepsImage(t *testing.T) {
	for _, path := range []string{"/api/photos", "/api/vision", "/api/v2beta"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", path, stream), func(t *testing.T) {
				payloads := make(chan string, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path != path+"/v1/responses" {
						http.Error(w, "wrong endpoint", 404)
						return
					}
					var req struct {
						Input []struct{ Content json.RawMessage }
					}
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Input) != 1 {
						http.Error(w, "bad JSON", 400)
						return
					}
					payloads <- string(req.Input[0].Content)
					replyGatewayPathResponses(w, stream)
				}))
				defer server.Close()
				c := &openai.ResponsesClient{ModelName: "fixture", BaseURL: server.URL + path, MaxRetries: 1}
				const imageURL = "https://images.example.org/poster.png"
				req := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "image_url", ImageURL: &llm.ImageURL{URL: imageURL}}}}}}}
				if _, err := runGatewayPath(t.Context(), c, req, stream); err != nil {
					t.Fatal(err)
				}
				if content := <-payloads; !strings.Contains(content, imageURL) || !strings.Contains(content, "input_image") {
					t.Fatalf("HTTP path=%s/v1/responses; content=%s, want input_image and supplied URL", path, content)
				}
			})
		}
	}
}
func runGatewayPath(ctx context.Context, model llm.StreamingChatModel, request llm.InvokeRequest, stream bool) (string, error) {
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
func replyGatewayPathResponses(w http.ResponseWriter, stream bool) {
	const answer = `{"id":"resp_next","status":"completed","output":[{"id":"msg_next","type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":"+answer+"}\n\n")
	} else {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, answer)
	}
}
func TestAuditAnthropicGatewayVersion(t *testing.T) {
	for _, tc := range []struct{ base, endpoint string }{{"/api/photos", "/api/photos/v1/messages"}, {"/api/v3", "/api/v3/messages"}, {"/api/vision", "/api/vision/v1/messages"}, {"/api/v2beta", "/api/v2beta/v1/messages"}} {
		t.Run(tc.base, func(t *testing.T) {
			seen := make(chan string, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				seen <- r.URL.Path
				if r.URL.Path != tc.endpoint {
					http.NotFound(w, r)
					return
				}
				fmt.Fprint(w, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":1}}`)
			}))
			defer server.Close()
			c := &anthropic.Client{BaseURL: server.URL + tc.base, ModelName: "fixture", MaxRetries: 1}
			_, err := c.Invoke(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("test")}})
			if got := <-seen; got != tc.endpoint || err != nil {
				t.Fatalf("HTTP path=%q, want %q, err=%v", got, tc.endpoint, err)
			}
		})
	}
}
