package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type issue425AnthropicTransport struct {
	body   string
	stream bool
}

func (r issue425AnthropicTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	ct := "application/json"
	if r.stream {
		ct = "text/event-stream"
	}
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {ct}}, Body: io.NopCloser(strings.NewReader(r.body)), Request: req}, nil
}

type issue425BufferedModel struct{ llm.ChatModel }

func TestIssue425AnthropicMalformedResponseAgentOutcome(t *testing.T) {
	for _, body := range []string{"null", "{}", `{"type":"error","error":{"type":"api_error","message":"synthetic failure"}}`} {
		t.Run(body, func(t *testing.T) {
			c := &Client{ModelName: "issue425-model", MaxRetries: 1, HTTPClient: &http.Client{Transport: issue425AnthropicTransport{body: body}}}
			a, err := agent.New(agent.Config{LLM: issue425BufferedModel{c}, MaxIterations: 1, InvokeRetryMaxAttempts: 1, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			success, failures := 0, 0
			for event := range a.QueryStream(context.Background(), llm.TextContent("hello")) {
				switch e := event.(type) {
				case agent.FinalResponseEvent:
					if e.Status == "complete" {
						success++
					}
				case agent.ErrorEvent:
					failures++
				}
			}
			if success != 0 || failures == 0 {
				t.Fatalf("invalid buffered payload caused success=%d error=%d; want success=0 and error>=1", success, failures)
			}
		})
	}
}
func TestIssue425AnthropicMalformedBufferedResponses(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		wantErr    bool
	}{
		{"null", "null", true},
		{"empty-object", "{}", true},
		{"scalar", `"invalid"`, true},
		{"array", `[]`, true},
		{"null-content", `{"type":"message","content":null}`, true},
		{"object-content", `{"type":"message","content":{}}`, true},
		{"string-content", `{"type":"message","content":"wrong"}`, true},
		{"wrong-role", `{"type":"message","role":"user","content":[]}`, true},
		{"wrong-type", `{"type":"completion","role":"assistant","content":[]}`, true},
		{"null-usage", `{"type":"message","content":[],"usage":null}`, false},
		{"empty-usage", `{"type":"message","content":[],"usage":{}}`, false},
		{"escaped-type", `{"type":"m\u0065ssage","role":"assistant","content":[]}`, false},
		{"error-envelope", `{"type":"error","error":{"type":"api_error","message":"synthetic upstream failure"}}`, true},
		{"valid-text", `{"id":"msg_issue425","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`, false},
		{"valid-empty-content", `{"id":"msg_issue425","type":"message","role":"assistant","content":[],"stop_reason":"end_turn"}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &Client{ModelName: "issue425-model", MaxRetries: 1, HTTPClient: &http.Client{Transport: issue425AnthropicTransport{body: tc.body}}}
			comp, err := c.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
			if tc.wantErr && err == nil {
				t.Fatalf("invalid payload accepted as success: %#v", comp)
			}
			if !tc.wantErr && err != nil {
				t.Fatal(err)
			}
		})
	}
}
func TestIssue426AnthropicInitialContentBlockText(t *testing.T) {
	for _, tc := range []struct {
		name           string
		starts, deltas []string
	}{
		{"start-and-delta", []string{"initial "}, []string{"delta"}},
		{"start-only", []string{"complete"}, []string{""}},
		{"empty-start", []string{""}, []string{"delta"}},
		{"whitespace-and-multiple-blocks", []string{" ", "next "}, []string{"text\n", "block"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var body strings.Builder
			write := func(event any) {
				data, err := json.Marshal(event)
				if err != nil {
					t.Fatal(err)
				}
				fmt.Fprintf(&body, "data: %s\n\n", data)
			}
			write(map[string]any{"type": "message_start", "message": map[string]any{"id": "msg_initial", "type": "message", "role": "assistant", "content": []any{}}})
			want := ""
			var content []map[string]string
			for i, initial := range tc.starts {
				write(map[string]any{"type": "content_block_start", "index": i, "content_block": map[string]any{"type": "text", "text": initial}})
				if tc.deltas[i] != "" {
					write(map[string]any{"type": "content_block_delta", "index": i, "delta": map[string]any{"type": "text_delta", "text": tc.deltas[i]}})
				}
				write(map[string]any{"type": "content_block_stop", "index": i})
				want += initial + tc.deltas[i]
				content = append(content, map[string]string{"type": "text", "text": initial + tc.deltas[i]})
			}
			write(map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "end_turn"}})
			write(map[string]any{"type": "message_stop"})
			client := &Client{ModelName: "issue426", MaxRetries: 1, HTTPClient: &http.Client{Transport: issue425AnthropicTransport{body: body.String(), stream: true}}}
			stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
			if err != nil {
				t.Fatal(err)
			}
			got, done := "", 0
			for event := range stream {
				switch e := event.(type) {
				case llm.StreamTextDeltaEvent:
					got += e.Delta
				case llm.StreamDoneEvent:
					done++
				case llm.StreamErrorEvent:
					t.Fatal(e.AsError())
				}
			}
			if got != want || done != 1 {
				t.Fatalf("text=%q want=%q done=%d", got, want, done)
			}
			buffered, _ := json.Marshal(map[string]any{"type": "message", "content": content, "stop_reason": "end_turn"})
			comp, err := parseResponse(buffered)
			if err != nil {
				t.Fatal(err)
			}
			var bufferedText string
			for _, block := range comp.Content.Blocks {
				bufferedText += block.Text
			}
			if bufferedText != got {
				t.Fatalf("buffered text=%q stream=%q", bufferedText, got)
			}
			a, err := agent.New(agent.Config{LLM: client, MaxIterations: 1, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			answer, err := a.Query(context.Background(), "hello")
			if err != nil {
				t.Fatal(err)
			}
			if answer != want {
				t.Fatalf("agent answer=%q want=%q", answer, want)
			}
		})
	}
}
