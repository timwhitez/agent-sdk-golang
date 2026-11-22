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
	"testing"
)

func TestAuditChatChoiceIsolation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			N      int
			Stream bool
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil || request.N != 2 {
			http.Error(w, "expected n=2", 400)
			return
		}
		if !request.Stream {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"chatcmpl_colors","choices":[{"index":0,"message":{"role":"assistant","content":"red"},"finish_reason":"stop"},{"index":1,"message":{"role":"assistant","content":"blue"},"finish_reason":"stop"}]}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: "+`{"id":"chatcmpl_colors","choices":[{"index":0,"delta":{"role":"assistant","content":"r"},"finish_reason":null},{"index":1,"delta":{"role":"assistant","content":"b"},"finish_reason":null}]}`+"\n\n")
		io.WriteString(w, "data: "+`{"id":"chatcmpl_colors","choices":[{"index":0,"delta":{"content":"ed"},"finish_reason":"stop"},{"index":1,"delta":{"content":"lue"},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	c := &openai.ChatClient{ModelName: "fixture", BaseURL: server.URL, Extra: map[string]any{"n": 2}, MaxRetries: 1}
	req := llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("Name a color")}}
	buffered, err := runChatChoice(t.Context(), c, req, false)
	if err != nil || buffered != "red" {
		t.Fatalf("buffered control=%q, %v", buffered, err)
	}
	streamed, err := runChatChoice(t.Context(), c, req, true)
	if err != nil {
		t.Fatal(err)
	}
	if streamed != buffered {
		t.Fatalf("answers from the server=%q, %q; buffered=%q, streamed=%q", "red", "blue", buffered, streamed)
	}
}
func runChatChoice(ctx context.Context, model llm.StreamingChatModel, request llm.InvokeRequest, stream bool) (string, error) {
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
