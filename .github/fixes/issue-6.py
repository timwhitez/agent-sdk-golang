from pathlib import Path


def patch_client(path: str, receiver: str) -> None:
    file = Path(path)
    text = file.read_text()
    old = f'''func (c *{receiver}) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {{
\tout := make(chan llm.StreamEvent, 128)
'''
    new = f'''func (c *{receiver}) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {{
\t// Keep provider production separate from caller delivery. If a caller stops
\t// consuming, cancellation switches the forwarding goroutine to drain-and-drop
\t// mode so the producer can finish and close the response body.
\tout := make(chan llm.StreamEvent, 128)
\tforwarded := make(chan llm.StreamEvent, 128)
\tgo forwardOpenAIStreamEvents(ctx, forwarded, out)
'''
    if text.count(old) != 1:
        raise SystemExit(f"{path}: stream start anchor count={text.count(old)}")
    text = text.replace(old, new)
    if text.count("\treturn out, nil\n}") != 1:
        raise SystemExit(f"{path}: stream return anchor count={text.count(chr(9) + 'return out, nil' + chr(10) + '}')}")
    text = text.replace("\treturn out, nil\n}", "\treturn forwarded, nil\n}")
    file.write_text(text)


patch_client("sdk/llm/openai/chat.go", "ChatClient")
patch_client("sdk/llm/openai/responses.go", "ResponsesClient")

Path("sdk/llm/openai/stream_forward.go").write_text(r'''package openai

import (
	"context"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// forwardOpenAIStreamEvents owns the public stream channel. Before
// cancellation it preserves backpressure. After cancellation it drains and
// drops internal events until the provider producer closes, guaranteeing that
// a producer blocked on its bounded queue can exit and release the HTTP body.
func forwardOpenAIStreamEvents(ctx context.Context, dst chan<- llm.StreamEvent, src <-chan llm.StreamEvent) {
	defer close(dst)
	dropping := false
	for event := range src {
		if dropping {
			continue
		}
		select {
		case dst <- event:
		case <-ctx.Done():
			dropping = true
		}
	}
}
''')

Path("sdk/llm/openai/stream_cancel_test.go").write_text(r'''package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestOpenAIStreamsCloseWhenUnreadConsumerCancels(t *testing.T) {
	tests := []struct {
		name   string
		event  string
		invoke func(context.Context, string) (<-chan llm.StreamEvent, error)
	}{
		{
			name:  "chat",
			event: `{"choices":[{"delta":{"content":"x"}}]}`,
			invoke: func(ctx context.Context, baseURL string) (<-chan llm.StreamEvent, error) {
				client := &ChatClient{BaseURL: baseURL, ModelName: "test-model", MaxRetries: 1}
				return client.InvokeStream(ctx, llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
			},
		},
		{
			name:  "responses",
			event: `{"type":"response.output_text.delta","delta":"x"}`,
			invoke: func(ctx context.Context, baseURL string) (<-chan llm.StreamEvent, error) {
				client := &ResponsesClient{BaseURL: baseURL, ModelName: "test-model", MaxRetries: 1}
				return client.InvokeStream(ctx, llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			streamWritten := make(chan struct{})
			requestClosed := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(requestClosed)
				w.Header().Set("Content-Type", "text/event-stream")
				flusher, ok := w.(http.Flusher)
				if !ok {
					t.Error("response writer does not implement Flusher")
					return
				}
				for i := 0; i < 512; i++ {
					if _, err := fmt.Fprintf(w, "data: %s\n\n", tc.event); err != nil {
						return
					flusher.Flush()
				}
				close(streamWritten)
				<-r.Context().Done()
			}))
			defer server.Close()

			ctx, cancel := context.WithCancel(context.Background())
			stream, err := tc.invoke(ctx, server.URL)
			if err != nil {
				cancel()
				t.Fatalf("InvokeStream() error = %v", err)
			}
			select {
			case <-streamWritten:
			case <-time.After(5 * time.Second):
				cancel()
				t.Fatal("server did not produce enough events to fill bounded channels")
			}
			// Deliberately consume nothing until after cancellation.
			cancel()

			closed := make(chan struct{})
			go func() {
				for range stream {
				}
				close(closed)
			}()
			select {
			case <-closed:
			case <-time.After(5 * time.Second):
				t.Fatal("public stream did not close after cancellation")
			}
			select {
			case <-requestClosed:
			case <-time.After(5 * time.Second):
				t.Fatal("provider response body/request remained open after cancellation")
			}
		})
	}
}
''')
