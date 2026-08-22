package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func newUnreadOpenAIStreamServer(t *testing.T, event string) (*httptest.Server, <-chan struct{}, <-chan struct{}) {
	t.Helper()
	streamWritten := make(chan struct{})
	requestClosed := make(chan struct{})
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(requestClosed)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("response writer does not implement Flusher")
			return
		}
		for i := 0; i < 1024; i++ {
			if _, err := fmt.Fprintf(w, "data: %s\n\n", event); err != nil {
				t.Errorf("write SSE event %d: %v", i, err)
				return
			}
			flusher.Flush()
		}
		close(streamWritten)
		<-r.Context().Done()
	})
	return httptest.NewServer(handler), streamWritten, requestClosed
}

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
			server, streamWritten, requestClosed := newUnreadOpenAIStreamServer(t, tc.event)
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
