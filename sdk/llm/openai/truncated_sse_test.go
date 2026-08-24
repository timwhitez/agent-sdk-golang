package openai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func openAIStreamFixture(t *testing.T, events ...string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("response writer does not support flushing")
			return
		}
		for _, event := range events {
			if _, err := fmt.Fprintf(w, "data: %s\n\n", event); err != nil {
				return
			}
			flusher.Flush()
		}
	}))
}

func collectOpenAIStream(events <-chan llm.StreamEvent) []llm.StreamEvent {
	var collected []llm.StreamEvent
	for event := range events {
		collected = append(collected, event)
	}
	return collected
}

func assertIncompleteOpenAIStream(t *testing.T, events []llm.StreamEvent) {
	t.Helper()
	var providerErr *llm.ProviderError
	sawError := false
	for _, event := range events {
		switch typed := event.(type) {
		case llm.StreamDoneEvent:
			t.Fatalf("truncated stream emitted StreamDoneEvent: %#v", events)
		case llm.StreamErrorEvent:
			sawError = true
			if !errors.As(typed.AsError(), &providerErr) {
				t.Fatalf("stream error = %v, want ProviderError", typed.AsError())
			}
		}
	}
	if !sawError || providerErr == nil {
		t.Fatalf("truncated stream emitted no provider error: %#v", events)
	}
}

func TestChatStreamEOFBeforeDoneIsIncomplete(t *testing.T) {
	server := openAIStreamFixture(t, `{"id":"resp_1","choices":[{"delta":{"content":"partial"}}]}`)
	defer server.Close()
	client := &ChatClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	events := collectOpenAIStream(stream)
	if len(events) == 0 {
		t.Fatal("expected partial stream events")
	}
	assertIncompleteOpenAIStream(t, events)
}

func TestChatPartialToolArgumentsEOFIsIncomplete(t *testing.T) {
	server := openAIStreamFixture(t, `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":"{\\\"path\\\":"}}]}}]}`)
	defer server.Close()
	client := &ChatClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	events := collectOpenAIStream(stream)
	sawToolDelta := false
	for _, event := range events {
		if _, ok := event.(llm.StreamToolCallDeltaEvent); ok {
			sawToolDelta = true
		}
	}
	if !sawToolDelta {
		t.Fatalf("missing partial tool-call delta: %#v", events)
	}
	assertIncompleteOpenAIStream(t, events)
}

func TestResponsesStreamEOFBeforeCompletedIsIncomplete(t *testing.T) {
	server := openAIStreamFixture(t, `{"type":"response.output_text.delta","delta":"partial"}`)
	defer server.Close()
	client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	assertIncompleteOpenAIStream(t, collectOpenAIStream(stream))
}

func TestExplicitOpenAITerminalsStillComplete(t *testing.T) {
	t.Run("chat done marker", func(t *testing.T) {
		server := openAIStreamFixture(t,
			`{"choices":[{"delta":{"content":"complete"},"finish_reason":"stop"}]}`,
			`[DONE]`,
		)
		defer server.Close()
		client := &ChatClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
		stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
		if err != nil {
			t.Fatal(err)
		}
		events := collectOpenAIStream(stream)
		if _, ok := events[len(events)-1].(llm.StreamDoneEvent); !ok {
			t.Fatalf("terminal events = %#v", events)
		}
	})
	t.Run("responses completed", func(t *testing.T) {
		server := openAIStreamFixture(t, `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}`)
		defer server.Close()
		client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
		stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
		if err != nil {
			t.Fatal(err)
		}
		events := collectOpenAIStream(stream)
		if _, ok := events[len(events)-1].(llm.StreamDoneEvent); !ok {
			t.Fatalf("terminal events = %#v", events)
		}
	})
}
