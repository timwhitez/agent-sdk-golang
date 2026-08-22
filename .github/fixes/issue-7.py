from pathlib import Path

chat = Path("sdk/llm/openai/chat.go")
text = chat.read_text()
old = '''\t\t\tif err != nil {
\t\t\t\tout <- llm.StreamErrorEvent{Err: err}
\t\t\t\treturn
\t\t\t}
\t\t\tout <- llm.StreamDoneEvent{StopReason: stopReason}
\t\t\treturn
'''
new = '''\t\t\tif err != nil {
\t\t\t\tout <- llm.StreamErrorEvent{Err: err}
\t\t\t\treturn
\t\t\t}
\t\t\tout <- llm.StreamErrorEvent{Err: &llm.ProviderError{
\t\t\t\tProvider: local.Provider(),
\t\t\t\tMessage:  fmt.Sprintf("stream ended before [DONE]; response is incomplete (model=%q endpoint=%s)", local.ModelName, endpoint),
\t\t\t}}
\t\t\treturn
'''
if text.count(old) != 1:
    raise SystemExit(f"chat EOF anchor count={text.count(old)}")
chat.write_text(text.replace(old, new))

responses = Path("sdk/llm/openai/responses.go")
text = responses.read_text()
old = '''\t\t\tstopReason := ""
\t\t\tthinkingEmitted := false
'''
new = '''\t\t\tstopReason := ""
\t\t\tsawCompleted := false
\t\t\tthinkingEmitted := false
'''
if text.count(old) != 1:
    raise SystemExit(f"responses terminal flag anchor count={text.count(old)}")
text = text.replace(old, new)
old = '''\t\t\t\tcase "response.completed":
\t\t\t\t\trespObj, _ := root["response"].(map[string]any)
'''
new = '''\t\t\t\tcase "response.completed":
\t\t\t\t\tsawCompleted = true
\t\t\t\t\trespObj, _ := root["response"].(map[string]any)
'''
if text.count(old) != 1:
    raise SystemExit(f"response.completed anchor count={text.count(old)}")
text = text.replace(old, new)
old = '''\t\t\tif err != nil {
\t\t\t\tout <- llm.StreamErrorEvent{Err: err}
\t\t\t\treturn
\t\t\t}
\t\t\tout <- llm.StreamDoneEvent{StopReason: stopReason}
\t\t\treturn
'''
new = '''\t\t\tif err != nil {
\t\t\t\tout <- llm.StreamErrorEvent{Err: err}
\t\t\t\treturn
\t\t\t}
\t\t\tif !sawCompleted {
\t\t\t\tout <- llm.StreamErrorEvent{Err: &llm.ProviderError{
\t\t\t\t\tProvider: local.Provider(),
\t\t\t\t\tMessage:  fmt.Sprintf("stream ended before response.completed or [DONE]; response is incomplete (model=%q endpoint=%s)", local.ModelName, endpoint),
\t\t\t\t}}
\t\t\t\treturn
\t\t\t}
\t\t\tout <- llm.StreamDoneEvent{StopReason: stopReason}
\t\t\treturn
'''
if text.count(old) != 1:
    raise SystemExit(f"responses EOF anchor count={text.count(old)}")
responses.write_text(text.replace(old, new))

Path("sdk/llm/openai/truncated_sse_test.go").write_text(r'''package openai

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
''')
