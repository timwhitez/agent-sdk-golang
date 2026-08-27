package openai

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestResponsesInvokeReturnsFailedRefusalWithTypedError(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_failed","status":"failed","error":{"code":"server_error","message":"generation failed"},"output":[{"type":"message","content":[{"type":"refusal","refusal":"Request refused."}]}]}`))
	}))
	defer server.Close()

	client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1, ProviderLabel: "gateway-responses"}
	completion, err := client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("error = %v (%T), want ProviderError", err, err)
	}
	if completion == nil || completion.PlainText() != "Request refused." {
		t.Fatalf("partial completion = %#v, want visible refusal", completion)
	}
	if providerErr.Provider != "gateway-responses" || !strings.Contains(providerErr.Message, "Request refused.") {
		t.Fatalf("provider error = %#v", providerErr)
	}
}

func TestParseResponsesTerminalStatuses(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name           string
		payload        string
		wantStopReason string
		wantText       string
		wantError      bool
	}{
		{
			name:           "max output tokens",
			payload:        `{"id":"resp_max","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","content":[{"type":"output_text","text":"partial"}]}]}`,
			wantStopReason: "max_tokens",
			wantText:       "partial",
		},
		{
			name:           "content filter refusal",
			payload:        `{"id":"resp_filter","status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[{"type":"message","content":[{"type":"refusal","refusal":"I cannot help with that."}]}]}`,
			wantStopReason: "content_filter",
			wantText:       "I cannot help with that.",
		},
		{
			name:      "legacy max tokens reason is not auto continued",
			payload:   `{"id":"resp_legacy","status":"incomplete","incomplete_details":{"reason":"max_tokens"},"output":[{"type":"message","content":[{"type":"output_text","text":"partial"}]}]}`,
			wantText:  "partial",
			wantError: true,
		},
		{
			name:      "failed refusal remains visible",
			payload:   `{"id":"resp_failed","status":"failed","error":{"code":"server_error","message":"generation failed"},"output":[{"type":"message","content":[{"type":"refusal","refusal":"Request refused."}]}]}`,
			wantText:  "Request refused.",
			wantError: true,
		},
		{
			name:      "cancelled",
			payload:   `{"id":"resp_cancelled","status":"cancelled","output":[]}`,
			wantError: true,
		},
		{
			name:      "queued 2xx",
			payload:   `{"id":"resp_queued","status":"queued","output":[]}`,
			wantError: true,
		},
		{
			name:      "in progress 2xx",
			payload:   `{"id":"resp_progress","status":"in_progress","output":[]}`,
			wantError: true,
		},
		{
			name:      "root error overrides completed status",
			payload:   `{"id":"resp_error","status":"completed","error":{"code":"server_error","message":"provider failed"},"output":[]}`,
			wantError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			completion, err := parseResponsesForProvider("gateway-responses", []byte(tc.payload))
			if completion == nil {
				t.Fatal("terminal payload returned no partial completion")
			}
			if completion.StopReason != tc.wantStopReason {
				t.Fatalf("stop reason = %q, want %q", completion.StopReason, tc.wantStopReason)
			}
			if completion.PlainText() != tc.wantText {
				t.Fatalf("text = %q, want %q", completion.PlainText(), tc.wantText)
			}
			if tc.wantError {
				var providerErr *llm.ProviderError
				if !errors.As(err, &providerErr) {
					t.Fatalf("error = %v (%T), want ProviderError", err, err)
				}
				if providerErr.Provider != "gateway-responses" {
					t.Fatalf("provider = %q, want gateway-responses", providerErr.Provider)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestResponsesStreamIncompleteTerminalEvents(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name           string
		reason         string
		wantStopReason string
		wantError      bool
	}{
		{name: "max output tokens", reason: "max_output_tokens", wantStopReason: "max_tokens"},
		{name: "content filter", reason: "content_filter", wantStopReason: "content_filter"},
		{name: "unknown reason", reason: "provider_extension", wantError: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			server := openAIStreamFixture(t, `{"type":"response.incomplete","response":{"id":"resp_incomplete","status":"incomplete","incomplete_details":{"reason":"`+tc.reason+`"},"output":[{"type":"message","content":[{"type":"output_text","text":"partial"}]}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}`)
			defer server.Close()
			client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1, ProviderLabel: "gateway-responses"}
			stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
			if err != nil {
				t.Fatal(err)
			}
			events := collectOpenAIStream(stream)
			assertResponsesTerminalMetadata(t, events, tc.wantStopReason, tc.wantError)
		})
	}
}

func TestResponsesStreamFailedPreservesRefusalAndMetadata(t *testing.T) {
	t.Parallel()
	server := openAIStreamFixture(t,
		`{"type":"response.refusal.delta","response_id":"resp_failed","delta":"Request"}`,
		`{"type":"response.refusal.done","response_id":"resp_failed","refusal":"Request refused."}`,
		`{"type":"response.failed","response":{"id":"resp_failed","status":"failed","error":{"code":"server_error","message":"generation failed"},"output":[{"type":"message","content":[{"type":"refusal","refusal":"Request refused."}]}],"usage":{"input_tokens":7,"output_tokens":1,"total_tokens":8}}}`,
	)
	defer server.Close()
	client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1, ProviderLabel: "gateway-responses"}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	events := collectOpenAIStream(stream)
	assertResponsesTerminalMetadata(t, events, "", true)

	text := ""
	var providerErr *llm.ProviderError
	for _, event := range events {
		switch typed := event.(type) {
		case llm.StreamTextDeltaEvent:
			text += typed.Delta
		case llm.StreamErrorEvent:
			if !errors.As(typed.AsError(), &providerErr) {
				t.Fatalf("error = %v, want ProviderError", typed.AsError())
			}
		}
	}
	if text != "Request refused." {
		t.Fatalf("refusal text = %q, want exactly one streamed refusal", text)
	}
	if providerErr == nil || providerErr.Provider != "gateway-responses" {
		t.Fatalf("provider error = %#v", providerErr)
	}
}

func TestResponsesStreamDoneCannotPromoteInProgressResponse(t *testing.T) {
	t.Parallel()
	server := openAIStreamFixture(t,
		`{"type":"response.in_progress","response":{"id":"resp_progress","status":"in_progress","output":[]}}`,
		`[DONE]`,
	)
	defer server.Close()
	client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1, ProviderLabel: "gateway-responses"}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	events := collectOpenAIStream(stream)
	var providerErr *llm.ProviderError
	for _, event := range events {
		switch typed := event.(type) {
		case llm.StreamDoneEvent:
			t.Fatalf("in-progress response emitted success: %#v", events)
		case llm.StreamErrorEvent:
			if !errors.As(typed.AsError(), &providerErr) {
				t.Fatalf("error = %v, want ProviderError", typed.AsError())
			}
		}
	}
	if providerErr == nil || providerErr.Provider != "gateway-responses" || !strings.Contains(providerErr.Message, "in_progress") {
		t.Fatalf("provider error = %#v", providerErr)
	}
}

func TestResponsesStreamRejectsTerminalEventStatusConflict(t *testing.T) {
	t.Parallel()
	server := openAIStreamFixture(t,
		`{"type":"response.incomplete","response":{"id":"resp_conflict","status":"failed","incomplete_details":{"reason":"max_output_tokens"},"output":[],"usage":{"input_tokens":4,"output_tokens":1,"total_tokens":5}}}`,
	)
	defer server.Close()
	client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1, ProviderLabel: "gateway-responses"}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	events := collectOpenAIStream(stream)
	responseIndex, usageIndex, errorIndex := -1, -1, -1
	var providerErr *llm.ProviderError
	for index, event := range events {
		switch typed := event.(type) {
		case llm.StreamResponseEvent:
			responseIndex = index
		case llm.StreamUsageEvent:
			usageIndex = index
		case llm.StreamDoneEvent:
			t.Fatalf("conflicting terminal status emitted success: %#v", events)
		case llm.StreamErrorEvent:
			errorIndex = index
			if !errors.As(typed.AsError(), &providerErr) {
				t.Fatalf("error = %v, want ProviderError", typed.AsError())
			}
		}
	}
	if responseIndex < 0 || usageIndex < 0 || errorIndex < 0 || responseIndex >= errorIndex || usageIndex >= errorIndex {
		t.Fatalf("metadata/error order = %d/%d/%d: %#v", responseIndex, usageIndex, errorIndex, events)
	}
	if providerErr == nil || !strings.Contains(providerErr.Message, "conflicts") {
		t.Fatalf("provider error = %#v", providerErr)
	}
}

func TestResponsesStreamRefusalDoneWithoutDeltaIsVisible(t *testing.T) {
	t.Parallel()
	server := openAIStreamFixture(t,
		`{"type":"response.refusal.done","response_id":"resp_refusal","refusal":"I cannot comply."}`,
		`{"type":"response.completed","response":{"id":"resp_refusal","status":"completed","output":[]}}`,
	)
	defer server.Close()
	client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	events := collectOpenAIStream(stream)
	text := ""
	doneReason := ""
	for _, event := range events {
		switch typed := event.(type) {
		case llm.StreamTextDeltaEvent:
			text += typed.Delta
		case llm.StreamDoneEvent:
			doneReason = typed.StopReason
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected error: %v", typed.AsError())
		}
	}
	if text != "I cannot comply." || doneReason != "end_turn" {
		t.Fatalf("text/reason = %q/%q", text, doneReason)
	}
}

func assertResponsesTerminalMetadata(t *testing.T, events []llm.StreamEvent, wantStopReason string, wantError bool) {
	t.Helper()
	responseIndex, usageIndex, terminalIndex := -1, -1, -1
	terminalIsError := false
	for index, event := range events {
		switch typed := event.(type) {
		case llm.StreamResponseEvent:
			if typed.ResponseID == "resp_incomplete" || typed.ResponseID == "resp_failed" {
				responseIndex = index
			}
		case llm.StreamUsageEvent:
			if typed.Usage.TotalTokens != 5 && typed.Usage.TotalTokens != 8 {
				t.Fatalf("usage = %#v", typed.Usage)
			}
			usageIndex = index
		case llm.StreamDoneEvent:
			terminalIndex = index
			if typed.StopReason != wantStopReason {
				t.Fatalf("stop reason = %q, want %q", typed.StopReason, wantStopReason)
			}
		case llm.StreamErrorEvent:
			terminalIndex = index
			terminalIsError = true
			var providerErr *llm.ProviderError
			if !errors.As(typed.AsError(), &providerErr) {
				t.Fatalf("terminal error = %v, want ProviderError", typed.AsError())
			}
		}
	}
	if responseIndex < 0 || usageIndex < 0 || terminalIndex < 0 {
		t.Fatalf("missing response/usage/terminal events: %#v", events)
	}
	if responseIndex >= terminalIndex || usageIndex >= terminalIndex {
		t.Fatalf("metadata followed terminal event: %#v", events)
	}
	if terminalIsError != wantError {
		t.Fatalf("terminal error = %t, want %t: %#v", terminalIsError, wantError, events)
	}
}
