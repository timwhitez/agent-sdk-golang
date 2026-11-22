package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestMessageStopCompletesBeforeTransportEOF(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_terminal\",\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"complete\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
		w.(http.Flusher).Flush()
		// The logical message is complete, but the HTTP body stays open until
		// the adapter releases it. This must not hold the SDK terminal hostage.
		<-r.Context().Done()
	}))
	defer server.Close()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	client := &Client{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
	stream, err := client.InvokeStream(ctx, llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	text, done, usage := "", 0, 0
	for {
		select {
		case event, ok := <-stream:
			if !ok {
				if done != 1 || text != "complete" || usage == 0 {
					t.Fatalf("terminal=%d text=%q usage=%d", done, text, usage)
				}
				return
			}
			switch event := event.(type) {
			case llm.StreamDoneEvent:
				done++
			case llm.StreamTextDeltaEvent:
				text += event.Delta
			case llm.StreamUsageEvent:
				usage++
			case llm.StreamErrorEvent:
				t.Fatal(event.AsError())
			}
		case <-deadline.C:
			t.Fatalf("message_stop already arrived but stream awaits EOF: terminal=%d text=%q usage=%d", done, text, usage)
		}
	}
}

func TestMessageStopIsAnAuthoritativeSingleTerminal(t *testing.T) {
	const stop = `{"type":"message_stop"}`
	const failure = `{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`
	for _, tc := range []struct {
		name      string
		events    []string
		wantError int
		wantDone  int
	}{
		{"missing_stop", nil, 1, 0},
		{"error_before_stop", []string{failure, stop}, 1, 0},
		{"error_after_stop", []string{stop, failure}, 0, 1},
		{"duplicate_stop", []string{stop, stop}, 0, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test\",\"content\":[]}}\n\n")
				for _, event := range tc.events {
					fmt.Fprintf(w, "data: %s\n\n", event)
				}
			}))
			defer server.Close()
			client := &Client{BaseURL: server.URL, ModelName: "fixture", MaxRetries: 1}
			stream, err := client.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
			if err != nil {
				t.Fatal(err)
			}
			errors, done := 0, 0
			for event := range stream {
				switch event := event.(type) {
				case llm.StreamErrorEvent:
					errors++
					if tc.name == "missing_stop" && !strings.Contains(event.AsError().Error(), "before message_stop") {
						t.Fatal(event.AsError())
					}
				case llm.StreamDoneEvent:
					done++
				}
			}
			if errors != tc.wantError || done != tc.wantDone {
				t.Fatalf("errors=%d done=%d, want %d/%d", errors, done, tc.wantError, tc.wantDone)
			}
		})
	}
}
