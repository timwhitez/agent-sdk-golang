package anthropic_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
)

// Anthropic documents no structured context-overflow field: its 400
// invalid_request_error is a broad class and 413 is a byte-size limit. The
// adapter therefore never types a context overflow, even when the message
// text says the prompt is too long, and a real Agent with compaction and
// overflow recovery enabled makes exactly one request with no summary.
func TestAnthropicContextOverflowStaysUnsupported(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
		stream string
	}{
		{
			name:   "400 invalid_request_error prompt too long",
			status: http.StatusBadRequest,
			body:   `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 208000 tokens > 200000 maximum"}}`,
		},
		{
			name:   "413 request_too_large",
			status: http.StatusRequestEntityTooLarge,
			body:   `{"type":"error","error":{"type":"request_too_large","message":"Request exceeds the maximum allowed number of bytes."}}`,
		},
		{
			name:   "stream error event prompt too long",
			stream: `data: {"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 208000 tokens > 200000 maximum"}}` + "\n\n",
		},
	}
	history := []llm.Message{llm.NewSystemMessage("system")}
	for i := 0; i < 8; i++ {
		history = append(history, llm.NewUserMessage(strings.Repeat("earlier request ", 60)), llm.NewAssistantMessage(strings.Repeat("earlier answer ", 60), nil))
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			main, summary := 0, 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				data, _ := io.ReadAll(r.Body)
				mu.Lock()
				if strings.Contains(string(data), "operational checkpoint") {
					summary++
				} else {
					main++
				}
				mu.Unlock()
				if tc.stream != "" {
					w.Header().Set("Content-Type", "text/event-stream")
					io.WriteString(w, tc.stream)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				io.WriteString(w, tc.body)
			}))
			defer server.Close()
			client := &anthropic.Client{BaseURL: server.URL, APIKey: "test", ModelName: "m", MaxTokens: 64, MaxRetries: 1}

			req := llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hi")}}
			_, buffered := client.Invoke(context.Background(), req)
			var streamed error
			if events, err := client.InvokeStream(context.Background(), req); err != nil {
				streamed = err
			} else {
				for ev := range events {
					if e, ok := ev.(llm.StreamErrorEvent); ok {
						streamed = e.AsError()
					}
				}
			}
			if buffered == nil || streamed == nil {
				t.Fatalf("expected errors: buffered=%v streamed=%v", buffered, streamed)
			}
			if llm.IsContextOverflow(buffered) || llm.IsContextOverflow(streamed) {
				t.Fatalf("anthropic error typed as context overflow: buffered=%v streamed=%v", buffered, streamed)
			}

			mu.Lock()
			main, summary = 0, 0
			mu.Unlock()
			ag, err := agent.New(agent.Config{
				LLM:                    client,
				InitialMessages:        history,
				InvokeRetryMaxAttempts: 1,
				Compaction:             &compaction.Config{Enabled: true, ContextWindow: 100000, ThresholdRatio: 0.85},
				Warningf:               func(string, ...any) {},
			})
			if err != nil {
				t.Fatal(err)
			}
			errorsSeen, compactions, recoveries := 0, 0, 0
			for ev := range ag.QueryStream(context.Background(), llm.TextContent("current request")) {
				switch e := ev.(type) {
				case agent.ErrorEvent:
					errorsSeen++
				case agent.CompactionEvent:
					compactions++
				case agent.WarnEvent:
					if e.Kind == "context_overflow_recovery" {
						recoveries++
					}
				}
			}
			mu.Lock()
			defer mu.Unlock()
			if main != 1 || summary != 0 || compactions != 0 || recoveries != 0 || errorsSeen != 1 {
				t.Fatalf("main=%d summary=%d compactions=%d recoveries=%d errors=%d", main, summary, compactions, recoveries, errorsSeen)
			}
		})
	}
}
