package llm_test

import (
	"encoding/json"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestR7AnthropicInterruptedUsageBoundary(t *testing.T) {
	for _, shape := range []string{"redacted_thinking", "text"} {
		t.Run(shape, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req map[string]any
				err := json.NewDecoder(r.Body).Decode(&req)
				thinking, _ := req["thinking"].(map[string]any)
				if err != nil || thinking["type"] != "enabled" {
					t.Errorf("expected explicit enabled thinking: body=%v err=%v", req, err)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				event := func(kind string, fields map[string]any) {
					fields["type"] = kind
					data, err := json.Marshal(fields)
					if err != nil {
						t.Error(err)
						return
					}
					fmt.Fprintf(w, "event: %s\ndata: %s\n\n", kind, data)
				}
				event("message_start", map[string]any{"message": map[string]any{"id": "r1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-20250514", "content": []any{}, "stop_reason": nil, "stop_sequence": nil, "usage": map[string]int{"input_tokens": 19, "output_tokens": 2}}})
				if shape == "redacted_thinking" {
					event("content_block_start", map[string]any{"index": 0, "content_block": map[string]string{"type": shape, "data": "opaque-fixture"}})
				} else {
					event("content_block_start", map[string]any{"index": 0, "content_block": map[string]string{"type": shape, "text": ""}})
					event("content_block_delta", map[string]any{"index": 0, "delta": map[string]string{"type": "text_delta", "text": "draft"}})
				}
				event("content_block_stop", map[string]any{"index": 0})
				event("error", map[string]any{"error": map[string]string{"type": "api_error", "message": "fixture ended"}})
			}))
			defer s.Close()
			budget := 1024
			model := &anthropic.Client{BaseURL: s.URL, ModelName: "claude-sonnet-4-20250514", MaxRetries: 1, ThinkingBudgetTokens: &budget}
			ch, err := model.InvokeStream(t.Context(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("check")}})
			if err != nil {
				t.Fatal(err)
			}
			var id string
			var usage, visible, failures int
			for event := range ch {
				switch e := event.(type) {
				case llm.StreamResponseEvent:
					id = e.ResponseID
				case llm.StreamUsageEvent:
					usage = e.Usage.TotalTokens
				case llm.StreamThinkingDeltaEvent:
					if e.BlockType == shape && e.Data != "" {
						visible++
					}
				case llm.StreamTextDeltaEvent:
					if shape == "text" && e.Delta == "draft" {
						visible++
					}
				case llm.StreamErrorEvent:
					failures++
				}
			}
			if id != "r1" || usage != 21 || visible != 1 || failures != 1 {
				t.Fatalf("adapter id=%q usage=%d blocks=%d errors=%d", id, usage, visible, failures)
			}
			t.Logf("adapter id=%q tokens=%d expected-blocks=%d errors=%d", id, usage, visible, failures)
			a, err := agent.New(agent.Config{LLM: model, InvokeRetryMaxAttempts: 1, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			id = ""
			usage = 0
			failures = 0
			var usageEvents, accounting int
			for event := range a.QueryStream(t.Context(), llm.TextContent("check")) {
				switch e := event.(type) {
				case agent.UsageEvent:
					id = e.ResponseID
					usage = e.Usage.TotalTokens
					usageEvents++
				case agent.AccountingEvent:
					accounting++
				case agent.ErrorEvent:
					failures++
				}
			}
			t.Logf("agent id=%q tokens=%d usage-events=%d accounting-events=%d errors=%d", id, usage, usageEvents, accounting, failures)
			if id != "r1" || usage != 21 || usageEvents != 1 || accounting != 1 || failures != 1 {
				t.Error("agent lost observed metadata")
			}
		})
	}
}
