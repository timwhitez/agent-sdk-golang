package llm_test

import (
	"context"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type d5Buffered struct{ llm.ChatModel }

func TestD5AbsentAnthropicUsageRemainsAbsent(t *testing.T) {
	for _, mode := range []string{"buffered", "streaming"} {
		for _, input := range []struct {
			name                    string
			usage                   string
			wantUsage, wantFallback int
		}{
			{"omitted", "", 0, 0},
			{"null", `,"usage":null`, 0, 0},
			{"partial", `,"usage":{"output_tokens":4}`, 1, 1},
			{"reported", `,"usage":{"input_tokens":18,"output_tokens":4}`, 1, 0},
		} {
			t.Run(mode+"/"+input.name, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					if mode == "streaming" {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprintf(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\"%s}}\n\n", input.usage)
						fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n")
						fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n")
						fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
						return
					}
					fmt.Fprintf(w, `{"id":"m1","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"%s}`, input.usage)
				}))
				defer server.Close()
				provider := &anthropic.Client{BaseURL: server.URL, HTTPClient: server.Client(), ModelName: "fixture", MaxRetries: 1}
				var model llm.ChatModel = provider
				if mode == "buffered" {
					model = d5Buffered{provider}
				}
				a, err := agent.New(agent.Config{LLM: model, MaxIterations: 1, InvokeRetryMaxAttempts: 1, Warningf: func(string, ...any) {}})
				if err != nil {
					t.Fatal(err)
				}
				var usages, fallbacks int
				var final, observedSource string
				for e := range a.QueryStream(context.Background(), llm.TextContent("hello")) {
					switch event := e.(type) {
					case agent.UsageEvent:
						usages++
						observedSource = event.Usage.PromptTokensSource
					case agent.WarnEvent:
						if event.Kind == "provider_usage_prompt_tokens_missing" {
							fallbacks++
						}
					case agent.FinalResponseEvent:
						final = event.Content
					case agent.ErrorEvent:
						t.Fatal(event.Message)
					}
				}
				if strings.TrimSpace(final) != "ok" {
					t.Fatalf("final=%q want ok", final)
				}
				if usages != input.wantUsage || fallbacks != input.wantFallback {
					t.Fatalf("usage events=%d want %d, estimation warnings=%d want %d, source=%q", usages, input.wantUsage, fallbacks, input.wantFallback, observedSource)
				}
			})
		}
	}
}
