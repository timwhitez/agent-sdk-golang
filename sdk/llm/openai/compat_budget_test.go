package openai

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func rejectWhenKey(key, msg string) func(map[string]any) (int, string, bool) {
	return func(body map[string]any) (int, string, bool) {
		if _, ok := body[key]; ok {
			return http.StatusBadRequest, msg, true
		}
		return 0, "", false
	}
}

func responsesInputHasContentArray(body map[string]any) bool {
	items, _ := body["input"].([]any)
	for _, item := range items {
		m, _ := item.(map[string]any)
		if _, ok := m["content"].([]any); ok {
			return true
		}
	}
	return false
}

// Compatibility downgrades are request changes, not transient retries: with
// MaxRetries 1 (an outer wrapper owns retries) each downgrade still resends
// once and succeeds, on every path where it applies.
func TestCompatibilityDowngradesDoNotConsumeRetryBudget(t *testing.T) {
	t.Parallel()
	maxTokens := 64
	type downgrade struct {
		name   string
		cases  []toolChoiceCase
		opts   compatClientOptions
		reject func(map[string]any) (int, string, bool)
	}
	all := toolChoiceCases
	downgrades := []downgrade{
		{
			name:   "reasoning_effort_chat",
			cases:  []toolChoiceCase{{"chat", false}, {"chat", true}},
			opts:   compatClientOptions{reasoningEffort: "low"},
			reject: rejectWhenKey("reasoning_effort", `{"error":{"message":"unknown parameter: reasoning_effort"}}`),
		},
		{
			name:   "reasoning_effort_responses",
			cases:  []toolChoiceCase{{"responses", false}, {"responses", true}},
			opts:   compatClientOptions{reasoningEffort: "low"},
			reject: rejectWhenKey("reasoning", `{"error":{"message":"unknown parameter: reasoning_effort"}}`),
		},
		{
			name:   "extra_body",
			cases:  all,
			opts:   compatClientOptions{extraBody: map[string]any{"vendor_flag": true}},
			reject: rejectWhenKey("extra_body", `{"error":{"message":"unknown field extra_body"}}`),
		},
		{
			name:   "thinking_extra",
			cases:  all,
			opts:   compatClientOptions{extra: map[string]any{"enable_thinking": true}},
			reject: rejectWhenKey("enable_thinking", `{"error":{"message":"enable_thinking is not supported"}}`),
		},
		{
			name:   "max_completion_tokens",
			cases:  []toolChoiceCase{{"chat", false}, {"chat", true}},
			opts:   compatClientOptions{maxCompletionTokens: &maxTokens},
			reject: rejectWhenKey("max_completion_tokens", `{"error":{"message":"Unsupported parameter: max_completion_tokens"}}`),
		},
		{
			name:   "stream_options",
			cases:  []toolChoiceCase{{"chat", true}},
			reject: rejectWhenKey("stream_options", `{"error":{"message":"unknown field stream_options"}}`),
		},
		{
			name:  "input_content_string",
			cases: []toolChoiceCase{{"responses", false}, {"responses", true}},
			reject: func(body map[string]any) (int, string, bool) {
				if responsesInputHasContentArray(body) {
					return http.StatusBadRequest, `{"error":{"code":"MissingParameter","message":"MissingParameter input.content"}}`, true
				}
				return 0, "", false
			},
		},
		{
			name:   "tool_choice",
			cases:  all,
			reject: rejectForcedToolChoice,
		},
	}
	for _, dg := range downgrades {
		for _, tc := range dg.cases {
			dg, tc := dg, tc
			t.Run(dg.name+"_"+tc.name(), func(t *testing.T) {
				t.Parallel()
				gw := &toolChoiceGateway{api: tc.api, reject: dg.reject}
				server := httptest.NewServer(gw)
				defer server.Close()
				opts := dg.opts
				opts.maxRetries = 1
				// Only the tool_choice case forces a tool; the others stay auto.
				choice := llm.ToolChoice("")
				if dg.name == "tool_choice" {
					choice = "required"
				}
				text, _, err := invokeCompatClient(t, tc.api, tc.stream, server.URL, &warningRecorder{}, opts, toolChoiceRequest(choice))
				if err != nil {
					t.Fatalf("invoke with MaxRetries 1: %v", err)
				}
				if text != "ok" {
					t.Fatalf("text = %q, want ok", text)
				}
				if got := len(gw.snapshot()); got != 2 {
					t.Fatalf("requests = %d, want 2 (one compatibility resend)", got)
				}
			})
		}
	}
}

// Several distinct downgrades on one request each get their own resend
// without any transient budget: reasoning_effort, then the forced tool_choice.
func TestCompatibilityDowngradesChainWithoutRetryBudget(t *testing.T) {
	t.Parallel()
	for _, tc := range toolChoiceCases {
		tc := tc
		t.Run(tc.name(), func(t *testing.T) {
			t.Parallel()
			key := "reasoning_effort"
			if tc.api == "responses" {
				key = "reasoning"
			}
			reject := func(body map[string]any) (int, string, bool) {
				if _, ok := body[key]; ok {
					return http.StatusBadRequest, `{"error":{"message":"unknown parameter: reasoning_effort"}}`, true
				}
				return rejectForcedToolChoice(body)
			}
			gw := &toolChoiceGateway{api: tc.api, reject: reject}
			server := httptest.NewServer(gw)
			defer server.Close()
			text, _, err := invokeCompatClient(t, tc.api, tc.stream, server.URL, &warningRecorder{}, compatClientOptions{maxRetries: 1, reasoningEffort: "high"}, toolChoiceRequest("required"))
			if err != nil || text != "ok" {
				t.Fatalf("text=%q err=%v, want ok", text, err)
			}
			if got := len(gw.snapshot()); got != 3 {
				t.Fatalf("requests = %d, want 3", got)
			}
		})
	}
}

// A downgrade that is already applied cannot resend again: a gateway that
// keeps answering MissingParameter input.content after string input is used
// gets exactly one compatibility resend.
func TestAppliedCompatibilityDowngradeDoesNotResendAgain(t *testing.T) {
	t.Parallel()
	for _, stream := range []bool{false, true} {
		stream := stream
		t.Run(map[bool]string{false: "buffered", true: "stream"}[stream], func(t *testing.T) {
			t.Parallel()
			always := func(map[string]any) (int, string, bool) {
				return http.StatusBadRequest, `{"error":{"code":"MissingParameter","message":"MissingParameter input.content"}}`, true
			}
			gw := &toolChoiceGateway{api: "responses", reject: always}
			server := httptest.NewServer(gw)
			defer server.Close()
			_, _, err := invokeCompatClient(t, "responses", stream, server.URL, &warningRecorder{}, compatClientOptions{maxRetries: 1}, toolChoiceRequest(""))
			var pe *llm.ProviderError
			if !errors.As(err, &pe) || pe.StatusCode != http.StatusBadRequest {
				t.Fatalf("err = %v, want provider 400", err)
			}
			if got := len(gw.snapshot()); got != 2 {
				t.Fatalf("requests = %d, want 2", got)
			}
		})
	}
}
