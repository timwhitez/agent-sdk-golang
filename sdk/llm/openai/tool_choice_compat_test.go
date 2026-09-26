package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// gatewayToolChoiceRejection mirrors the live rejection from an
// OpenAI-compatible gateway serving a reasoning model when tool_choice forces
// a call. The request id is a fixture value.
const gatewayToolChoiceRejection = `{"error":{"code":"invalid_request_error","message":"Thinking mode does not support this tool_choice (request_id: req-fixture)","type":"invalid_request_error"}}`

const toolChoiceDowngradeWarning = "provider rejected forced tool_choice; retried with auto"

// Reviewer probes from #192: errors that mention tool_choice but do not say
// the forced choice itself is unsupported.
const (
	invalidToolChoiceValueRejection = `{"error":{"message":"Invalid value: 'requird'. Supported values are: 'none', 'auto', and 'required'.","type":"invalid_request_error","param":"tool_choice","code":"invalid_value"}}`
	parallelToolCallsRejection      = `{"error":{"message":"parallel_tool_calls is not supported with tool_choice required","type":"invalid_request_error","param":null,"code":null}}`
	temperatureRejection            = `{"error":{"message":"Unsupported value: 'temperature' does not support 0.2 with this model (request had tool_choice=auto).","type":"invalid_request_error","param":null,"code":"unsupported_value"}}`
	// structuredToolChoiceRejection names tool_choice only through param.
	structuredToolChoiceRejection = `{"error":{"message":"Forced tool calls are not supported in thinking mode.","type":"invalid_request_error","param":"tool_choice","code":null}}`
)

func TestLooksLikeToolChoiceUnsupported(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		body string
		want bool
	}{
		// Positive: the refusal refers to tool_choice itself.
		{"live_gateway_thinking_mode", gatewayToolChoiceRejection, true},
		{"live_gateway_plain_text", "Thinking mode does not support this tool_choice (request_id: req-fixture)", true},
		{"tool_choice_value_not_supported", `{"error":{"message":"tool_choice 'required' is not supported for this model"}}`, true},
		{"quoted_tool_choice_does_not_support", `{"error":{"message":"Unsupported value: 'tool_choice' does not support 'required' with this model."}}`, true},
		{"doesnt_support_forced_tool_choice", `{"error":{"message":"model doesn't support forced tool_choice"}}`, true},
		{"unsupported_tool_choice", `{"error":{"message":"unsupported tool_choice for this model"}}`, true},
		{"param_tool_choice_with_refusal_message", structuredToolChoiceRejection, true},
		{"param_tool_choice_with_unsupported_code", `{"error":{"message":"The requested value cannot be used with this model.","param":"tool_choice","code":"unsupported_value"}}`, true},
		{"top_level_param_tool_choice", `{"message":"Forced tool use is not supported by this model","param":"tool_choice"}`, true},
		{"string_error_adjacent", `{"error":"this model does not support the tool_choice parameter"}`, true},

		// Negative: #192 reviewer probes.
		{"invalid_value_param_tool_choice", invalidToolChoiceValueRejection, false},
		{"invalid_value_without_param", `{"error":{"message":"Invalid value for tool_choice: 'requird' is not supported"}}`, false},
		{"unknown_value_not_supported", `{"error":{"message":"tool_choice 'requird' is not supported"}}`, false},
		{"invalid_value_code_with_refusal_text", `{"error":{"message":"'requird' is not supported","param":"tool_choice","code":"invalid_value"}}`, false},
		{"invalid_value_text_with_refusal_text", `{"error":{"message":"Invalid value for tool_choice: value is not supported","param":"tool_choice"}}`, false},
		{"parallel_tool_calls_mentions_tool_choice", parallelToolCallsRejection, false},
		{"parallel_tool_calls_param", `{"error":{"message":"parallel_tool_calls is not supported with tool_choice required","param":"parallel_tool_calls"}}`, false},
		{"temperature_mentions_tool_choice_auto", temperatureRejection, false},
		{"temperature_param_with_adjacent_text", `{"error":{"message":"temperature does not support this tool_choice","param":"temperature"}}`, false},
		{"temperature_plain_text", "temperature is unsupported when tool_choice=auto", false},

		// Negative: "invalid" wording describes a malformed value, not a
		// capability gap; a silent auto would hide it.
		{"invalid_tool_choice", `{"error":{"message":"Invalid tool_choice: function choice unavailable in reasoning mode"}}`, false},
		{"invalid_parameter_tool_choice", `{"error":{"message":"Invalid parameter: tool_choice"}}`, false},
		{"param_tool_choice_without_refusal", `{"error":{"message":"Function 'x' named in tool_choice was not found in tools.","param":"tool_choice"}}`, false},
		{"param_tool_choice_function_name", `{"error":{"message":"not supported","param":"tool_choice.function.name"}}`, false},

		// Negative: unrelated errors.
		{"empty", "", false},
		{"context_length", `{"error":{"code":"invalid_request_error","message":"This model's maximum context length is 8192 tokens","type":"invalid_request_error"}}`, false},
		{"tool_choice_requires_tools", `{"error":{"code":"invalid_request_error","message":"tool_choice requires tools to be provided","type":"invalid_request_error"}}`, false},
		{"thinking_temperature", `{"error":{"message":"Thinking mode does not support this temperature"}}`, false},
		{"unknown_reasoning_effort", `{"error":{"message":"unknown field reasoning_effort"}}`, false},
		{"invalid_input", `{"error":{"message":"Invalid value for 'input[2].content'"}}`, false},
		{"invalid_tool_name", `{"error":{"message":"tools[0].function.name is invalid"}}`, false},
	}
	for _, tc := range cases {
		if got := looksLikeToolChoiceUnsupported(tc.body); got != tc.want {
			t.Errorf("%s: looksLikeToolChoiceUnsupported(%q) = %v, want %v", tc.name, tc.body, got, tc.want)
		}
	}
}

// toolChoiceGateway is a loopback OpenAI-compatible endpoint that rejects
// requests according to reject and otherwise answers "ok" in the shape the
// request asked for (Chat or Responses, streaming or buffered).
type toolChoiceGateway struct {
	api    string // "chat" or "responses"
	reject func(body map[string]any) (int, string, bool)

	mu     sync.Mutex
	bodies []map[string]any
}

func forcedWireToolChoice(body map[string]any) bool {
	switch tc := body["tool_choice"].(type) {
	case string:
		return tc == "required"
	case map[string]any:
		return true
	}
	return false
}

func rejectForcedToolChoice(body map[string]any) (int, string, bool) {
	if forcedWireToolChoice(body) {
		return http.StatusBadRequest, gatewayToolChoiceRejection, true
	}
	return 0, "", false
}

func (g *toolChoiceGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	g.mu.Lock()
	g.bodies = append(g.bodies, body)
	g.mu.Unlock()
	if status, msg, ok := g.reject(body); ok {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, msg)
		return
	}
	stream, _ := body["stream"].(bool)
	switch {
	case g.api == "chat" && stream:
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	case g.api == "chat":
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chat_1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	case stream:
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_1\",\"delta\":\"ok\"}\n\ndata: [DONE]\n\n")
	default:
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`)
	}
}

func (g *toolChoiceGateway) snapshot() []map[string]any {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]map[string]any(nil), g.bodies...)
}

type warningRecorder struct {
	mu   sync.Mutex
	msgs []string
}

func (w *warningRecorder) warnf(format string, args ...any) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.msgs = append(w.msgs, fmt.Sprintf(format, args...))
}

func (w *warningRecorder) count(substr string) int {
	w.mu.Lock()
	defer w.mu.Unlock()
	n := 0
	for _, m := range w.msgs {
		if strings.Contains(m, substr) {
			n++
		}
	}
	return n
}

func toolChoiceRequest(choice llm.ToolChoice) llm.InvokeRequest {
	return llm.InvokeRequest{
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
		Tools:      []llm.ToolDefinition{{Name: "done", Description: "finish", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}},
		ToolChoice: choice,
	}
}

// invokeToolChoiceClient runs one request through the selected client and
// path and returns the text, diagnostics (buffered only) and error.
func invokeToolChoiceClient(t *testing.T, api string, stream bool, baseURL string, warn *warningRecorder, req llm.InvokeRequest) (string, []llm.Diagnostic, error) {
	t.Helper()
	return invokeCompatClient(t, api, stream, baseURL, warn, compatClientOptions{maxRetries: 1}, req)
}

// compatClientOptions configures the client under test. maxRetries 1 is the
// production shape when an outer wrapper owns transient retries.
type compatClientOptions struct {
	maxRetries          int
	reasoningEffort     string
	extra, extraBody    map[string]any
	maxCompletionTokens *int
}

func invokeCompatClient(t *testing.T, api string, stream bool, baseURL string, warn *warningRecorder, opts compatClientOptions, req llm.InvokeRequest) (string, []llm.Diagnostic, error) {
	t.Helper()
	var model interface {
		Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error)
		InvokeStream(context.Context, llm.InvokeRequest) (<-chan llm.StreamEvent, error)
	}
	if api == "chat" {
		model = &ChatClient{BaseURL: baseURL, ModelName: "test-model", MaxRetries: opts.maxRetries, Warningf: warn.warnf,
			ReasoningEffort: opts.reasoningEffort, Extra: opts.extra, ExtraBody: opts.extraBody, MaxCompletionTokens: opts.maxCompletionTokens}
	} else {
		model = &ResponsesClient{BaseURL: baseURL, ModelName: "test-model", MaxRetries: opts.maxRetries, Warningf: warn.warnf,
			ReasoningEffort: opts.reasoningEffort, Extra: opts.extra, ExtraBody: opts.extraBody}
	}
	if !stream {
		comp, err := model.Invoke(context.Background(), req)
		if err != nil {
			return "", nil, err
		}
		return comp.Content.PlainText(), comp.Diagnostics, nil
	}
	events, err := model.InvokeStream(context.Background(), req)
	if err != nil {
		return "", nil, err
	}
	var text strings.Builder
	var streamErr error
	for ev := range events {
		switch e := ev.(type) {
		case llm.StreamTextDeltaEvent:
			text.WriteString(e.Delta)
		case llm.StreamErrorEvent:
			streamErr = e.AsError()
		}
	}
	return text.String(), nil, streamErr
}

type toolChoiceCase struct {
	api    string
	stream bool
}

func (c toolChoiceCase) name() string {
	mode := "buffered"
	if c.stream {
		mode = "stream"
	}
	return c.api + "_" + mode
}

var toolChoiceCases = []toolChoiceCase{{"chat", false}, {"chat", true}, {"responses", false}, {"responses", true}}

func withoutToolChoice(body map[string]any) map[string]any {
	out := make(map[string]any, len(body))
	for k, v := range body {
		if k != "tool_choice" {
			out[k] = v
		}
	}
	return out
}

// A forced tool_choice rejected by the provider is retried exactly once with
// the client's auto representation; nothing else in the request changes and
// the downgrade is reported through the warning sink.
func TestForcedToolChoiceRejectedRetriesOnceWithAuto(t *testing.T) {
	t.Parallel()
	for _, tc := range toolChoiceCases {
		for _, choice := range []llm.ToolChoice{"required", "done"} {
			for _, maxRetries := range []int{1, 3} {
				t.Run(fmt.Sprintf("%s_%s_max%d", tc.name(), choice, maxRetries), func(t *testing.T) {
					t.Parallel()
					gw := &toolChoiceGateway{api: tc.api, reject: rejectForcedToolChoice}
					server := httptest.NewServer(gw)
					defer server.Close()
					warn := &warningRecorder{}

					text, diags, err := invokeCompatClient(t, tc.api, tc.stream, server.URL, warn, compatClientOptions{maxRetries: maxRetries}, toolChoiceRequest(choice))
					if err != nil {
						t.Fatalf("invoke: %v", err)
					}
					if text != "ok" {
						t.Fatalf("text = %q, want ok", text)
					}
					bodies := gw.snapshot()
					if len(bodies) != 2 {
						t.Fatalf("requests = %d, want 2", len(bodies))
					}
					if !forcedWireToolChoice(bodies[0]) {
						t.Fatalf("first request tool_choice = %#v, want forced", bodies[0]["tool_choice"])
					}
					switch tc.api {
					case "chat":
						if got := bodies[1]["tool_choice"]; got != "auto" {
							t.Fatalf("retry tool_choice = %#v, want auto", got)
						}
					default:
						if got, ok := bodies[1]["tool_choice"]; ok {
							t.Fatalf("retry tool_choice = %#v, want omitted (auto)", got)
						}
					}
					if !reflect.DeepEqual(withoutToolChoice(bodies[0]), withoutToolChoice(bodies[1])) {
						t.Fatalf("retry changed more than tool_choice:\nfirst: %#v\nretry: %#v", bodies[0], bodies[1])
					}
					if got := warn.count(toolChoiceDowngradeWarning); got != 1 {
						t.Fatalf("downgrade warnings = %d, want 1 (all: %q)", got, warn.msgs)
					}
					if !tc.stream {
						found := 0
						for _, d := range diags {
							if d.Kind == "provider_compatibility_downgrade" && strings.Contains(d.Message, toolChoiceDowngradeWarning) {
								found++
							}
						}
						if found != 1 {
							t.Fatalf("downgrade diagnostics = %d, want 1 (%#v)", found, diags)
						}
					}
				})
			}
		}
	}
}

// The downgrade is local to one request: the next forced request on the same
// client still sends the forced tool_choice first.
func TestForcedToolChoiceDowngradeIsNotSticky(t *testing.T) {
	t.Parallel()
	for _, api := range []string{"chat", "responses"} {
		t.Run(api, func(t *testing.T) {
			t.Parallel()
			gw := &toolChoiceGateway{api: api, reject: rejectForcedToolChoice}
			server := httptest.NewServer(gw)
			defer server.Close()
			warn := &warningRecorder{}
			var model llm.ChatModel
			if api == "chat" {
				model = &ChatClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1, Warningf: warn.warnf}
			} else {
				model = &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1, Warningf: warn.warnf}
			}
			for i := 0; i < 2; i++ {
				if _, err := model.Invoke(context.Background(), toolChoiceRequest("required")); err != nil {
					t.Fatalf("invoke %d: %v", i, err)
				}
			}
			bodies := gw.snapshot()
			if len(bodies) != 4 {
				t.Fatalf("requests = %d, want 4", len(bodies))
			}
			if !forcedWireToolChoice(bodies[2]) {
				t.Fatalf("second invoke first tool_choice = %#v, want forced (downgrade must not be sticky)", bodies[2]["tool_choice"])
			}
		})
	}
}

// Requests that must not be downgraded fail on the first rejection: an
// unrelated 400 for a forced request, an auto or none request that the
// provider rejects with the tool_choice message, and a gateway that keeps
// rejecting after the single downgrade.
func TestToolChoiceDowngradeNegativeCases(t *testing.T) {
	t.Parallel()
	const unrelated = `{"error":{"code":"invalid_request_error","message":"This model's maximum context length is 8192 tokens","type":"invalid_request_error"}}`
	type negCase struct {
		name     string
		choice   llm.ToolChoice
		reject   func(map[string]any) (int, string, bool)
		requests int
	}
	always := func(msg string) func(map[string]any) (int, string, bool) {
		return func(map[string]any) (int, string, bool) { return http.StatusBadRequest, msg, true }
	}
	cases := []negCase{
		{name: "unrelated_400_forced", choice: "required", reject: always(unrelated), requests: 1},
		{name: "auto_not_retried", choice: "", reject: always(gatewayToolChoiceRejection), requests: 1},
		{name: "explicit_auto_not_retried", choice: "auto", reject: always(gatewayToolChoiceRejection), requests: 1},
		{name: "none_not_retried", choice: "none", reject: always(gatewayToolChoiceRejection), requests: 1},
		{name: "one_downgrade_per_request", choice: "required", reject: always(gatewayToolChoiceRejection), requests: 2},
		{name: "structured_param_downgrades_once", choice: "required", reject: always(structuredToolChoiceRejection), requests: 2},
		{name: "invalid_value_not_retried", choice: "required", reject: always(invalidToolChoiceValueRejection), requests: 1},
		{name: "parallel_tool_calls_not_retried", choice: "required", reject: always(parallelToolCallsRejection), requests: 1},
		{name: "temperature_not_retried", choice: "required", reject: always(temperatureRejection), requests: 1},
		// A named choice that is not a declared tool is a caller error, even
		// when the provider's text would otherwise match.
		{name: "undeclared_named_choice_not_retried", choice: "requird", reject: always(gatewayToolChoiceRejection), requests: 1},
		{name: "declared_named_choice_downgrades_once", choice: "done", reject: always(gatewayToolChoiceRejection), requests: 2},
	}
	for _, tc := range toolChoiceCases {
		for _, nc := range cases {
			t.Run(tc.name()+"_"+nc.name, func(t *testing.T) {
				t.Parallel()
				gw := &toolChoiceGateway{api: tc.api, reject: nc.reject}
				server := httptest.NewServer(gw)
				defer server.Close()
				warn := &warningRecorder{}
				_, _, err := invokeToolChoiceClient(t, tc.api, tc.stream, server.URL, warn, toolChoiceRequest(nc.choice))
				var pe *llm.ProviderError
				if !errors.As(err, &pe) || pe.StatusCode != http.StatusBadRequest {
					t.Fatalf("err = %v, want provider 400", err)
				}
				if got := len(gw.snapshot()); got != nc.requests {
					t.Fatalf("requests = %d, want %d", got, nc.requests)
				}
				wantWarnings := nc.requests - 1
				if got := warn.count(toolChoiceDowngradeWarning); got != wantWarnings {
					t.Fatalf("downgrade warnings = %d, want %d", got, wantWarnings)
				}
			})
		}
	}
}
