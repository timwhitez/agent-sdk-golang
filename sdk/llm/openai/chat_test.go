package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

type timeoutErr struct{}

func (timeoutErr) Error() string { return "timeout" }
func (timeoutErr) Timeout() bool { return true }

type closeTrackingReadCloser struct {
	io.Reader
	closed bool
}

func (c *closeTrackingReadCloser) Close() error {
	c.closed = true
	return nil
}

func newErrorClient(status int, body string) *http.Client {
	return &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: status,
				Status:     fmt.Sprintf("%d %s", status, http.StatusText(status)),
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
				Request:    r,
			}, nil
		}),
	}
}

func TestChatClientDowngradeDoesNotMutateConfig(t *testing.T) {
	t.Parallel()
	httpClient := newErrorClient(http.StatusBadRequest, "unknown field reasoning_effort extra_body; unsupported thinking")

	extra := map[string]any{"thinking": true}
	extraBody := map[string]any{"enable_thinking": true}
	client := &ChatClient{
		HTTPClient:      httpClient,
		BaseURL:         "https://example.com",
		ModelName:       "test-model",
		ReasoningEffort: "low",
		Extra:           extra,
		ExtraBody:       extraBody,
		MaxRetries:      1,
	}

	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	if _, err := client.Invoke(context.Background(), req); err == nil {
		t.Fatalf("expected error from server")
	}

	if client.ReasoningEffort != "low" {
		t.Fatalf("expected reasoning_effort unchanged, got %q", client.ReasoningEffort)
	}
	if _, ok := client.Extra["thinking"]; !ok {
		t.Fatalf("expected extra thinking key preserved")
	}
	if _, ok := client.ExtraBody["enable_thinking"]; !ok {
		t.Fatalf("expected extra_body enable_thinking preserved")
	}
}

func TestResponsesClientDowngradeDoesNotMutateConfig(t *testing.T) {
	t.Parallel()
	httpClient := newErrorClient(http.StatusBadRequest, "unknown field reasoning_effort extra_body; unsupported thinking; MissingParameter input.content")

	extra := map[string]any{"enable_thinking": true}
	extraBody := map[string]any{"thinking": true}
	client := &ResponsesClient{
		HTTPClient:      httpClient,
		BaseURL:         "https://example.com",
		ModelName:       "test-model",
		ReasoningEffort: "medium",
		Extra:           extra,
		ExtraBody:       extraBody,
		MaxRetries:      1,
	}

	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	if _, err := client.Invoke(context.Background(), req); err == nil {
		t.Fatalf("expected error from server")
	}

	if client.ReasoningEffort != "medium" {
		t.Fatalf("expected reasoning_effort unchanged, got %q", client.ReasoningEffort)
	}
	if client.ForceStringInput {
		t.Fatalf("expected force_string_input unchanged")
	}
	if _, ok := client.Extra["enable_thinking"]; !ok {
		t.Fatalf("expected extra enable_thinking preserved")
	}
	if _, ok := client.ExtraBody["thinking"]; !ok {
		t.Fatalf("expected extra_body thinking preserved")
	}
}

func TestMakeStrictSchemaDoesNotMutateTypeBackingArray(t *testing.T) {
	arr := make([]any, 1, 4)
	arr[0] = "string"
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type": arr,
			},
		},
		"required": []any{},
	}

	strict := makeStrictSchema(schema)
	props, ok := strict["properties"].(map[string]any)
	if !ok {
		t.Fatalf("strict schema missing properties")
	}
	nameProp, ok := props["name"].(map[string]any)
	if !ok {
		t.Fatalf("strict schema missing name property")
	}
	strictTypes, ok := nameProp["type"].([]any)
	if !ok {
		t.Fatalf("strict schema missing type array")
	}
	if len(strictTypes) != 2 || strictTypes[0] != "string" || strictTypes[1] != "null" {
		t.Fatalf("unexpected strict type array: %#v", strictTypes)
	}

	origProps, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("original schema missing properties")
	}
	origNameProp, ok := origProps["name"].(map[string]any)
	if !ok {
		t.Fatalf("original schema missing name property")
	}
	origTypes, ok := origNameProp["type"].([]any)
	if !ok {
		t.Fatalf("original schema type changed type")
	}
	if len(origTypes) != 1 || origTypes[0] != "string" {
		t.Fatalf("original schema mutated: %#v", origTypes)
	}
	if cap(origTypes) > len(origTypes) {
		expanded := origTypes[:len(origTypes)+1]
		if s, ok := expanded[1].(string); ok && s == "null" {
			t.Fatalf("original schema backing array mutated: %#v", expanded)
		}
	}
}

func TestMakeStrictSchemaRecursesArrayObjectItems(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"questions": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"header": map[string]any{"type": "string"},
						"options": map[string]any{
							"type": "array",
							"items": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"label":       map[string]any{"type": "string"},
									"description": map[string]any{"type": "string"},
								},
								"required": []any{"label"},
							},
						},
					},
					"required": []any{"header", "options"},
				},
			},
		},
		"required": []any{"questions"},
	}

	strict := makeStrictSchema(schema)
	props := strict["properties"].(map[string]any)
	questions := props["questions"].(map[string]any)
	questionItems := questions["items"].(map[string]any)
	if questionItems["additionalProperties"] != false {
		t.Fatalf("question item additionalProperties = %#v, want false", questionItems["additionalProperties"])
	}
	questionProps := questionItems["properties"].(map[string]any)
	options := questionProps["options"].(map[string]any)
	optionItems := options["items"].(map[string]any)
	if optionItems["additionalProperties"] != false {
		t.Fatalf("option item additionalProperties = %#v, want false", optionItems["additionalProperties"])
	}
	if !requiredContainsAll(t, optionItems["required"], "label", "description") {
		t.Fatalf("option item required = %#v, want label and description", optionItems["required"])
	}
	optionProps := optionItems["properties"].(map[string]any)
	description := optionProps["description"].(map[string]any)
	if !typeAllowsNull(description["type"]) {
		t.Fatalf("optional nested array item property was not nullable: %#v", description["type"])
	}

	origProps := schema["properties"].(map[string]any)
	origQuestions := origProps["questions"].(map[string]any)
	origQuestionItems := origQuestions["items"].(map[string]any)
	origQuestionProps := origQuestionItems["properties"].(map[string]any)
	origOptions := origQuestionProps["options"].(map[string]any)
	origOptionItems := origOptions["items"].(map[string]any)
	if requiredContainsAll(t, origOptionItems["required"], "description") {
		t.Fatalf("original nested item schema was mutated: %#v", origOptionItems["required"])
	}
}

func TestMakeStrictSchemaAddsTypeForFreeformMapValues(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tool_calls": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"tool": map[string]any{"type": "string"},
						"parameters": map[string]any{
							"type":                 "object",
							"additionalProperties": map[string]any{},
						},
					},
					"required": []any{"tool", "parameters"},
				},
			},
		},
		"required": []any{"tool_calls"},
	}

	strict := makeStrictSchema(schema)
	props := strict["properties"].(map[string]any)
	toolCalls := props["tool_calls"].(map[string]any)
	callItem := toolCalls["items"].(map[string]any)
	callProps := callItem["properties"].(map[string]any)
	parameters := callProps["parameters"].(map[string]any)
	additional, ok := parameters["additionalProperties"].(map[string]any)
	if !ok {
		t.Fatalf("additionalProperties = %#v, want schema", parameters["additionalProperties"])
	}
	if !typeContains(additional["type"], "object") || !typeContains(additional["type"], "array") || !typeContains(additional["type"], "null") {
		t.Fatalf("additionalProperties type = %#v, want any JSON type set", additional["type"])
	}
	if additional["additionalProperties"] != false {
		t.Fatalf("additionalProperties nested object guard = %#v, want false", additional["additionalProperties"])
	}
	if _, ok := additional["items"].(map[string]any); !ok {
		t.Fatalf("additionalProperties array items schema = %#v, want schema", additional["items"])
	}

	origProps := schema["properties"].(map[string]any)
	origToolCalls := origProps["tool_calls"].(map[string]any)
	origCallItem := origToolCalls["items"].(map[string]any)
	origCallProps := origCallItem["properties"].(map[string]any)
	origParameters := origCallProps["parameters"].(map[string]any)
	origAdditional := origParameters["additionalProperties"].(map[string]any)
	if len(origAdditional) != 0 {
		t.Fatalf("original additionalProperties was mutated: %#v", origAdditional)
	}
}

func requiredContainsAll(t *testing.T, raw any, names ...string) bool {
	t.Helper()
	required, ok := raw.([]any)
	if !ok {
		return false
	}
	set := map[string]bool{}
	for _, item := range required {
		name, ok := item.(string)
		if !ok {
			return false
		}
		set[name] = true
	}
	for _, name := range names {
		if !set[name] {
			return false
		}
	}
	return true
}

func typeAllowsNull(raw any) bool {
	return typeContains(raw, "null")
}

func typeContains(raw any, want string) bool {
	types, ok := raw.([]any)
	if !ok {
		return false
	}
	for _, item := range types {
		if item == want {
			return true
		}
	}
	return false
}

func TestParseUsageDoesNotDoubleCountReasoningTokens(t *testing.T) {
	usage := parseUsage(map[string]any{
		"prompt_tokens":     120.0,
		"completion_tokens": 80.0,
		"total_tokens":      200.0,
		"completion_tokens_details": map[string]any{
			"reasoning_tokens": 50.0,
		},
		"prompt_tokens_details": map[string]any{
			"cached_tokens": 12.0,
		},
	})
	if usage == nil {
		t.Fatalf("expected usage")
	}
	if usage.CompletionTokens != 80 {
		t.Fatalf("expected completion tokens 80, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 200 {
		t.Fatalf("expected total tokens 200, got %d", usage.TotalTokens)
	}
	if usage.PromptCachedTokens == nil || *usage.PromptCachedTokens != 12 {
		t.Fatalf("expected cached prompt tokens 12, got %v", usage.PromptCachedTokens)
	}
	if usage.PromptTokens != 120 || !usage.PromptTokensValid || usage.PromptTokensSource != llm.PromptTokensSourceProvider || usage.PromptTokensSemantics != llm.PromptTokensSemanticsTotalInputV1 {
		t.Fatalf("unexpected normalized prompt usage: %#v", usage)
	}
}

func TestParseUsageInfersCompletionTokensFromTotalWithoutReasoningDoubleCount(t *testing.T) {
	usage := parseUsage(map[string]any{
		"prompt_tokens": 120.0,
		"total_tokens":  200.0,
		"completion_tokens_details": map[string]any{
			"reasoning_tokens": 50.0,
		},
	})
	if usage == nil {
		t.Fatalf("expected usage")
	}
	if usage.CompletionTokens != 80 {
		t.Fatalf("expected inferred completion tokens 80, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 200 {
		t.Fatalf("expected total tokens 200, got %d", usage.TotalTokens)
	}
}

func TestParseUsageInfersTotalFromPromptAndCompletion(t *testing.T) {
	usage := parseUsage(map[string]any{
		"prompt_tokens":     12.0,
		"completion_tokens": 34.0,
		"completion_tokens_details": map[string]any{
			"reasoning_tokens": 8.0,
		},
	})
	if usage == nil {
		t.Fatalf("expected usage")
	}
	if usage.TotalTokens != 46 {
		t.Fatalf("expected inferred total tokens 46, got %d", usage.TotalTokens)
	}
}

func TestOpenAIEndpointEnterpriseVersionDetection(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		baseURL string
		suffix  string
		want    string
	}{
		{
			name:    "adds v1 for enterprise proxy path without version",
			baseURL: "https://proxy.example.com/api/openai",
			suffix:  "chat/completions",
			want:    "https://proxy.example.com/api/openai/v1/chat/completions",
		},
		{
			name:    "adds v1 when api segment is not numeric version",
			baseURL: "https://host/api/v2beta",
			suffix:  "chat/completions",
			want:    "https://host/api/v2beta/v1/chat/completions",
		},
		{
			name:    "keeps explicit numeric enterprise version",
			baseURL: "https://host/api/v3",
			suffix:  "chat/completions",
			want:    "https://host/api/v3/chat/completions",
		},
		{
			name:    "keeps v1 suffix without duplication",
			baseURL: "https://api.openai.com/v1",
			suffix:  "chat/completions",
			want:    "https://api.openai.com/v1/chat/completions",
		},
		{
			name:    "strips wrapping quotes before v1 detection",
			baseURL: `"http://69.63.215.40:24634/v1"`,
			suffix:  "responses",
			want:    "http://69.63.215.40:24634/v1/responses",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := openAIEndpoint(tt.baseURL, tt.suffix); got != tt.want {
				t.Fatalf("openAIEndpoint(%q, %q) = %q, want %q", tt.baseURL, tt.suffix, got, tt.want)
			}
		})
	}
}

func TestConsumeSSELargeDataLine(t *testing.T) {
	payload := strings.Repeat("a", 5*1024*1024)
	input := "data: " + payload + "\n\n"
	var got string
	if err := consumeSSE(strings.NewReader(input), func(data string) error {
		got = data
		return nil
	}); err != nil {
		t.Fatalf("consumeSSE: %v", err)
	}
	if got != payload {
		t.Fatalf("unexpected payload length: got %d want %d", len(got), len(payload))
	}
}

func TestConsumeSSESkipsFlushWhenNoDataLines(t *testing.T) {
	t.Parallel()

	input := strings.Join([]string{
		"",
		"",
		"data: payload",
		"",
		"",
	}, "\n")

	calls := 0
	if err := consumeSSE(strings.NewReader(input), func(data string) error {
		calls++
		if data != "payload" {
			t.Fatalf("unexpected payload: %q", data)
		}
		return nil
	}); err != nil {
		t.Fatalf("consumeSSE: %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected 1 callback invocation, got %d", calls)
	}
}

func TestConsumeSSEWithBodyCloseClosesOnPanic(t *testing.T) {
	body := &closeTrackingReadCloser{Reader: strings.NewReader("data: x\n\n")}
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic from callback")
		}
		if !body.closed {
			t.Fatalf("expected response body to be closed on panic")
		}
	}()

	_ = consumeSSEWithBodyClose(body, func(string) error {
		panic("boom")
	})
}

func TestChatStreamDowngradesUnsupportedStreamOptions(t *testing.T) {
	t.Parallel()

	calls := 0
	requestBodies := make([]string, 0, 2)
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		_ = r.Body.Close()
		body := string(bodyBytes)
		requestBodies = append(requestBodies, body)
		calls++
		if calls == 1 {
			return &http.Response{
				StatusCode: http.StatusBadRequest,
				Status:     "400 Bad Request",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"error":{"message":"unknown field stream_options"}}`)),
				Request:    r,
			}, nil
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     make(http.Header),
			Body: io.NopCloser(strings.NewReader(`data: {"choices":[{"delta":{"content":"ok"}}]}

data: [DONE]

`)),
			Request: r,
		}, nil
	})}

	client := &ChatClient{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		MaxRetries: 2,
	}

	events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var textOut string
	for ev := range events {
		switch e := ev.(type) {
		case llm.StreamTextDeltaEvent:
			textOut += e.Delta
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.Err)
		}
	}

	if textOut != "ok" {
		t.Fatalf("expected downgraded stream output ok, got %q", textOut)
	}
	if calls != 2 {
		t.Fatalf("expected 2 stream attempts, got %d", calls)
	}
	if len(requestBodies) != 2 {
		t.Fatalf("expected 2 captured request bodies, got %d", len(requestBodies))
	}
	if !strings.Contains(requestBodies[0], "stream_options") {
		t.Fatalf("expected first request to include stream_options, got %s", requestBodies[0])
	}
	if strings.Contains(requestBodies[1], "stream_options") {
		t.Fatalf("expected second request to omit stream_options, got %s", requestBodies[1])
	}
}

func TestChatStreamDecodeError(t *testing.T) {
	t.Parallel()
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			body := "data: {not json}\n\n"
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
				Request:    r,
			}, nil
		}),
	}
	client := &ChatClient{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		MaxRetries: 1,
	}

	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	events, err := client.InvokeStream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var errEvent llm.StreamErrorEvent
	var errSeen bool
	for ev := range events {
		if e, ok := ev.(llm.StreamErrorEvent); ok {
			errSeen = true
			errEvent = e
		}
	}

	if !errSeen {
		t.Fatalf("expected error event")
	}
	if errEvent.Err == nil {
		t.Fatalf("expected decode error")
	}
	errMsg := errEvent.Err.Error()
	if !strings.Contains(errMsg, "decode error") {
		t.Fatalf("expected decode error message, got %v", errEvent.Err)
	}
	if !strings.Contains(errMsg, "provider=openai") {
		t.Fatalf("expected provider context, got %q", errMsg)
	}
	if !strings.Contains(errMsg, "status=200") {
		t.Fatalf("expected status context, got %q", errMsg)
	}
	if !strings.Contains(errMsg, "model=\"test-model\"") {
		t.Fatalf("expected model context, got %q", errMsg)
	}
	if !strings.Contains(errMsg, "https://example.com/v1/chat/completions") {
		t.Fatalf("expected endpoint context, got %q", errMsg)
	}
}

func TestChatStreamHandlesPrematureSSEBoundary(t *testing.T) {
	t.Parallel()

	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"ok"}}`,
		"",
		`data: ]}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
				Request:    r,
			}, nil
		}),
	}

	client := &ChatClient{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		MaxRetries: 1,
	}

	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	events, err := client.InvokeStream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var text string
	for ev := range events {
		switch e := ev.(type) {
		case llm.StreamTextDeltaEvent:
			text += e.Delta
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.Err)
		}
	}

	if text != "ok" {
		t.Fatalf("expected recovered text delta, got %q", text)
	}
}

func TestExponentialBackoffDelayCapsWithoutOverflow(t *testing.T) {
	t.Parallel()

	d := exponentialBackoffDelay(63, time.Second, 60*time.Second)
	if d != 60*time.Second {
		t.Fatalf("expected capped delay, got %v", d)
	}
}

func TestBuildRequestParallelToolCallsOmittedByDefault(t *testing.T) {
	t.Parallel()

	c := &ChatClient{ModelName: "test-model"}
	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
		Tools: []llm.ToolDefinition{{
			Name:       "echo",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
		}},
	}
	built, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	if built.ParallelToolCalls != nil {
		t.Fatalf("expected parallel_tool_calls to be omitted by default, got %v", *built.ParallelToolCalls)
	}
}

func TestBuildRequestParallelToolCallsIncludedWhenEnabled(t *testing.T) {
	t.Parallel()

	c := &ChatClient{ModelName: "test-model", ParallelToolCalls: true}
	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
		Tools: []llm.ToolDefinition{{
			Name:       "echo",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
		}},
	}
	built, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	if built.ParallelToolCalls == nil || !*built.ParallelToolCalls {
		t.Fatalf("expected parallel_tool_calls=true when explicitly enabled, got %#v", built.ParallelToolCalls)
	}
}

func TestChatBuildRequestRejectsInvalidToolHistory(t *testing.T) {
	t.Parallel()

	c := &ChatClient{ModelName: "test-model"}
	_, err := c.buildRequest(llm.InvokeRequest{Messages: []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewToolMessage("call-1", "read", llm.TextContent("result"), false),
	}})
	if err == nil || !strings.Contains(err.Error(), "invalid tool history") {
		t.Fatalf("expected invalid tool history error, got %v", err)
	}
}

func TestResponsesBuildRequestRejectsIncompleteToolHistory(t *testing.T) {
	t.Parallel()

	c := &ResponsesClient{ModelName: "test-model"}
	_, err := c.buildRequest(llm.InvokeRequest{Messages: []llm.Message{
		llm.NewAssistantMessage("using tool", []llm.ToolCall{{
			ID:   "call-1",
			Type: "function",
			Function: llm.FunctionCall{
				Name:      "read",
				Arguments: `{}`,
			},
		}}),
		llm.NewUserMessage("next"),
	}})
	if err == nil || !strings.Contains(err.Error(), "invalid tool history") {
		t.Fatalf("expected invalid tool history error, got %v", err)
	}
}

func TestResponsesBuildRequestAssistantContentUsesOutputText(t *testing.T) {
	t.Parallel()

	c := &ResponsesClient{ModelName: "test-model"}
	req := llm.InvokeRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: llm.TextContent("hello")},
			{Role: llm.RoleAssistant, Content: llm.TextContent("hi")},
		},
	}
	built, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	msgs, ok := built.Input.([]responsesMessage)
	if !ok {
		t.Fatalf("expected []responsesMessage input, got %T", built.Input)
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 input messages, got %d", len(msgs))
	}
	if msgs[1].Role != "assistant" {
		t.Fatalf("expected assistant role, got %q", msgs[1].Role)
	}
	parts, ok := msgs[1].Content.([]responsesContentPart)
	if !ok {
		t.Fatalf("expected assistant content parts, got %T", msgs[1].Content)
	}
	if len(parts) != 1 {
		t.Fatalf("expected 1 assistant content part, got %d", len(parts))
	}
	if parts[0].Type != "output_text" {
		t.Fatalf("expected assistant part type output_text, got %q", parts[0].Type)
	}
}

func TestResponsesBuildRequestItemsAssistantContentUsesOutputText(t *testing.T) {
	t.Parallel()

	useItems := true
	useInstructions := false
	c := &ResponsesClient{ModelName: "test-model"}
	req := llm.InvokeRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: llm.TextContent("hello")},
			{Role: llm.RoleAssistant, Content: llm.TextContent("hi")},
		},
		Responses: &llm.ResponsesOptions{
			UseResponseItems: &useItems,
			UseInstructions:  &useInstructions,
		},
	}
	built, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	items, ok := built.Input.([]responsesInputItem)
	if !ok {
		t.Fatalf("expected []responsesInputItem input, got %T", built.Input)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 input items, got %d", len(items))
	}
	if items[1].Type != "message" || items[1].Role != "assistant" {
		t.Fatalf("expected assistant message item, got type=%q role=%q", items[1].Type, items[1].Role)
	}
	parts, ok := items[1].Content.([]responsesContentPart)
	if !ok {
		t.Fatalf("expected assistant content parts, got %T", items[1].Content)
	}
	if len(parts) != 1 {
		t.Fatalf("expected 1 assistant content part, got %d", len(parts))
	}
	if parts[0].Type != "output_text" {
		t.Fatalf("expected assistant part type output_text, got %q", parts[0].Type)
	}
}

func TestResponsesBuildRequestItemsToolErrorOutputIsString(t *testing.T) {
	t.Parallel()

	useItems := true
	useInstructions := false
	c := &ResponsesClient{ModelName: "test-model"}
	req := llm.InvokeRequest{
		Messages: []llm.Message{
			llm.NewUserMessage("hello"),
			llm.NewAssistantMessage("using tool", []llm.ToolCall{{
				ID:   "call-1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "bash",
					Arguments: `{"command":"false"}`,
				},
			}}),
			llm.NewToolMessage("call-1", "bash", llm.TextContent(`{"title":"failed","output":"boom"}`), true),
		},
		Responses: &llm.ResponsesOptions{
			UseResponseItems: &useItems,
			UseInstructions:  &useInstructions,
		},
	}
	built, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	items, ok := built.Input.([]responsesInputItem)
	if !ok {
		t.Fatalf("expected []responsesInputItem input, got %T", built.Input)
	}
	var found bool
	for _, item := range items {
		if item.Type != "function_call_output" {
			continue
		}
		found = true
		out, ok := item.Output.(string)
		if !ok {
			t.Fatalf("function_call_output output type = %T, want string", item.Output)
		}
		if !strings.Contains(out, "(error)") || !strings.Contains(out, "boom") {
			t.Fatalf("unexpected tool error output: %q", out)
		}
	}
	if !found {
		t.Fatalf("missing function_call_output item: %#v", items)
	}
}

func TestIsRetryableNetErr(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil", err: nil, want: false},
		{name: "timeout interface", err: timeoutErr{}, want: true},
		{name: "deadline exceeded", err: context.DeadlineExceeded, want: true},
		{name: "connection reset", err: fmt.Errorf("connection reset by peer"), want: true},
		{name: "other error", err: fmt.Errorf("boom"), want: false},
	}

	for _, tt := range tests {
		if got := isRetryableNetErr(tt.err); got != tt.want {
			t.Fatalf("%s: expected %v, got %v", tt.name, tt.want, got)
		}
	}
}

func TestOpenAIDefaultRetryableStatus(t *testing.T) {
	t.Parallel()

	chat := &ChatClient{}
	responses := &ResponsesClient{}
	tests := []struct {
		code int
		want bool
	}{
		{code: 400, want: false},
		{code: 401, want: false},
		{code: 403, want: false},
		{code: 404, want: false},
		{code: 408, want: true},
		{code: 409, want: true},
		{code: 422, want: false},
		{code: 425, want: true},
		{code: 429, want: true},
		{code: 500, want: true},
		{code: 529, want: true},
	}
	for _, tt := range tests {
		if got := chat.isRetryableStatus(tt.code); got != tt.want {
			t.Fatalf("chat status %d retryable=%v, want %v", tt.code, got, tt.want)
		}
		if got := responses.isRetryableStatus(tt.code); got != tt.want {
			t.Fatalf("responses status %d retryable=%v, want %v", tt.code, got, tt.want)
		}
	}
}

func TestOpenAIChatRejectsOversizedResponseBody(t *testing.T) {
	prevLimit := maxProviderResponseBytes
	maxProviderResponseBytes = 64
	t.Cleanup(func() { maxProviderResponseBytes = prevLimit })

	body := strings.Repeat("x", int(maxProviderResponseBytes)+8)
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
				Request:    r,
			}, nil
		}),
	}

	client := &ChatClient{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		MaxRetries: 1,
	}
	_, err := client.Invoke(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err == nil {
		t.Fatal("expected oversized response error")
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected provider error, got %T", err)
	}
	if providerErr.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want %d", providerErr.StatusCode, http.StatusOK)
	}
	assertOpenAIResponseSizeDiagnostic(t, providerErr.Message, maxProviderResponseBytes)
}

func TestOpenAIResponsesRejectsOversizedResponseBody(t *testing.T) {
	prevLimit := maxProviderResponseBytes
	maxProviderResponseBytes = 64
	t.Cleanup(func() { maxProviderResponseBytes = prevLimit })

	body := strings.Repeat("x", int(maxProviderResponseBytes)+8)
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
				Request:    r,
			}, nil
		}),
	}

	client := &ResponsesClient{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		MaxRetries: 1,
	}
	_, err := client.Invoke(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err == nil {
		t.Fatal("expected oversized response error")
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected provider error, got %T", err)
	}
	if providerErr.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want %d", providerErr.StatusCode, http.StatusOK)
	}
	assertOpenAIResponseSizeDiagnostic(t, providerErr.Message, maxProviderResponseBytes)
}

func TestOpenAIChatStreamIncludesResponseSizeDiagnostic(t *testing.T) {
	prevLimit := maxProviderResponseBytes
	maxProviderResponseBytes = 64
	t.Cleanup(func() { maxProviderResponseBytes = prevLimit })

	body := strings.Repeat("x", int(maxProviderResponseBytes)+8)
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusBadGateway,
				Status:     "502 Bad Gateway",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
				Request:    r,
			}, nil
		}),
	}

	client := &ChatClient{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		MaxRetries: 1,
	}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	var streamErr error
	for ev := range stream {
		if errEv, ok := ev.(llm.StreamErrorEvent); ok {
			streamErr = errEv.AsError()
		}
	}
	if streamErr == nil {
		t.Fatal("expected stream error")
	}
	var providerErr *llm.ProviderError
	if !errors.As(streamErr, &providerErr) {
		t.Fatalf("expected provider error, got %T", streamErr)
	}
	if providerErr.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d", providerErr.StatusCode, http.StatusBadGateway)
	}
	assertOpenAIResponseSizeDiagnostic(t, providerErr.Message, maxProviderResponseBytes)
}

func TestOpenAIResponsesStreamIncludesResponseSizeDiagnostic(t *testing.T) {
	prevLimit := maxProviderResponseBytes
	maxProviderResponseBytes = 64
	t.Cleanup(func() { maxProviderResponseBytes = prevLimit })

	body := strings.Repeat("x", int(maxProviderResponseBytes)+8)
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusServiceUnavailable,
				Status:     "503 Service Unavailable",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
				Request:    r,
			}, nil
		}),
	}

	client := &ResponsesClient{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		MaxRetries: 1,
	}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	var streamErr error
	for ev := range stream {
		if errEv, ok := ev.(llm.StreamErrorEvent); ok {
			streamErr = errEv.AsError()
		}
	}
	if streamErr == nil {
		t.Fatal("expected stream error")
	}
	var providerErr *llm.ProviderError
	if !errors.As(streamErr, &providerErr) {
		t.Fatalf("expected provider error, got %T", streamErr)
	}
	if providerErr.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", providerErr.StatusCode, http.StatusServiceUnavailable)
	}
	assertOpenAIResponseSizeDiagnostic(t, providerErr.Message, maxProviderResponseBytes)
}

func assertOpenAIResponseSizeDiagnostic(t *testing.T, msg string, limit int64) {
	t.Helper()
	lower := strings.ToLower(strings.TrimSpace(msg))
	if !strings.Contains(lower, "response body too large") {
		t.Fatalf("expected oversized response diagnostic, got %q", msg)
	}
	if !strings.Contains(lower, "read=") {
		t.Fatalf("expected read-bytes diagnostic, got %q", msg)
	}
	if !strings.Contains(lower, fmt.Sprintf("limit=%d", limit)) {
		t.Fatalf("expected limit=%d in diagnostic, got %q", limit, msg)
	}
	if !strings.Contains(lower, "request a smaller response") {
		t.Fatalf("expected actionable hint, got %q", msg)
	}
}

func TestParseChatCompletionCapturesResponseID(t *testing.T) {
	comp, err := parseChatCompletion([]byte(`{"id":"chatcmpl_123","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`))
	if err != nil {
		t.Fatalf("parseChatCompletion: %v", err)
	}
	if comp.ResponseID != "chatcmpl_123" {
		t.Fatalf("response id = %q, want %q", comp.ResponseID, "chatcmpl_123")
	}
}

func TestChatStreamEmitsResponseID(t *testing.T) {
	t.Parallel()
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		body := "data: {\"id\":\"chatcmpl_stream_123\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(body)),
			Request:    r,
		}, nil
	})}
	client := &ChatClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}})
	if err != nil {
		t.Fatalf("InvokeStream: %v", err)
	}
	responseID := ""
	for ev := range events {
		if e, ok := ev.(llm.StreamResponseEvent); ok {
			responseID = e.ResponseID
		}
	}
	if responseID != "chatcmpl_stream_123" {
		t.Fatalf("response id = %q, want %q", responseID, "chatcmpl_stream_123")
	}
}

func TestParseChatCompletionAcceptsArrayTextContent(t *testing.T) {
	t.Parallel()

	comp, err := parseChatCompletion([]byte(`{"id":"chatcmpl_arr","choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatalf("parseChatCompletion: %v", err)
	}
	if comp.PlainText() != "ok" {
		t.Fatalf("content = %q, want %q", comp.PlainText(), "ok")
	}
	if comp.ResponseID != "chatcmpl_arr" {
		t.Fatalf("response id = %q, want %q", comp.ResponseID, "chatcmpl_arr")
	}
}

func TestParseChatCompletionHandlesRefusalWhenContentNull(t *testing.T) {
	t.Parallel()

	comp, err := parseChatCompletion([]byte(`{"id":"chatcmpl_refusal","choices":[{"message":{"role":"assistant","content":null,"refusal":"I can’t help with that."},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatalf("parseChatCompletion: %v", err)
	}
	if comp.PlainText() != "I can’t help with that." {
		t.Fatalf("content = %q, want refusal text", comp.PlainText())
	}
	if comp.ResponseID != "chatcmpl_refusal" {
		t.Fatalf("response id = %q, want %q", comp.ResponseID, "chatcmpl_refusal")
	}
}

func TestParseChatCompletionAcceptsLegacyFunctionCall(t *testing.T) {
	t.Parallel()

	comp, err := parseChatCompletion([]byte(`{"id":"chatcmpl_legacy","choices":[{"message":{"role":"assistant","content":null,"function_call":{"name":"done","arguments":"{\"message\":\"legacy ok\"}"}},"finish_reason":"function_call"}]}`))
	if err != nil {
		t.Fatalf("parseChatCompletion: %v", err)
	}
	if len(comp.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(comp.ToolCalls))
	}
	if comp.ToolCalls[0].Function.Name != "done" {
		t.Fatalf("tool call name = %q, want done", comp.ToolCalls[0].Function.Name)
	}
	if comp.ToolCalls[0].Function.Arguments != `{"message":"legacy ok"}` {
		t.Fatalf("tool call args = %q", comp.ToolCalls[0].Function.Arguments)
	}
	if !strings.HasPrefix(comp.ToolCalls[0].ID, "call_") {
		t.Fatalf("tool call id = %q, want synthetic call_ prefix", comp.ToolCalls[0].ID)
	}
	if comp.StopReason != "tool_calls" {
		t.Fatalf("stop reason = %q, want %q", comp.StopReason, "tool_calls")
	}
}

func TestChatStreamParsesLegacyFunctionCallDelta(t *testing.T) {
	t.Parallel()

	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		body := `data: {"id":"chatcmpl_stream_legacy","choices":[{"delta":{"function_call":{"name":"done","arguments":"{\"message\":\"legacy stream ok\"}"}},"finish_reason":"function_call"}]}` + "\n\n" + `data: [DONE]` + "\n\n"
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(body)),
			Request:    r,
		}, nil
	})}
	client := &ChatClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}})
	if err != nil {
		t.Fatalf("InvokeStream: %v", err)
	}
	toolName := ""
	toolArgs := ""
	doneReason := ""
	for ev := range events {
		switch e := ev.(type) {
		case llm.StreamToolCallDeltaEvent:
			if strings.TrimSpace(e.NameDelta) != "" {
				toolName = e.NameDelta
			}
			if strings.TrimSpace(e.ArgumentsDelta) != "" {
				toolArgs = e.ArgumentsDelta
			}
		case llm.StreamDoneEvent:
			doneReason = e.StopReason
		}
	}
	if toolName != "done" {
		t.Fatalf("tool name = %q, want done", toolName)
	}
	if toolArgs != `{"message":"legacy stream ok"}` {
		t.Fatalf("tool args = %q", toolArgs)
	}
	if doneReason != "tool_calls" {
		t.Fatalf("done reason = %q, want %q", doneReason, "tool_calls")
	}
}

func TestBuildRequestIncludesImageURLDetail(t *testing.T) {
	client := &ChatClient{ModelName: "test-model"}
	payload, err := client.buildRequest(llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://example.com/image.png", Detail: "high"}}}}}}})
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	parts, ok := payload.Messages[0].Content.([]map[string]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("content = %#v, want single image part", payload.Messages[0].Content)
	}
	image, ok := parts[0]["image_url"].(map[string]any)
	if !ok {
		t.Fatalf("image payload = %#v", parts[0]["image_url"])
	}
	if image["url"] != "https://example.com/image.png" {
		t.Fatalf("image url = %#v", image["url"])
	}
	if image["detail"] != "high" {
		t.Fatalf("image detail = %#v, want high", image["detail"])
	}
}

func TestBuildRequestFallsBackForUnsupportedContentBlocks(t *testing.T) {
	client := &ChatClient{ModelName: "test-model"}
	payload, err := client.buildRequest(llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "document", Source: &llm.DocSrc{MediaType: "application/pdf"}}}}}}})
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	parts, ok := payload.Messages[0].Content.([]map[string]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("content = %#v, want fallback text part", payload.Messages[0].Content)
	}
	if parts[0]["type"] != "text" {
		t.Fatalf("part type = %#v, want text", parts[0]["type"])
	}
	textValue, _ := parts[0]["text"].(string)
	if !strings.Contains(textValue, "document") {
		t.Fatalf("fallback text = %q, want document marker", textValue)
	}
}

func TestChatInvokeAppliesAllCompatDowngradesBeforeSingleRetry(t *testing.T) {
	attempt := 0
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		attempt++
		body, _ := io.ReadAll(r.Body)
		bodyText := string(body)
		status := http.StatusOK
		respBody := `{"id":"chatcmpl_ok","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
		if attempt == 1 {
			status = http.StatusBadRequest
			respBody = `unknown field reasoning_effort extra_body; unsupported thinking`
		} else if strings.Contains(bodyText, "reasoning_effort") || strings.Contains(bodyText, "extra_body") || strings.Contains(bodyText, "enable_thinking") || strings.Contains(bodyText, `"thinking"`) {
			status = http.StatusBadRequest
			respBody = `still unsupported`
		}
		return &http.Response{StatusCode: status, Status: fmt.Sprintf("%d %s", status, http.StatusText(status)), Header: make(http.Header), Body: io.NopCloser(strings.NewReader(respBody)), Request: r}, nil
	})}
	client := &ChatClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 2, ReasoningEffort: "low", Extra: map[string]any{"thinking": true}, ExtraBody: map[string]any{"enable_thinking": true}}
	comp, err := client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if comp.PlainText() != "ok" {
		t.Fatalf("content = %q, want ok", comp.PlainText())
	}
	if attempt != 2 {
		t.Fatalf("attempts = %d, want 2", attempt)
	}
	if len(comp.Diagnostics) != 3 {
		t.Fatalf("diagnostics = %#v, want three compatibility downgrade warnings", comp.Diagnostics)
	}
	for _, diag := range comp.Diagnostics {
		if diag.Kind != "provider_compatibility_downgrade" || !strings.Contains(diag.Message, "retrying without") {
			t.Fatalf("unexpected diagnostic: %#v", diag)
		}
	}
}

func TestResponsesInvokeIncludesCompatibilityDowngradeDiagnostics(t *testing.T) {
	attempt := 0
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		attempt++
		body, _ := io.ReadAll(r.Body)
		bodyText := string(body)
		status := http.StatusOK
		respBody := `{"id":"resp_ok","status":"completed","output_text":"ok"}`
		if attempt == 1 {
			status = http.StatusBadRequest
			respBody = `unknown field reasoning_effort extra_body; unsupported thinking; MissingParameter input.content`
		} else if strings.Contains(bodyText, "reasoning_effort") || strings.Contains(bodyText, "extra_body") || strings.Contains(bodyText, "enable_thinking") || strings.Contains(bodyText, `"thinking"`) || strings.Contains(bodyText, `"content":[`) {
			status = http.StatusBadRequest
			respBody = `still unsupported`
		}
		return &http.Response{StatusCode: status, Status: fmt.Sprintf("%d %s", status, http.StatusText(status)), Header: make(http.Header), Body: io.NopCloser(strings.NewReader(respBody)), Request: r}, nil
	})}
	client := &ResponsesClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 2, ReasoningEffort: "medium", Extra: map[string]any{"thinking": true}, ExtraBody: map[string]any{"enable_thinking": true}}
	comp, err := client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "text", Text: "hi"}}}}}})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if comp.PlainText() != "ok" {
		t.Fatalf("content = %q, want ok", comp.PlainText())
	}
	if attempt != 2 {
		t.Fatalf("attempts = %d, want 2", attempt)
	}
	if len(comp.Diagnostics) != 5 {
		t.Fatalf("diagnostics = %#v, want five compatibility downgrade warnings", comp.Diagnostics)
	}
	for _, diag := range comp.Diagnostics {
		if diag.Kind != "provider_compatibility_downgrade" || !strings.Contains(diag.Message, "retrying") {
			t.Fatalf("unexpected diagnostic: %#v", diag)
		}
	}
}

func TestChatStreamAppliesAllCompatDowngradesBeforeSingleRetry(t *testing.T) {
	attempt := 0
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		attempt++
		body, _ := io.ReadAll(r.Body)
		bodyText := string(body)
		status := http.StatusOK
		respBody := `data: {"id":"chatcmpl_stream_ok","choices":[{"delta":{"content":"ok"}}]}

data: [DONE]

`
		if attempt == 1 {
			status = http.StatusBadRequest
			respBody = `unknown field reasoning_effort stream_options`
		} else if strings.Contains(bodyText, "reasoning_effort") || strings.Contains(bodyText, "stream_options") {
			status = http.StatusBadRequest
			respBody = `still unsupported`
		}
		return &http.Response{StatusCode: status, Status: fmt.Sprintf("%d %s", status, http.StatusText(status)), Header: make(http.Header), Body: io.NopCloser(strings.NewReader(respBody)), Request: r}, nil
	})}
	client := &ChatClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 2, ReasoningEffort: "low"}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}})
	if err != nil {
		t.Fatalf("InvokeStream: %v", err)
	}
	var textOut string
	for ev := range stream {
		if e, ok := ev.(llm.StreamTextDeltaEvent); ok {
			textOut += e.Delta
		}
	}
	if textOut != "ok" {
		t.Fatalf("stream text = %q, want ok", textOut)
	}
	if attempt != 2 {
		t.Fatalf("attempts = %d, want 2", attempt)
	}
}

func TestResponsesBuildRequestFallsBackForUnsupportedContentBlocks(t *testing.T) {
	t.Parallel()

	c := &ResponsesClient{ModelName: "test-model"}
	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "document", Source: &llm.DocSrc{MediaType: "application/pdf"}}}}}},
	}
	built, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	msgs, ok := built.Input.([]responsesMessage)
	if !ok || len(msgs) != 1 {
		t.Fatalf("input = %#v, want single responsesMessage", built.Input)
	}
	parts, ok := msgs[0].Content.([]responsesContentPart)
	if !ok || len(parts) != 1 {
		t.Fatalf("content = %#v, want one fallback part", msgs[0].Content)
	}
	if parts[0].Type != "input_text" {
		t.Fatalf("part type = %q, want input_text", parts[0].Type)
	}
	if !strings.Contains(parts[0].Text, "document") {
		t.Fatalf("fallback text = %q, want document marker", parts[0].Text)
	}
}

func TestResponsesBuildRequestForceStringInputFallsBackForImagesAndDocuments(t *testing.T) {
	t.Parallel()

	c := &ResponsesClient{ModelName: "test-model", ForceStringInput: true}
	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://example.com/a.png", MediaType: "image/png"}}, {Type: "document", Source: &llm.DocSrc{MediaType: "application/pdf"}}}}}},
	}
	built, err := c.buildRequest(req)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	msgs, ok := built.Input.([]responsesMessage)
	if !ok || len(msgs) != 1 {
		t.Fatalf("input = %#v, want single responsesMessage", built.Input)
	}
	content, ok := msgs[0].Content.(string)
	if !ok {
		t.Fatalf("content type = %T, want string", msgs[0].Content)
	}
	if !strings.Contains(content, "[image") || !strings.Contains(content, "document") {
		t.Fatalf("content = %q, want image/document fallback text", content)
	}
}
