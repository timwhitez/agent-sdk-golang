package anthropic

import (
	"context"
	"encoding/json"
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

func httpResponse(status int, body string, r *http.Request) *http.Response {
	return &http.Response{
		StatusCode: status,
		Status:     fmt.Sprintf("%d %s", status, http.StatusText(status)),
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    r,
	}
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

func TestParseResponse_CacheReadTokensNotDoubleCounted(t *testing.T) {
	data := []byte(`{
		"id": "msg_sync_123",
		"content": [{"type": "text", "text": "hi"}],
		"stop_reason": "end",
		"usage": {
			"input_tokens": 100,
			"output_tokens": 20,
			"cache_read_input_tokens": 30,
			"cache_creation_input_tokens": 5
		}
	}`)

	comp, err := parseResponse(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if comp.Usage == nil {
		t.Fatalf("expected usage data")
	}
	if comp.Usage.PromptTokens != 100 {
		t.Fatalf("expected prompt tokens 100, got %d", comp.Usage.PromptTokens)
	}
	if comp.Usage.TotalTokens != 120 {
		t.Fatalf("expected total tokens 120, got %d", comp.Usage.TotalTokens)
	}
	if comp.Usage.PromptCachedTokens == nil || *comp.Usage.PromptCachedTokens != 30 {
		t.Fatalf("expected cached tokens 30, got %v", comp.Usage.PromptCachedTokens)
	}
	if comp.Usage.PromptCacheCreationTokens == nil || *comp.Usage.PromptCacheCreationTokens != 5 {
		t.Fatalf("expected cache creation tokens 5, got %v", comp.Usage.PromptCacheCreationTokens)
	}
	if comp.ResponseID != "msg_sync_123" {
		t.Fatalf("expected response id msg_sync_123, got %q", comp.ResponseID)
	}
}

func TestInvokeStreamUsageIncludesCacheTokenFields(t *testing.T) {
	t.Parallel()

	body := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"msg_stream_123","usage":{"input_tokens":100,"cache_read_input_tokens":30,"cache_creation_input_tokens":5}}}`,
		"",
		`data: {"type":"message_delta","usage":{"output_tokens":20}}`,
		"",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")

	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return httpResponse(http.StatusOK, body, r), nil
		}),
	}

	client := &Client{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
	}

	events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	var usage *llm.Usage
	var responseID string
	for ev := range events {
		switch e := ev.(type) {
		case llm.StreamUsageEvent:
			u := e.Usage
			usage = &u
		case llm.StreamResponseEvent:
			responseID = e.ResponseID
		}
	}

	if usage == nil {
		t.Fatalf("expected usage event")
	}
	if usage.PromptTokens != 100 {
		t.Fatalf("expected prompt tokens 100, got %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 20 {
		t.Fatalf("expected completion tokens 20, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 120 {
		t.Fatalf("expected total tokens 120, got %d", usage.TotalTokens)
	}
	if usage.PromptCachedTokens == nil || *usage.PromptCachedTokens != 30 {
		t.Fatalf("expected prompt cached tokens 30, got %v", usage.PromptCachedTokens)
	}
	if usage.PromptCacheCreationTokens == nil || *usage.PromptCacheCreationTokens != 5 {
		t.Fatalf("expected prompt cache creation tokens 5, got %v", usage.PromptCacheCreationTokens)
	}
	if responseID != "msg_stream_123" {
		t.Fatalf("expected response id msg_stream_123, got %q", responseID)
	}
}

func TestInvokeStreamResponseIDFallbackFromMessageDelta(t *testing.T) {
	t.Parallel()

	body := strings.Join([]string{
		`data: {"type":"message_delta","id":"msg_stream_fallback","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		"",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")

	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return httpResponse(http.StatusOK, body, r), nil
		}),
	}

	client := &Client{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
	}

	events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	responseID := ""
	for ev := range events {
		switch e := ev.(type) {
		case llm.StreamResponseEvent:
			responseID = e.ResponseID
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.Err)
		}
	}

	if responseID != "msg_stream_fallback" {
		t.Fatalf("expected fallback response id msg_stream_fallback, got %q", responseID)
	}
}

func TestInvokeStreamPreservesWhitespaceThinkingDelta(t *testing.T) {
	t.Parallel()

	body := strings.Join([]string{
		`data: {"type":"content_block_delta","index":0,"delta":{"thinking":"\n\n"}}`,
		"",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")

	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return httpResponse(http.StatusOK, body, r), nil
		}),
	}

	client := &Client{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
	}

	events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	var deltas []string
	for ev := range events {
		switch e := ev.(type) {
		case llm.StreamThinkingDeltaEvent:
			deltas = append(deltas, e.Delta)
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.Err)
		}
	}

	if len(deltas) != 1 || deltas[0] != "\n\n" {
		t.Fatalf("expected whitespace thinking delta preserved, got %#v", deltas)
	}
}

func TestInvokeStreamHandlesPrematureSSEBoundary(t *testing.T) {
	t.Parallel()

	body := strings.Join([]string{
		`data: {"type":"content_block_delta","index":0`,
		"",
		`data: ,"delta":{"text":"ok"}}`,
		"",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")

	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return httpResponse(http.StatusOK, body, r), nil
		}),
	}

	client := &Client{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
	}

	events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
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

func TestInvokeStreamReportsDecodeErrorForNonObjectSSEJSON(t *testing.T) {
	t.Parallel()

	body := strings.Join([]string{
		`data: ["not-an-object"]`,
		"",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")

	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return httpResponse(http.StatusOK, body, r), nil
		}),
	}

	client := &Client{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
	}

	events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	var streamErr error
	var doneSeen bool
	for ev := range events {
		switch e := ev.(type) {
		case llm.StreamErrorEvent:
			streamErr = e.AsError()
		case llm.StreamDoneEvent:
			doneSeen = true
		}
	}

	if streamErr == nil {
		t.Fatalf("expected stream decode error event")
	}
	if !strings.Contains(streamErr.Error(), "failed to decode SSE JSON event") {
		t.Fatalf("expected decode error message, got %v", streamErr)
	}
	if doneSeen {
		t.Fatalf("did not expect done event after decode failure")
	}
}

func TestSerializeMessagesKeepsNonCachedSystemTextWithCachedBlocks(t *testing.T) {
	in := []llm.Message{
		{Role: llm.RoleSystem, Content: llm.TextContent("cached system"), Cache: true},
		{Role: llm.RoleSystem, Content: llm.TextContent("dynamic system")},
		{Role: llm.RoleUser, Content: llm.TextContent("hello")},
	}

	system, msgs, err := serializeMessages(in)
	if err != nil {
		t.Fatalf("serialize messages: %v", err)
	}
	blocks, ok := system.([]contentBlockParam)
	if !ok {
		t.Fatalf("expected structured system blocks, got %T", system)
	}
	if len(blocks) != 2 {
		t.Fatalf("expected 2 system blocks, got %d", len(blocks))
	}
	if blocks[0].Type != "text" || blocks[0].Text != "cached system" || blocks[0].CacheCtrl == nil {
		t.Fatalf("expected cached system block preserved, got %#v", blocks[0])
	}
	if blocks[1].Type != "text" || blocks[1].Text != "dynamic system" || blocks[1].CacheCtrl != nil {
		t.Fatalf("expected non-cached system block preserved, got %#v", blocks[1])
	}
	if len(msgs) != 1 || msgs[0].Role != "user" {
		t.Fatalf("expected user message preserved, got %#v", msgs)
	}
}

func TestClientBetaDowngradeDoesNotMutateConfig(t *testing.T) {
	t.Parallel()
	calls := 0
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			calls++
			if calls == 1 {
				if got := r.Header.Get("anthropic-beta"); got != "custom-beta" {
					t.Fatalf("expected beta header custom-beta, got %q", got)
				}
				return httpResponse(http.StatusBadRequest, "unsupported beta", r), nil
			}
			if got := r.Header.Get("anthropic-beta"); got != "prompt-caching-2024-07-31" {
				t.Fatalf("expected downgraded beta header, got %q", got)
			}
			body := `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end","usage":{"input_tokens":1,"output_tokens":1}}`
			return httpResponse(http.StatusOK, body, r), nil
		}),
	}

	client := &Client{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		Beta:       []string{"custom-beta"},
		MaxRetries: 2,
	}
	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	if _, err := client.Invoke(context.Background(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 2 {
		t.Fatalf("expected 2 attempts, got %d", calls)
	}
	if got := strings.Join(client.Beta, ","); got != "custom-beta" {
		t.Fatalf("expected beta unchanged, got %q", got)
	}
}

func TestClientBetaDowngradeDoesNotMutateConfigStream(t *testing.T) {
	t.Parallel()
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return httpResponse(http.StatusBadRequest, "unsupported beta", r), nil
		}),
	}
	client := &Client{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		Beta:       []string{"custom-beta"},
		MaxRetries: 1,
	}
	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	events, err := client.InvokeStream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var errSeen bool
	for ev := range events {
		if _, ok := ev.(llm.StreamErrorEvent); ok {
			errSeen = true
		}
	}
	if !errSeen {
		t.Fatalf("expected error event from stream")
	}
	if got := strings.Join(client.Beta, ","); got != "custom-beta" {
		t.Fatalf("expected beta unchanged, got %q", got)
	}
}

func TestClientThinkingDowngradeDoesNotMutateConfig(t *testing.T) {
	t.Parallel()
	calls := 0
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			calls++
			data, _ := io.ReadAll(r.Body)
			_ = r.Body.Close()
			var payload map[string]any
			if err := json.Unmarshal(data, &payload); err != nil {
				t.Fatalf("unmarshal request: %v", err)
			}
			_, hasThinking := payload["thinking"]
			if calls == 1 {
				if !hasThinking {
					t.Fatalf("expected thinking payload on first request")
				}
				return httpResponse(http.StatusBadRequest, "thinking unsupported", r), nil
			}
			if hasThinking {
				t.Fatalf("expected thinking payload removed on retry")
			}
			body := `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end","usage":{"input_tokens":1,"output_tokens":1}}`
			return httpResponse(http.StatusOK, body, r), nil
		}),
	}

	budget := 64
	client := &Client{
		HTTPClient:           httpClient,
		BaseURL:              "https://example.com",
		ModelName:            "test-model",
		ThinkingBudgetTokens: &budget,
		MaxRetries:           2,
	}
	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	if _, err := client.Invoke(context.Background(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 2 {
		t.Fatalf("expected 2 attempts, got %d", calls)
	}
	if client.ThinkingBudgetTokens == nil || *client.ThinkingBudgetTokens != budget {
		t.Fatalf("expected thinking budget unchanged, got %v", client.ThinkingBudgetTokens)
	}
}

func TestClientThinkingDowngradeDoesNotMutateConfigStream(t *testing.T) {
	t.Parallel()
	var sawThinking bool
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			data, _ := io.ReadAll(r.Body)
			_ = r.Body.Close()
			var payload map[string]any
			if err := json.Unmarshal(data, &payload); err != nil {
				t.Fatalf("unmarshal request: %v", err)
			}
			if _, hasThinking := payload["thinking"]; hasThinking {
				sawThinking = true
			}
			return httpResponse(http.StatusBadRequest, "thinking unsupported", r), nil
		}),
	}

	budget := 64
	client := &Client{
		HTTPClient:           httpClient,
		BaseURL:              "https://example.com",
		ModelName:            "test-model",
		ThinkingBudgetTokens: &budget,
		MaxRetries:           1,
	}
	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	events, err := client.InvokeStream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var errSeen bool
	for ev := range events {
		if _, ok := ev.(llm.StreamErrorEvent); ok {
			errSeen = true
		}
	}
	if !errSeen {
		t.Fatalf("expected error event from stream")
	}
	if !sawThinking {
		t.Fatalf("expected thinking payload in stream request")
	}
	if client.ThinkingBudgetTokens == nil || *client.ThinkingBudgetTokens != budget {
		t.Fatalf("expected thinking budget unchanged, got %v", client.ThinkingBudgetTokens)
	}
}

func TestInvokeThinkingDowngradeGetsExtraRetryOnFinalAttempt(t *testing.T) {
	t.Parallel()

	calls := 0
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			calls++
			data, _ := io.ReadAll(r.Body)
			_ = r.Body.Close()
			var payload map[string]any
			if err := json.Unmarshal(data, &payload); err != nil {
				t.Fatalf("unmarshal request: %v", err)
			}
			_, hasThinking := payload["thinking"]
			if calls == 1 {
				if !hasThinking {
					t.Fatalf("expected thinking payload on first request")
				}
				errBody := `{"error":{"type":"invalid_request_error","code":"unsupported_parameter","param":"thinking.budget_tokens","message":"unknown field"}}`
				return httpResponse(http.StatusBadRequest, errBody, r), nil
			}
			if calls == 2 {
				if hasThinking {
					t.Fatalf("expected thinking payload removed on downgrade retry")
				}
				body := `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end","usage":{"input_tokens":1,"output_tokens":1}}`
				return httpResponse(http.StatusOK, body, r), nil
			}
			t.Fatalf("unexpected request count: %d", calls)
			return nil, nil
		}),
	}

	budget := 64
	client := &Client{
		HTTPClient:           httpClient,
		BaseURL:              "https://example.com",
		ModelName:            "test-model",
		ThinkingBudgetTokens: &budget,
		MaxRetries:           1,
	}
	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	if _, err := client.Invoke(context.Background(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 2 {
		t.Fatalf("expected 2 attempts (1 final downgrade retry), got %d", calls)
	}
}

func TestInvokeStreamThinkingDowngradeGetsExtraRetryOnFinalAttempt(t *testing.T) {
	t.Parallel()

	calls := 0
	streamBody := strings.Join([]string{
		`data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
		"",
		`data: {"type":"content_block_delta","index":0,"delta":{"text":"ok"}}`,
		"",
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
		"",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			calls++
			data, _ := io.ReadAll(r.Body)
			_ = r.Body.Close()
			var payload map[string]any
			if err := json.Unmarshal(data, &payload); err != nil {
				t.Fatalf("unmarshal request: %v", err)
			}
			_, hasThinking := payload["thinking"]
			if calls == 1 {
				if !hasThinking {
					t.Fatalf("expected thinking payload on first request")
				}
				errBody := `{"error":{"type":"invalid_request_error","code":"unsupported_parameter","param":"thinking.budget_tokens","message":"unknown field"}}`
				return httpResponse(http.StatusBadRequest, errBody, r), nil
			}
			if calls == 2 {
				if hasThinking {
					t.Fatalf("expected thinking payload removed on downgrade retry")
				}
				return httpResponse(http.StatusOK, streamBody, r), nil
			}
			t.Fatalf("unexpected request count: %d", calls)
			return nil, nil
		}),
	}

	budget := 64
	client := &Client{
		HTTPClient:           httpClient,
		BaseURL:              "https://example.com",
		ModelName:            "test-model",
		ThinkingBudgetTokens: &budget,
		MaxRetries:           1,
	}
	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	events, err := client.InvokeStream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var errSeen bool
	var doneSeen bool
	for ev := range events {
		switch ev.(type) {
		case llm.StreamErrorEvent:
			errSeen = true
		case llm.StreamDoneEvent:
			doneSeen = true
		}
	}
	if errSeen {
		t.Fatalf("did not expect stream error after downgrade retry")
	}
	if !doneSeen {
		t.Fatalf("expected done event after downgraded retry")
	}
	if calls != 2 {
		t.Fatalf("expected 2 attempts (1 final downgrade retry), got %d", calls)
	}
}

func TestLooksLikeThinkingUnsupported(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		msg  string
		want bool
	}{
		{name: "plain unsupported thinking field", msg: "unknown field thinking.budget_tokens", want: true},
		{name: "structured unsupported code and param", msg: `{"error":{"type":"invalid_request_error","code":"unsupported_parameter","param":"thinking.budget_tokens","message":"unsupported field"}}`, want: true},
		{name: "generic thinking error", msg: "thinking about your request failed", want: false},
		{name: "structured non-compat error", msg: `{"error":{"type":"internal_error","message":"thinking about your request failed"}}`, want: false},
	}

	for _, tt := range tests {
		if got := looksLikeThinkingUnsupported(tt.msg); got != tt.want {
			t.Fatalf("%s: expected %v, got %v", tt.name, tt.want, got)
		}
	}
}

func TestLooksLikeBetaUnsupported(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		msg  string
		want bool
	}{
		{name: "structured unsupported beta param", msg: `{"error":{"type":"invalid_request_error","code":"unsupported_parameter","param":"anthropic-beta","message":"unsupported header"}}`, want: true},
		{name: "plain unsupported beta header", msg: "unknown anthropic-beta header", want: true},
		{name: "generic beta message", msg: "beta rollout delayed", want: false},
	}

	for _, tt := range tests {
		if got := looksLikeBetaUnsupported(tt.msg); got != tt.want {
			t.Fatalf("%s: expected %v, got %v", tt.name, tt.want, got)
		}
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

func TestConsumeSSEWithBodyCloseClosesOnPanic(t *testing.T) {
	body := &closeTrackingReadCloser{Reader: strings.NewReader("data: {\"type\":\"message_stop\"}\n\n")}
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

func TestConsumeSSEReturnsErrorForMalformedFinalEvent(t *testing.T) {
	input := "data: {\"type\":\"message_delta\"\n\n"
	err := consumeSSE(strings.NewReader(input), func(string) error { return nil })
	if err == nil {
		t.Fatalf("expected malformed SSE payload error")
	}
	if !strings.Contains(err.Error(), "malformed SSE event payload") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRandomBackoffFractionVariesAndStaysBounded(t *testing.T) {
	t.Parallel()

	seen := map[float64]struct{}{}
	for i := 0; i < 8; i++ {
		v := randomBackoffFraction()
		if v < 0 || v >= 1 {
			t.Fatalf("expected fraction in [0,1), got %f", v)
		}
		seen[v] = struct{}{}
	}
	if len(seen) < 2 {
		t.Fatalf("expected non-deterministic jitter samples, got %d unique values", len(seen))
	}
}

func TestNormalizeToolCallIDWithWarningLogsOriginalAndNormalizedIDs(t *testing.T) {
	var warnings []string
	warnf := func(format string, args ...any) {
		warnings = append(warnings, fmt.Sprintf(format, args...))
	}

	got := normalizeToolCallIDWithWarning("call:1/alpha", warnf)
	if got != "call_1_alpha" {
		t.Fatalf("unexpected normalized id: %q", got)
	}
	if len(warnings) != 1 {
		t.Fatalf("expected one warning, got %d", len(warnings))
	}
	if !strings.Contains(warnings[0], `original="call:1/alpha"`) || !strings.Contains(warnings[0], `normalized="call_1_alpha"`) {
		t.Fatalf("warning should include both ids, got %q", warnings[0])
	}
}

func TestNormalizeToolCallIDWithWarningSkipsLogWhenUnchanged(t *testing.T) {
	var warnings int
	warnf := func(string, ...any) {
		warnings++
	}

	got := normalizeToolCallIDWithWarning("call_1_alpha", warnf)
	if got != "call_1_alpha" {
		t.Fatalf("unexpected normalized id: %q", got)
	}
	if warnings != 0 {
		t.Fatalf("expected no warning when id is unchanged")
	}
}

func TestParseRetryAfterWarnsOnNonPositiveSeconds(t *testing.T) {
	origWarn := retryAfterWarningf
	defer func() { retryAfterWarningf = origWarn }()

	var warning string
	retryAfterWarningf = func(format string, args ...any) {
		warning = fmt.Sprintf(format, args...)
	}

	if got := parseRetryAfter("-5"); got != 0 {
		t.Fatalf("expected zero duration for non-positive value, got %v", got)
	}
	if warning == "" {
		t.Fatalf("expected warning for non-positive Retry-After")
	}
	if !strings.Contains(warning, "[WARN]") || !strings.Contains(warning, "-5") {
		t.Fatalf("expected actionable warning with raw Retry-After value, got %q", warning)
	}
}

func TestParseRetryAfterWarnsOnPastHTTPDate(t *testing.T) {
	origWarn := retryAfterWarningf
	defer func() { retryAfterWarningf = origWarn }()

	past := time.Now().Add(-1 * time.Minute).UTC().Format(http.TimeFormat)
	var warning string
	retryAfterWarningf = func(format string, args ...any) {
		warning = fmt.Sprintf(format, args...)
	}

	if got := parseRetryAfter(past); got != 0 {
		t.Fatalf("expected zero duration for past date value, got %v", got)
	}
	if warning == "" {
		t.Fatalf("expected warning for past-date Retry-After")
	}
	if !strings.Contains(warning, "[WARN]") || !strings.Contains(warning, past) {
		t.Fatalf("expected actionable warning with Retry-After date, got %q", warning)
	}
}

func TestAnthropicRejectsOversizedResponseBody(t *testing.T) {
	prevLimit := maxProviderResponseBytes
	maxProviderResponseBytes = 64
	t.Cleanup(func() { maxProviderResponseBytes = prevLimit })

	body := strings.Repeat("x", int(maxProviderResponseBytes)+8)
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return httpResponse(http.StatusOK, body, r), nil
		}),
	}

	client := &Client{
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
	assertAnthropicResponseSizeDiagnostic(t, providerErr.Message, maxProviderResponseBytes)
}

func TestAnthropicIncludesResponseSizeDiagnostic(t *testing.T) {
	prevLimit := maxProviderResponseBytes
	maxProviderResponseBytes = 64
	t.Cleanup(func() { maxProviderResponseBytes = prevLimit })

	body := strings.Repeat("x", int(maxProviderResponseBytes)+8)
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return httpResponse(http.StatusBadGateway, body, r), nil
		}),
	}

	client := &Client{
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
	assertAnthropicResponseSizeDiagnostic(t, providerErr.Message, maxProviderResponseBytes)
}

func assertAnthropicResponseSizeDiagnostic(t *testing.T, msg string, limit int64) {
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
