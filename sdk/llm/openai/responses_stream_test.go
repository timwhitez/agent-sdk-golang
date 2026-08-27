package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestResponsesStreamDecodeError(t *testing.T) {
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
	client := &ResponsesClient{
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
	if !strings.Contains(errEvent.Err.Error(), "decode error") {
		t.Fatalf("expected decode error message, got %v", errEvent.Err)
	}
}

func TestResponsesStreamEmitsResponseIDWithoutCompletedEvent(t *testing.T) {
	t.Parallel()
	body := strings.Join([]string{
		`data: {"type":"response.output_text.delta","response_id":"resp_stream_123","delta":"hello"}`,
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

	client := &ResponsesClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	ch, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	responseID := ""
	text := ""
	for ev := range ch {
		switch e := ev.(type) {
		case llm.StreamResponseEvent:
			responseID = e.ResponseID
		case llm.StreamTextDeltaEvent:
			text += e.Delta
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.AsError())
		}
	}
	if responseID != "resp_stream_123" {
		t.Fatalf("response id = %q, want %q", responseID, "resp_stream_123")
	}
	if text != "hello" {
		t.Fatalf("text = %q, want %q", text, "hello")
	}
}

func TestResponsesStreamToolIndexTracksOutputIndexWhenIDsAppearLater(t *testing.T) {
	t.Parallel()
	body := strings.Join([]string{
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","name":"lookup","arguments":""}}`,
		"",
		`data: {"type":"response.output_item.added","output_index":1,"item":{"id":"call_real","type":"function_call","name":"search","arguments":""}}`,
		"",
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"call_later","delta":"{\"path\":\"notes.txt\"}"}`,
		"",
		`data: {"type":"response.function_call_arguments.delta","output_index":1,"item_id":"call_real","delta":"{\"query\":\"golang\"}"}`,
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

	client := &ResponsesClient{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		MaxRetries: 1,
	}

	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	ch, err := client.InvokeStream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lookupNameIdx := -1
	callLaterArgsIdx := -1
	realNameIdx := -1
	realArgsIdx := -1

	for ev := range ch {
		switch e := ev.(type) {
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.AsError())
		case llm.StreamToolCallDeltaEvent:
			if e.NameDelta == "lookup" {
				lookupNameIdx = e.Index
			}
			if e.ID == "call_later" && e.ArgumentsDelta != "" {
				callLaterArgsIdx = e.Index
			}
			if e.ID == "call_real" && e.NameDelta == "search" {
				realNameIdx = e.Index
			}
			if e.ID == "call_real" && e.ArgumentsDelta != "" {
				realArgsIdx = e.Index
			}
		}
	}

	if lookupNameIdx != 0 {
		t.Fatalf("expected lookup tool call index 0, got %d", lookupNameIdx)
	}
	if callLaterArgsIdx != lookupNameIdx {
		t.Fatalf("expected late call id args to map to lookup index %d, got %d", lookupNameIdx, callLaterArgsIdx)
	}
	if realNameIdx != 1 {
		t.Fatalf("expected real tool call index 1, got %d", realNameIdx)
	}
	if realArgsIdx != realNameIdx {
		t.Fatalf("expected real id args to map to index %d, got %d", realNameIdx, realArgsIdx)
	}
}

func TestResponsesStreamDoesNotDuplicateToolIdentityFromOutputItemDone(t *testing.T) {
	t.Parallel()
	body := strings.Join([]string{
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"call_1","type":"function_call","name":"ls","arguments":"{\"path\":\".\"}"}}`,
		"",
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"call_1","type":"function_call","name":"ls","arguments":"{\"path\":\".\"}"}}`,
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

	client := &ResponsesClient{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		MaxRetries: 1,
	}

	req := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}
	ch, err := client.InvokeStream(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	nameByIndex := map[int]string{}
	argsByIndex := map[int]string{}
	for ev := range ch {
		switch e := ev.(type) {
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.AsError())
		case llm.StreamToolCallDeltaEvent:
			nameByIndex[e.Index] += e.NameDelta
			argsByIndex[e.Index] += e.ArgumentsDelta
		}
	}

	if got := nameByIndex[0]; got != "ls" {
		t.Fatalf("expected tool name emitted once, got %q", got)
	}
	if got := argsByIndex[0]; got != `{"path":"."}` {
		t.Fatalf("expected tool args emitted once, got %q", got)
	}
}

func TestResponsesStreamResponseErrorPreservesRateLimitMetadata(t *testing.T) {
	t.Parallel()
	body := strings.Join([]string{
		`data: {"type":"response.error","error":{"message":"too many requests","code":"rate_limit_exceeded","status":429,"retry_after_ms":2500}}`,
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

	client := &ResponsesClient{
		HTTPClient: httpClient,
		BaseURL:    "https://example.com",
		ModelName:  "test-model",
		MaxRetries: 1,
	}

	ch, err := client.InvokeStream(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	var streamErr llm.StreamErrorEvent
	errSeen := false
	for ev := range ch {
		if e, ok := ev.(llm.StreamErrorEvent); ok {
			streamErr = e
			errSeen = true
		}
	}
	if !errSeen {
		t.Fatalf("expected stream error event")
	}

	var rateLimitErr *llm.RateLimitError
	if !errors.As(streamErr.AsError(), &rateLimitErr) {
		t.Fatalf("expected rate limit error, got %T", streamErr.AsError())
	}
	if rateLimitErr.Provider != "openai" {
		t.Fatalf("provider = %q, want %q", rateLimitErr.Provider, "openai")
	}
	if rateLimitErr.RetryAfter != 2500*time.Millisecond {
		t.Fatalf("retry_after = %s, want %s", rateLimitErr.RetryAfter, 2500*time.Millisecond)
	}
	if !strings.Contains(strings.ToLower(rateLimitErr.Message), "too many requests") {
		t.Fatalf("message = %q, want rate-limit reason", rateLimitErr.Message)
	}
}

func TestResponsesStreamConsumesFunctionCallArgumentsDone(t *testing.T) {
	t.Parallel()
	body := strings.Join([]string{
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"call_done","type":"function_call","name":"search"}}`,
		"",
		`data: {"type":"response.function_call_arguments.done","output_index":0,"item_id":"call_done","arguments":"{\"query\":\"golang\"}"}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
	})}
	client := &ResponsesClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	ch, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}
	toolName := ""
	toolArgs := ""
	for ev := range ch {
		switch e := ev.(type) {
		case llm.StreamToolCallDeltaEvent:
			if e.NameDelta != "" {
				toolName = e.NameDelta
			}
			if e.ArgumentsDelta != "" {
				toolArgs += e.ArgumentsDelta
			}
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.AsError())
		}
	}
	if toolName != "search" {
		t.Fatalf("tool name = %q, want %q", toolName, "search")
	}
	if toolArgs != `{"query":"golang"}` {
		t.Fatalf("tool args = %q, want %q", toolArgs, `{"query":"golang"}`)
	}
}

func TestResponsesStreamAcceptsObjectArgumentsInOutputItemDone(t *testing.T) {
	t.Parallel()
	body := strings.Join([]string{
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"call_obj","type":"function_call","name":"search","arguments":{"query":"golang"}}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
	})}
	client := &ResponsesClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	ch, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}
	toolArgs := ""
	for ev := range ch {
		switch e := ev.(type) {
		case llm.StreamToolCallDeltaEvent:
			if e.ArgumentsDelta != "" {
				toolArgs += e.ArgumentsDelta
			}
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.AsError())
		}
	}
	if toolArgs != `{"query":"golang"}` {
		t.Fatalf("tool args = %q, want %q", toolArgs, `{"query":"golang"}`)
	}
}

func TestResponsesStreamSeparatesItemIDFromFunctionCallID(t *testing.T) {
	t.Parallel()
	body := strings.Join([]string{
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_stream","call_id":"call_stream","type":"function_call","name":"lookup","arguments":""}}`,
		"",
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_stream","delta":"{\"query\":"}`,
		"",
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_stream","delta":"\"golang\"}"}`,
		"",
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_stream","call_id":"call_stream","type":"function_call","name":"lookup","arguments":"{\"query\":\"golang\"}"}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
	})}
	client := &ResponsesClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	ch, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	name := ""
	args := ""
	for ev := range ch {
		switch e := ev.(type) {
		case llm.StreamToolCallDeltaEvent:
			if e.ID != "" && e.ID != "call_stream" {
				t.Fatalf("stream exposed item id as tool-call id: %#v", e)
			}
			name += e.NameDelta
			args += e.ArgumentsDelta
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.AsError())
		}
	}
	if name != "lookup" {
		t.Fatalf("tool name = %q, want lookup", name)
	}
	if args != `{"query":"golang"}` {
		t.Fatalf("tool args = %q, want complete arguments", args)
	}
}

func TestResponsesStreamBuffersArgumentsUntilLateCallID(t *testing.T) {
	t.Parallel()
	body := strings.Join([]string{
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_late","delta":"{\"path\":\"notes.txt\"}"}`,
		"",
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_late","call_id":"call_late","type":"function_call","name":"read","arguments":""}}`,
		"",
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_late","call_id":"call_late","type":"function_call","name":"read","arguments":"{\"path\":\"notes.txt\"}"}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
	})}
	client := &ResponsesClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	ch, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	name := ""
	args := ""
	for ev := range ch {
		switch e := ev.(type) {
		case llm.StreamToolCallDeltaEvent:
			if e.ID != "" && e.ID != "call_late" {
				t.Fatalf("late identity exposed wrong tool-call ID: %#v", e)
			}
			name += e.NameDelta
			args += e.ArgumentsDelta
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.AsError())
		}
	}
	if name != "read" || args != `{"path":"notes.txt"}` {
		t.Fatalf("streamed tool = name %q args %q", name, args)
	}
}

func TestResponsesStreamKeepsItemAndCallIDNamespacesSeparate(t *testing.T) {
	t.Parallel()
	body := strings.Join([]string{
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_a","call_id":"shared_identifier","type":"function_call","name":"first","arguments":""}}`,
		"",
		`data: {"type":"response.output_item.added","output_index":1,"item":{"id":"shared_identifier","call_id":"call_b","type":"function_call","name":"second","arguments":""}}`,
		"",
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_a","delta":"{\"value\":1}"}`,
		"",
		`data: {"type":"response.function_call_arguments.delta","output_index":1,"item_id":"shared_identifier","delta":"{\"value\":2}"}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
	})}
	client := &ResponsesClient{HTTPClient: httpClient, BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	ch, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}}})
	if err != nil {
		t.Fatalf("invoke stream: %v", err)
	}

	names := map[int]string{}
	args := map[int]string{}
	ids := map[int]string{}
	for ev := range ch {
		switch e := ev.(type) {
		case llm.StreamToolCallDeltaEvent:
			names[e.Index] += e.NameDelta
			args[e.Index] += e.ArgumentsDelta
			if e.ID != "" {
				ids[e.Index] = e.ID
			}
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected stream error: %v", e.AsError())
		}
	}
	if names[0] != "first" || args[0] != `{"value":1}` || ids[0] != "shared_identifier" {
		t.Fatalf("first tool identity collapsed: names=%#v args=%#v ids=%#v", names, args, ids)
	}
	if names[1] != "second" || args[1] != `{"value":2}` || ids[1] != "call_b" {
		t.Fatalf("second tool identity collapsed: names=%#v args=%#v ids=%#v", names, args, ids)
	}
}
