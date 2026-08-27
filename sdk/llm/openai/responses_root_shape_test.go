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

func TestParseResponsesRejectsNonObjectRoots(t *testing.T) {
	const provider = "gateway-root-shape"
	tests := []struct {
		name    string
		payload string
	}{
		{name: "null", payload: `null`},
		{name: "whitespace null", payload: " \n null \t"},
		{name: "array", payload: `[]`},
		{name: "string", payload: `"response"`},
		{name: "number", payload: `1`},
		{name: "boolean", payload: `true`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			completion, err := parseResponsesForProvider(provider, []byte(tc.payload))
			if err == nil || completion != nil {
				t.Fatalf("non-object root accepted: completion=%#v err=%v", completion, err)
			}
			var providerErr *llm.ProviderError
			if !errors.As(err, &providerErr) || providerErr.Provider != provider {
				t.Fatalf("root-shape error = %#v, want ProviderError for %q", err, provider)
			}
		})
	}
}

func TestParseResponsesStillAcceptsObjectRoots(t *testing.T) {
	completion, err := parseResponses([]byte(`{"id":"resp_valid","status":"completed","output":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	if completion == nil || completion.ResponseID != "resp_valid" || completion.StopReason != "end_turn" {
		t.Fatalf("valid object completion = %#v", completion)
	}
}

func TestResponsesInvokeRejectsNullHTTPPayload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("null"))
	}))
	defer server.Close()

	client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1, ProviderLabel: "gateway-root-shape"}
	completion, err := client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err == nil || completion != nil || !strings.Contains(err.Error(), "non-null JSON object") {
		t.Fatalf("null HTTP payload result: completion=%#v err=%v", completion, err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Provider != "gateway-root-shape" {
		t.Fatalf("HTTP root-shape error = %#v, want labeled ProviderError", err)
	}
}
