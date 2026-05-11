package openai

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestChatClientProviderLabelPropagatesProviderError(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "chat failed", http.StatusBadGateway)
	}))
	defer server.Close()

	client := &ChatClient{
		HTTPClient:    server.Client(),
		BaseURL:       server.URL,
		APIKey:        "test",
		ModelName:     "test-model",
		ProviderLabel: "openai-chat",
		MaxRetries:    1,
	}
	if got := client.Provider(); got != "openai-chat" {
		t.Fatalf("Provider() = %q, want openai-chat", got)
	}
	_, err := client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hi")}})
	if err == nil {
		t.Fatalf("expected provider error")
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T", err)
	}
	if providerErr.Provider != "openai-chat" {
		t.Fatalf("ProviderError.Provider = %q, want openai-chat", providerErr.Provider)
	}
}

func TestResponsesClientProviderLabelPropagatesProviderError(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "responses failed", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	client := &ResponsesClient{
		HTTPClient:    server.Client(),
		BaseURL:       server.URL,
		APIKey:        "test",
		ModelName:     "test-model",
		ProviderLabel: "openai-responses",
		MaxRetries:    1,
	}
	if got := client.Provider(); got != "openai-responses" {
		t.Fatalf("Provider() = %q, want openai-responses", got)
	}
	_, err := client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hi")}})
	if err == nil {
		t.Fatalf("expected provider error")
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T", err)
	}
	if providerErr.Provider != "openai-responses" {
		t.Fatalf("ProviderError.Provider = %q, want openai-responses", providerErr.Provider)
	}
}
