package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func authFailureServer(status int, requests *atomic.Int32) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = fmt.Fprint(w, `{"error":{"message":"denied"}}`)
	}))
}

func TestOpenAIAuthFailuresAreNotRetriedByDefault(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		status := status
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			for _, provider := range []string{"chat", "responses"} {
				provider := provider
				t.Run(provider, func(t *testing.T) {
					var requests atomic.Int32
					server := authFailureServer(status, &requests)
					defer server.Close()
					var err error
					switch provider {
					case "chat":
						client := &ChatClient{BaseURL: server.URL, ModelName: "test-model"}
						_, err = client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
					case "responses":
						client := &ResponsesClient{BaseURL: server.URL, ModelName: "test-model"}
						_, err = client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
					}
					if err == nil {
						t.Fatal("expected authentication failure")
					}
					if got := requests.Load(); got != 1 {
						t.Fatalf("HTTP requests = %d, want 1", got)
					}
				})
			}
		})
	}
}

func TestOpenAIAuthRetryCanBeExplicitlyEnabled(t *testing.T) {
	var requests atomic.Int32
	server := authFailureServer(http.StatusUnauthorized, &requests)
	defer server.Close()
	client := &ChatClient{
		BaseURL:              server.URL,
		ModelName:            "test-model",
		MaxRetries:           2,
		RetryableStatusCodes: map[int]struct{}{http.StatusUnauthorized: {}},
	}
	_, err := client.Invoke(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err == nil {
		t.Fatal("expected authentication failure")
	}
	if got := requests.Load(); got != 2 {
		t.Fatalf("HTTP requests = %d, want explicit two attempts", got)
	}
}
