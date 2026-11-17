package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// #254: MaxRetries is an attempt budget that includes the first request.
func TestMaxRetriesIsAttemptBudget(t *testing.T) {
	for _, maxRetries := range []int{1, 2, 3} {
		var hits atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			hits.Add(1)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"synthetic"}}`))
		}))
		c := &Client{ModelName: "m", BaseURL: srv.URL, APIKey: "synthetic", MaxTokens: 16, MaxRetries: maxRetries,
			RetryBaseDelay: time.Millisecond, RetryMaxDelay: time.Millisecond, HTTPClient: srv.Client()}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_, err := c.Invoke(ctx, llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("x")}})
		cancel()
		srv.Close()
		if err == nil {
			t.Fatal("expected an error from an always-503 endpoint")
		}
		if got := int(hits.Load()); got != maxRetries {
			t.Errorf("MaxRetries=%d sent %d requests, want %d", maxRetries, got, maxRetries)
		}
	}
}
