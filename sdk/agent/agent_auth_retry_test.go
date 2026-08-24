package agent

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
)

func TestAgentDoesNotRetryOpenAIAuthFailures(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		status := status
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				_, _ = fmt.Fprint(w, `{"error":{"message":"denied"}}`)
			}))
			defer server.Close()
			client := &openai.ChatClient{BaseURL: server.URL, ModelName: "test-model"}
			agent, err := New(Config{LLM: client})
			if err != nil {
				t.Fatal(err)
			}
			_, err = agent.Query(context.Background(), "hello")
			if err == nil {
				t.Fatal("expected authentication failure")
			}
			if got := requests.Load(); got != 1 {
				t.Fatalf("HTTP requests = %d, want one provider/framework attempt", got)
			}
		})
	}
}
