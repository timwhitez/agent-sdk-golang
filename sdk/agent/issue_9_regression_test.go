package agent

import (
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestAuditHTTPPermissionRetry(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusBadRequest} {
		for _, msg := range []string{"model permission denied", "rate limit settings are unavailable for this project"} {
			t.Run(fmt.Sprintf("%d/%s", status, msg), func(t *testing.T) {
				var n atomic.Int32
				s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					n.Add(1)
					w.WriteHeader(status)
					fmt.Fprintf(w, `{"error":{"message":%q,"type":"permission_error"}}`, msg)
				}))
				defer s.Close()
				m := &openai.ChatClient{BaseURL: s.URL, ModelName: "fixture", MaxRetries: 1}
				a, err := New(Config{LLM: m, InvokeRetryMaxAttempts: 3, InvokeRetryBackoff: time.Millisecond, Warningf: func(string, ...any) {}})
				if err != nil {
					t.Fatal(err)
				}
				if _, err = a.Query(t.Context(), "hello"); err == nil {
					t.Fatal("expected permission error")
				}
				if got := n.Load(); got != 1 {
					t.Fatalf("HTTP %d attempts=%d, want 1", status, got)
				}
			})
		}
	}
}
