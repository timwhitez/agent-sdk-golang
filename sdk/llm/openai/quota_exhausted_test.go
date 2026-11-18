package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// Only the structured code types quota exhaustion; the SDK then makes exactly
// one request, while any other 429 keeps retrying through the whole budget.
func TestOpenAIQuotaExhaustedRateLimitIsTypedAndNotRetried(t *testing.T) {
	const retries = 3
	req := llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hi")}}
	for _, tc := range []struct {
		name         string
		body         string
		wantReason   string
		wantRequests int32
	}{
		{"quota", `{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}`, llm.RateLimitReasonQuotaExhausted, 1},
		{"message only", `{"error":{"message":"insufficient_quota","type":"insufficient_quota","code":null}}`, "", retries},
		{"plain", `{"error":{"message":"slow down","type":"requests","code":"rate_limit_exceeded"}}`, "", retries},
	} {
		var requests atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			requests.Add(1)
			w.WriteHeader(http.StatusTooManyRequests)
			io.WriteString(w, tc.body)
		}))
		chat := &ChatClient{BaseURL: server.URL, ModelName: "m", MaxRetries: retries, RetryBaseDelay: time.Millisecond, RetryMaxDelay: time.Millisecond}
		responses := &ResponsesClient{BaseURL: server.URL, ModelName: "m", MaxRetries: retries, RetryBaseDelay: time.Millisecond, RetryMaxDelay: time.Millisecond}
		for name, invoke := range map[string]func() error{
			"chat/buffered": func() error { _, err := chat.Invoke(context.Background(), req); return err },
			"chat/stream": func() error {
				events, err := chat.InvokeStream(context.Background(), req)
				if err != nil {
					return err
				}
				return streamedError(events)
			},
			"responses/buffered": func() error { _, err := responses.Invoke(context.Background(), req); return err },
			"responses/stream": func() error {
				events, err := responses.InvokeStream(context.Background(), req)
				if err != nil {
					return err
				}
				return streamedError(events)
			},
		} {
			requests.Store(0)
			err := invoke()
			var rl *llm.RateLimitError
			if !errors.As(err, &rl) {
				t.Fatalf("%s/%s: want RateLimitError, got %T %v", tc.name, name, err, err)
			}
			if rl.Reason != tc.wantReason || llm.IsQuotaExhausted(err) != (tc.wantReason != "") {
				t.Fatalf("%s/%s: reason=%q want %q", tc.name, name, rl.Reason, tc.wantReason)
			}
			if got := requests.Load(); got != tc.wantRequests {
				t.Fatalf("%s/%s: requests=%d want %d", tc.name, name, got, tc.wantRequests)
			}
		}
		server.Close()
	}
}

// A Responses in-stream 429 error event carries the same typing.
func TestResponsesStreamErrorEventQuotaExhausted(t *testing.T) {
	for _, tc := range []struct {
		code string
		want bool
	}{{"insufficient_quota", true}, {"rate_limit_exceeded", false}} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintf(w, "data: {\"type\":\"error\",\"error\":{\"code\":%q,\"status\":429,\"message\":\"failed\"}}\n\n", tc.code)
		}))
		client := &ResponsesClient{BaseURL: server.URL, ModelName: "m", MaxRetries: 1}
		events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hi")}})
		if err != nil {
			t.Fatal(err)
		}
		got := streamedError(events)
		server.Close()
		var rl *llm.RateLimitError
		if !errors.As(got, &rl) || llm.IsQuotaExhausted(got) != tc.want {
			t.Fatalf("code=%s err=%v", tc.code, got)
		}
	}
}
