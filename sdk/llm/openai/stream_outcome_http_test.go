package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestStreamOutcomeActualHTTPBeforeForwardDrop(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, date := range []bool{false, true} {
			for _, terminal := range []bool{false, true} {
				t.Run(fmt.Sprintf("responses=%t/date=%t/native_stop=%t", responses, date, terminal), func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						hint := "1"
						if date {
							hint = time.Now().Add(3 * time.Second).UTC().Format(http.TimeFormat)
						}
						w.Header().Set("Retry-After", hint)
						w.Header().Set("Content-Type", "application/json")
						w.WriteHeader(429)
						fmt.Fprint(w, `{"error":{"message":"synthetic local fixture"}}`)
					}))
					defer server.Close()
					ctx, outcome := llm.WithStreamOutcome(context.Background())
					ctx, cancel := context.WithCancel(ctx)
					defer cancel()
					parsed := make(chan error, 1)
					dropped := make(chan struct{})
					hooks := &streamForwardTestHooks{
						beforeDelivery: func(event llm.StreamEvent, dst chan<- llm.StreamEvent) {
							if e, ok := event.(llm.StreamErrorEvent); ok {
								parsed <- e.AsError()
								// The production downstream queue is full and has no reader.
								// Thus its actual select must choose cancellation, not delivery.
								for i := 0; i < cap(dst); i++ {
									dst <- llm.StreamUsageEvent{}
								}
								cancel()
							}
						},
						dropped: func(event llm.StreamEvent) {
							if _, ok := event.(llm.StreamErrorEvent); ok {
								close(dropped)
							}
						},
					}
					ctx = context.WithValue(ctx, streamForwardTestHookKey{}, hooks)
					maxRetries := 1
					maxWait := time.Duration(0)
					if terminal {
						maxRetries = 2
						maxWait = time.Millisecond
					}
					var model llm.StreamingChatModel = &ChatClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "fixture", MaxRetries: maxRetries, RetryAfterMaxDelay: maxWait}
					if responses {
						model = &ResponsesClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "fixture", MaxRetries: maxRetries, RetryAfterMaxDelay: maxWait}
					}
					ch, err := model.InvokeStream(ctx, llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("local fixture")}})
					if err != nil {
						t.Fatal(err)
					}
					select {
					case <-dropped:
					case <-time.After(3 * time.Second):
						t.Fatal("actual parsed error/drop boundary not reached")
					}
					actual := <-parsed
					if llm.RetryAfterForError(actual) <= 0 || llm.IsRetryStop(actual) != terminal {
						t.Fatal("actual parser/native stop premise absent")
					}
					publicErrors := 0
					for event := range ch {
						if _, ok := event.(llm.StreamErrorEvent); ok {
							publicErrors++
						}
					}
					got := outcome.Snapshot()
					t.Logf("actual parsed_hint=%s native_stop=%t public_errors=%d receipt_started=%t complete=%t retained_hint=%s", llm.RetryAfterForError(actual), llm.IsRetryStop(actual), publicErrors, got.Started, got.Complete, got.RetryAfter)
					if publicErrors != 0 {
						t.Fatal("drop branch did not lose public error")
					}
					if !got.Started || !got.Complete || got.Error != actual || got.RetryAfter <= 0 || got.RetryAfterNotBefore.IsZero() || llm.IsRetryStop(got.Error) != terminal {
						t.Fatal("parsed Retry-After/native terminal lost before host delivery; channel closure cannot certify no hint")
					}
				})
			}
		}
	}
}
