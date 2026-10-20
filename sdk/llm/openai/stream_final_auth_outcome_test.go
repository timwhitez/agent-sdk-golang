package openai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// Exercise two real HTTP responses and the actual SDK forwarding drop boundary.
func TestStreamOutcomeFinalAuthAfterInternalRetry(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, firstStatus := range []int{500, 429} {
			for _, finalStatus := range []int{401, 403} {
				for _, drop := range []bool{false, true} {
					t.Run(fmt.Sprintf("responses=%t/first=%d/final=%d/drop=%t", responses, firstStatus, finalStatus, drop), func(t *testing.T) {
						var calls atomic.Int32
						server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							status := finalStatus
							if calls.Add(1) == 1 {
								status = firstStatus
								if firstStatus == 429 {
									w.Header().Set("Retry-After", "1")
								}
							}
							w.Header().Set("Content-Type", "application/json")
							w.WriteHeader(status)
							fmt.Fprint(w, `{"error":{"message":"synthetic final-auth fixture"}}`)
						}))
						defer server.Close()
						parent, stopParent := context.WithCancel(context.Background())
						defer stopParent()
						ctx, outcome := llm.WithStreamOutcome(parent)
						ctx, cancelChild := context.WithCancel(ctx)
						defer cancelChild()
						beforeWait := make(chan llm.StreamOutcomeSnapshot, 1)
						ctx = context.WithValue(ctx, retryBackoffTestHookKey{}, &retryBackoffTestHooks{waiting: func(_ error, _ time.Duration) {
							beforeWait <- outcome.Snapshot()
						}})
						dropped := make(chan struct{}, 1)
						if drop {
							ctx = context.WithValue(ctx, streamForwardTestHookKey{}, &streamForwardTestHooks{
								beforeDelivery: func(event llm.StreamEvent, dst chan<- llm.StreamEvent) {
									if _, ok := event.(llm.StreamErrorEvent); ok {
										for i := 0; i < cap(dst); i++ {
											dst <- llm.StreamUsageEvent{}
										}
										cancelChild()
									}
								},
								dropped: func(event llm.StreamEvent) {
									if _, ok := event.(llm.StreamErrorEvent); ok {
										dropped <- struct{}{}
									}
								},
							})
						}
						var model llm.StreamingChatModel = &ChatClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "fixture", MaxRetries: 2, RetryBaseDelay: time.Millisecond, RetryMaxDelay: time.Millisecond, RetryAfterMaxDelay: 2 * time.Second}
						if responses {
							model = &ResponsesClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "fixture", MaxRetries: 2, RetryBaseDelay: time.Millisecond, RetryMaxDelay: time.Millisecond, RetryAfterMaxDelay: 2 * time.Second}
						}
						ch, err := model.InvokeStream(ctx, llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("local fixture")}})
						if err != nil {
							t.Fatal(err)
						}
						if drop {
							select {
							case <-dropped:
							case <-time.After(4 * time.Second):
								t.Fatal("actual final-error drop not reached")
							}
						}
						publicErrors := 0
						for event := range ch {
							if e, ok := event.(llm.StreamErrorEvent); ok {
								publicErrors++
								var provider *llm.ProviderError
								if !errors.As(e.AsError(), &provider) || provider.StatusCode != finalStatus {
									t.Fatal("public final auth error missing")
								}
							}
						}
						original := <-beforeWait
						got := outcome.Snapshot()
						var provider *llm.ProviderError
						t.Logf("requests=%d public_errors=%d complete=%t error=%v original_floor=%v final_floor=%v", calls.Load(), publicErrors, got.Complete, got.Error, original.RetryAfterNotBefore, got.RetryAfterNotBefore)
						if calls.Load() != 2 || parent.Err() != nil || !got.Started || !got.Complete || got.Visible || llm.IsRetryStop(got.Error) {
							t.Fatal("two-send, live-parent, completed ordinary-error premise absent")
						}
						if (drop && publicErrors != 0) || (!drop && publicErrors != 1) {
							t.Fatal("actual delivery/drop premise absent")
						}
						if firstStatus == 429 {
							if original.RetryAfter != time.Second || got.RetryAfter != original.RetryAfter || got.RetryAfterNotBefore != original.RetryAfterNotBefore {
								t.Fatal("historical parsed absolute floor changed after final auth error")
							}
						} else if got.RetryAfter != 0 || !got.RetryAfterNotBefore.IsZero() {
							t.Fatal("no-hint 500 invented floor")
						}
						if !errors.As(got.Error, &provider) || provider.StatusCode != finalStatus || !errors.As(got.FinalError, &provider) || provider.StatusCode != finalStatus {
							t.Fatal("historical retryable error hid final permanent HTTP error")
						}
					})
				}
			}
		}
	}
}

func TestStreamOutcomeActualRetrySuccessHasNoFinalError(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, firstStatus := range []int{401, 403, 500, 429} {
			t.Run(fmt.Sprintf("responses=%t/first=%d", responses, firstStatus), func(t *testing.T) {
				var calls atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if calls.Add(1) == 1 {
						w.Header().Set("Content-Type", "application/json")
						if firstStatus == 429 {
							w.Header().Set("Retry-After", "1")
						}
						w.WriteHeader(firstStatus)
						fmt.Fprint(w, `{"error":{"message":"synthetic recovered failure"}}`)
						return
					}
					w.Header().Set("Content-Type", "text/event-stream")
					if responses {
						fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"fixture\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}]}}\n\n")
					} else {
						fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
					}
				}))
				defer server.Close()
				ctx, outcome := llm.WithStreamOutcome(context.Background())
				statuses := map[int]struct{}{firstStatus: {}}
				var model llm.StreamingChatModel = &ChatClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "fixture", MaxRetries: 2, RetryableStatusCodes: statuses, RetryBaseDelay: time.Millisecond, RetryMaxDelay: time.Millisecond, RetryAfterMaxDelay: 2 * time.Second}
				if responses {
					model = &ResponsesClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "fixture", MaxRetries: 2, RetryableStatusCodes: statuses, RetryBaseDelay: time.Millisecond, RetryMaxDelay: time.Millisecond, RetryAfterMaxDelay: 2 * time.Second}
				}
				ch, err := model.InvokeStream(ctx, llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("local fixture")}})
				if err != nil {
					t.Fatal(err)
				}
				done := false
				for event := range ch {
					if e, ok := event.(llm.StreamErrorEvent); ok {
						t.Fatalf("recovered stream failed: %v", e.AsError())
					}
					if _, ok := event.(llm.StreamDoneEvent); ok {
						done = true
					}
				}
				got := outcome.Snapshot()
				if calls.Load() != 2 || !done || !got.Complete || got.Error == nil || got.FinalError != nil || llm.IsRetryStop(got.Error) {
					t.Fatalf("historical parsed failure fabricated final error after real successful retry: calls=%d done=%t receipt=%+v", calls.Load(), done, got)
				}
				if firstStatus == 429 && (got.RetryAfter != time.Second || got.RetryAfterCause == nil || got.RetryAfterNotBefore.IsZero()) {
					t.Fatal("recovery erased historical floor/cause")
				}
			})
		}
	}
}
