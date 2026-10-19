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

func TestRetryAfterPhysicalDispatchFloor(t *testing.T) {
	testRetryAfterPhysicalDispatchFloor(t)
}

func TestRetryAfterPhysicalZeroJitter(t *testing.T) {
	original := backoffRandRead
	t.Cleanup(func() { backoffRandRead = original })
	backoffRandRead = func(b []byte) (int, error) {
		for i := range b {
			b[i] = 0
		}
		return len(b), nil
	}
	testRetryAfterPhysicalDispatchFloor(t)
}

func testRetryAfterPhysicalDispatchFloor(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for _, tc := range []struct {
				name, hint string
				stop       bool
				deadline   time.Duration
			}{
				{"floor_above_local_cap", "0.025", false, time.Second},
				{"long_hint_stop", "90", true, time.Second},
				{"deadline_stop", "0.1", true, 50 * time.Millisecond},
				{"http_date_floor", "http-date", false, 3 * time.Second},
				{"huge_hint_stop", "92233720370000000000000000", true, time.Second},
				{"no_hint", "", false, time.Second},
				{"invalid_hint", "1m", false, time.Second},
				{"success_once", "", false, time.Second},
				{"explicit_wait_limit", "0.025", true, time.Second},
			} {
				name := tc.name
				if responses {
					name += "/responses"
				} else {
					name += "/chat"
				}
				if stream {
					name += "/stream"
				}
				t.Run(name, func(t *testing.T) {
					var sends []time.Time
					var notBefore time.Time
					client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
						sends = append(sends, time.Now())
						status, body := 200, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
						if responses {
							body = `{"id":"r","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`
						}
						if stream {
							if responses {
								body = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":" + body + "}\n\n"
							} else {
								body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
							}
						}
						h := http.Header{}
						if len(sends) == 1 && tc.name != "success_once" {
							status = 429
							body = `{"error":{"message":"fixture"}}`
							h.Set("Retry-After", tc.hint)
							if tc.hint == "http-date" {
								notBefore = time.Now().Add(1100 * time.Millisecond).UTC().Truncate(time.Second)
								h.Set("Retry-After", notBefore.Format(http.TimeFormat))
							}
						}
						return &http.Response{StatusCode: status, Header: h, Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
					})}
					var model llm.StreamingChatModel
					var maxHintWait time.Duration
					if tc.name == "explicit_wait_limit" {
						maxHintWait = 10 * time.Millisecond
					}
					if responses {
						model = &ResponsesClient{HTTPClient: client, ModelName: "fixture", MaxRetries: 2, RetryBaseDelay: time.Millisecond, RetryMaxDelay: 5 * time.Millisecond, RetryAfterMaxDelay: maxHintWait}
					} else {
						model = &ChatClient{HTTPClient: client, ModelName: "fixture", MaxRetries: 2, RetryBaseDelay: time.Millisecond, RetryMaxDelay: 5 * time.Millisecond, RetryAfterMaxDelay: maxHintWait}
					}
					ctx, cancel := context.WithTimeout(context.Background(), tc.deadline)
					defer cancel()
					req := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("fixture")}}}
					var err error
					if stream {
						var events <-chan llm.StreamEvent
						events, err = model.InvokeStream(ctx, req)
						if err == nil {
							for event := range events {
								if e, ok := event.(llm.StreamErrorEvent); ok {
									err = e.Err
								}
							}
						}
					} else {
						_, err = model.Invoke(ctx, req)
					}
					if tc.stop {
						var stop interface{ AutomaticRetryStopped() bool }
						if !errors.As(err, &stop) || !stop.AutomaticRetryStopped() {
							t.Fatalf("expected typed stop, sends=%d err=%v", len(sends), err)
						}
						if len(sends) != 1 {
							t.Fatalf("stopped retry dispatched %d requests", len(sends))
						}
					} else {
						want := 2
						if tc.name == "success_once" {
							want = 1
						}
						if err != nil || len(sends) != want {
							t.Fatalf("sends=%d err=%v", len(sends), err)
						}
						if tc.hint == "0.025" && sends[1].Sub(sends[0]) < 25*time.Millisecond {
							t.Fatalf("physical send before hint: %v", sends[1].Sub(sends[0]))
						}
						if !notBefore.IsZero() && sends[1].Before(notBefore) {
							t.Fatalf("HTTP-date physical send before floor: %v < %v", sends[1], notBefore)
						}
					}
				})
			}
		}
	}
}

func TestRetryAfterCancellationDoesNotRedispatch(t *testing.T) {
	for _, cancelAfterSend := range []bool{false, true} {
		t.Run(map[bool]string{false: "before_send", true: "after_response"}[cancelAfterSend], func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			sends := 0
			client := &ChatClient{ModelName: "fixture", MaxRetries: 3, HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				sends++
				cancel()
				return &http.Response{StatusCode: 429, Header: http.Header{"Retry-After": []string{"0.1"}}, Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
			})}}
			if !cancelAfterSend {
				cancel()
			}
			_, err := client.Invoke(ctx, llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("fixture")}})
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel error=%v", err)
			}
			want := 0
			if cancelAfterSend {
				want = 1
			}
			if sends != want {
				t.Fatalf("sends=%d want %d", sends, want)
			}
		})
	}
}

func TestRetryAfterDurationBoundaries(t *testing.T) {
	const largest = time.Duration(1<<63 - 1)
	if got := parseRetryAfter("999999999999999999999999999999999999"); got != largest {
		t.Fatalf("huge floor wrapped: %v", got)
	}
	if got := parseRetryAfter("0.0000000001"); got != time.Nanosecond {
		t.Fatalf("subnanosecond floor rounded early: %v", got)
	}
	if got := resolveRetryDelay(0, time.Second, time.Second, largest); got != largest {
		t.Fatalf("jitter overflow: %v", got)
	}
	for _, invalid := range []string{"1m", "no-date", ".", "-1"} {
		if got := parseRetryAfter(invalid); got != 0 {
			t.Fatalf("invalid hint %q=%v", invalid, got)
		}
	}
}
