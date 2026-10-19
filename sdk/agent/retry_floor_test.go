package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
)

type retryFloorTransport func(*http.Request) (*http.Response, error)

func (f retryFloorTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type bufferedRetryFloorModel struct{ llm.ChatModel }

func TestAgentRetryAfterStopsPhysicalDispatch(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for _, providerAttempts := range []int{1, 2} {
				t.Run(fmt.Sprintf("responses=%t/stream=%t/providerAttempts=%d", responses, stream, providerAttempts), func(t *testing.T) {
					var sends atomic.Int32
					hint := "3"
					if providerAttempts == 2 {
						hint = "90"
					}
					client := &http.Client{Transport: retryFloorTransport(func(req *http.Request) (*http.Response, error) {
						n := sends.Add(1)
						status, body := 429, `{"error":{"message":"fixture"}}`
						h := http.Header{"Retry-After": []string{hint}}
						if n > 1 {
							status = 200
							if responses {
								body = `{"id":"r","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`
							} else {
								body = `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
							}
							if stream {
								if responses {
									body = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":" + body + "}\n\n"
								} else {
									body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
								}
							}
						}
						return &http.Response{StatusCode: status, Header: h, Body: io.NopCloser(strings.NewReader(body)), Request: req}, nil
					})}
					var model llm.ChatModel
					if responses {
						model = &openai.ResponsesClient{HTTPClient: client, ModelName: "fixture", MaxRetries: providerAttempts, RetryBaseDelay: time.Millisecond, RetryMaxDelay: 5 * time.Millisecond}
					} else {
						model = &openai.ChatClient{HTTPClient: client, ModelName: "fixture", MaxRetries: providerAttempts, RetryBaseDelay: time.Millisecond, RetryMaxDelay: 5 * time.Millisecond}
					}
					if !stream {
						model = bufferedRetryFloorModel{model}
					}
					ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 3})
					if err != nil {
						t.Fatal(err)
					}
					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
					defer cancel()
					_, err = ag.Query(ctx, "fixture")
					var stop interface{ AutomaticRetryStopped() bool }
					if !errors.As(err, &stop) || !stop.AutomaticRetryStopped() {
						t.Fatalf("typed stop lost through Agent.Query: sends=%d err=%v", sends.Load(), err)
					}
					if sends.Load() != 1 {
						t.Fatalf("automatic outer dispatches after stop: %d", sends.Load()-1)
					}
					var cause *llm.RateLimitError
					if !errors.As(err, &cause) || cause.RetryAfter < 3*time.Second {
						t.Fatalf("original provider floor/cause lost: %v", err)
					}
				})
			}
		}
	}
}

func TestRetryStopEventAndSummaryOwners(t *testing.T) {
	cause := &llm.RateLimitError{RetryAfter: 90 * time.Second, Message: "fixture-body"}
	stop := llm.BoundRetryAfter(context.Background(), cause, time.Second)
	event := errEventForModel(stop, nil)
	if event.Kind != "retry_stopped" || event.RetryStop == nil || event.RetryAfterMS != 90000 {
		t.Fatalf("stop lost in native stream event: %#v", event)
	}
	wire, err := json.Marshal(event)
	if err != nil || strings.Contains(string(wire), "fixture-body") || strings.Contains(string(wire), "RetryStop") {
		t.Fatalf("native error leaked into wire: %s %v", wire, err)
	}
	for _, overflow := range []bool{false, true} {
		for _, alreadyTyped := range []bool{false, true} {
			t.Run(fmt.Sprintf("overflow=%t/typed=%t", overflow, alreadyTyped), func(t *testing.T) {
				failure := error(cause)
				if alreadyTyped {
					failure = stop
				}
				model := &transientInvokeModel{failFor: 100, err: failure}
				messages := []llm.Message{llm.NewSystemMessage("base"), llm.NewUserMessage(strings.Repeat("constraint ", 3000))}
				ag, err := New(Config{LLM: model, InitialMessages: messages, Compaction: &compaction.Config{Enabled: true, ContextWindow: 100, CompactionRetries: 3, CompactionRetryBackoff: time.Millisecond}})
				if err != nil {
					t.Fatal(err)
				}
				if overflow {
					_, _, err = ag.compactOverflowWithPlan(context.Background(), messages, nil, true, 4)
				} else {
					_, err = ag.CompactNow(context.Background())
				}
				if !llm.IsRetryStop(err) || model.calls != 1 || ag.hasPendingCompaction() {
					t.Fatalf("summary retried or lost stop: calls=%d err=%v", model.calls, err)
				}
			})
		}
	}
}
