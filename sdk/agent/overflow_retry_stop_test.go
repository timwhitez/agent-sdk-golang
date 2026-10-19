package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
)

// Drive the real Query/stream -> HTTP overflow -> synchronous summary HTTP
// failure chain. A stopped summary is the terminal failure, not the rejected
// main request, and must never trigger emergency history trimming.
func TestOverflowSummaryRetryStopReachesQueryAndStream(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, streamModel := range []bool{false, true} {
			for _, streamEntry := range []bool{false, true} {
				t.Run(fmt.Sprintf("responses=%t/modelStream=%t/entryStream=%t", responses, streamModel, streamEntry), func(t *testing.T) {
					var sends atomic.Int32
					client := &http.Client{Transport: retryFloorTransport(func(req *http.Request) (*http.Response, error) {
						body, _ := io.ReadAll(req.Body)
						n := sends.Add(1)
						status, response := 400, `{"error":{"message":"fixture main overflow","type":"invalid_request_error","code":"context_length_exceeded"}}`
						h := http.Header{}
						if n == 1 && strings.Contains(string(body), "operational checkpoint") {
							t.Error("fixture compacted before main overflow")
						}
						if n == 2 {
							if !strings.Contains(string(body), "operational checkpoint") {
								t.Error("second physical request was not summary")
							}
							status, response = 429, `{"error":{"message":"fixture summary rate limit"}}`
							h.Set("Retry-After", "90")
						}
						if n > 2 {
							status, response = 400, `{"error":{"message":"unexpected extra dispatch"}}`
						}
						return &http.Response{StatusCode: status, Header: h, Body: io.NopCloser(strings.NewReader(response)), Request: req}, nil
					})}
					var model llm.ChatModel
					if responses {
						model = &openai.ResponsesClient{HTTPClient: client, ModelName: "fixture", MaxRetries: 2}
					} else {
						model = &openai.ChatClient{HTTPClient: client, ModelName: "fixture", MaxRetries: 2}
					}
					if !streamModel {
						model = bufferedRetryFloorModel{model}
					}
					ag := overflowAgent(t, model, nil)
					before := ag.Messages()
					// This bounded query deadline is shorter than the 90s floor,
					// so the stop must preserve the provider's deadline reason.
					ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
					defer cancel()
					var terminal error
					if streamEntry {
						errorsSeen := 0
						for env := range ag.QueryStreamEnveloped(ctx, llm.TextContent("fixture current input")) {
							switch e := env.Event.(type) {
							case ErrorEvent:
								errorsSeen++
								terminal = e.RetryStop
								if e.Kind != "retry_stopped" || e.RetryStop == nil || e.RetryAfterMS != 90000 || env.Origin != EventOriginCompaction {
									t.Errorf("summary terminal provenance lost: event=%+v origin=%s", e, env.Origin)
								}
								wire, err := json.Marshal(e)
								if err != nil || strings.Contains(string(wire), "fixture summary rate limit") {
									t.Errorf("native cause leaked to serialized event: %s %v", wire, err)
								}
							case CompactionEvent:
								t.Error("stopped summary published compaction/emergency trim")
							case FinalResponseEvent:
								t.Error("stopped summary emitted final response")
							}
						}
						if errorsSeen != 1 {
							t.Errorf("terminal errors=%d", errorsSeen)
						}
					} else {
						_, terminal = ag.Query(ctx, "fixture current input")
					}
					var stop *llm.RetryStopError
					var cause *llm.RateLimitError
					if !errors.As(terminal, &stop) || stop == nil || stop.RetryAfter != 90*time.Second || stop.Reason != "deadline" || !errors.As(terminal, &cause) || cause.RetryAfter != 90*time.Second {
						t.Errorf("summary typed stop/floor/cause lost: %v", terminal)
					}
					if sends.Load() != 2 {
						t.Errorf("physical sends=%d want main+summary only", sends.Load())
					}
					after := ag.Messages()
					if len(after) != len(before)+1 || !reflect.DeepEqual(before, after[:len(before)]) {
						t.Fatal("stopped summary changed prior history")
					}
				})
			}
		}
	}
}
