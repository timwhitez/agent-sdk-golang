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

func TestStreamOutcomeAllowedBackoffChildCancellation(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, hintKind := range []string{"numeric", "date", "no_hint"} {
			t.Run(fmt.Sprintf("responses=%t/%s", responses, hintKind), func(t *testing.T) {
				var calls atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls.Add(1)
					hint := "30"
					if hintKind == "date" {
						hint = time.Now().Add(30 * time.Second).UTC().Format(http.TimeFormat)
					}
					if hintKind != "no_hint" {
						w.Header().Set("Retry-After", hint)
					}
					w.Header().Set("Content-Type", "application/json")
					if hintKind == "no_hint" {
						w.WriteHeader(500)
					} else {
						w.WriteHeader(429)
					}
					fmt.Fprint(w, `{"error":{"message":"non-secret allowed-wait fixture"}}`)
				}))
				defer server.Close()
				parent, parentCancel := context.WithCancel(context.Background())
				defer parentCancel()
				child, cancelChild := context.WithCancel(parent)
				defer cancelChild()
				ctx, outcome := llm.WithStreamOutcome(child)
				type entry struct {
					cause     error
					delay     time.Duration
					before    llm.StreamOutcomeSnapshot
					childLive bool
				}
				entered := make(chan entry, 1)
				hook := &retryBackoffTestHooks{waiting: func(cause error, delay time.Duration) {
					entered <- entry{cause: cause, delay: delay, before: outcome.Snapshot(), childLive: child.Err() == nil}
					// Actual timer has been admitted/created. Cancel this child only,
					// after a distinct interval from the original parse publication.
					time.Sleep(20 * time.Millisecond)
					cancelChild()
				}}
				ctx = context.WithValue(ctx, retryBackoffTestHookKey{}, hook)
				var model llm.StreamingChatModel = &ChatClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "fixture", MaxRetries: 2, RetryBaseDelay: time.Second, RetryMaxDelay: time.Second, RetryAfterMaxDelay: time.Minute}
				if responses {
					model = &ResponsesClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "fixture", MaxRetries: 2, RetryBaseDelay: time.Second, RetryMaxDelay: time.Second, RetryAfterMaxDelay: time.Minute}
				}
				ch, err := model.InvokeStream(ctx, llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("local fixture")}})
				if err != nil {
					t.Fatal(err)
				}
				var observed entry
				select {
				case observed = <-entered:
				case <-time.After(3 * time.Second):
					t.Fatal("allowed internal backoff not reached")
				}
				for range ch {
				}
				got := outcome.Snapshot()
				hint := llm.RetryAfterForError(observed.cause)
				if !observed.childLive || parent.Err() != nil || !errors.Is(child.Err(), context.Canceled) || calls.Load() != 1 || llm.IsRetryStop(observed.cause) || !got.Started || !got.Complete {
					t.Fatal("actual allowed-wait / child-only cancellation / unique-send / completion premise absent")
				}
				if hintKind == "no_hint" {
					if hint != 0 || got.RetryAfter != 0 || !got.RetryAfterNotBefore.IsZero() || llm.IsRetryStop(got.Error) {
						t.Fatal("genuine no-hint cancellation invented a floor/stop")
					}
					return
				}
				if hint < 25*time.Second || observed.delay < hint {
					t.Fatal("parsed long hint was not admitted as the actual wait floor")
				}
				unchanged := got.RetryAfterNotBefore == observed.before.RetryAfterNotBefore
				t.Logf("actual parsed_hint=%s allowed_timer=%s child_canceled=%t parent_live=%t pre_wait_hint=%s completed_hint=%s floor_unchanged=%t", hint, observed.delay, child.Err() != nil, parent.Err() == nil, observed.before.RetryAfter, got.RetryAfter, unchanged)
				if observed.before.RetryAfter != hint || observed.before.Error != observed.cause || observed.before.RetryAfterNotBefore.IsZero() || got.RetryAfter != hint || got.Error != observed.cause || !unchanged || !got.RetryAfterNotBefore.After(time.Now().Add(10*time.Second)) {
					t.Fatal("original parsed cause/floor lost before allowed wait or restarted after child cancellation")
				}
			})
		}
	}
}
