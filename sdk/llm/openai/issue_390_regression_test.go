package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// These wrappers interrupt the real adapter only after it has produced its
// terminal output state and before the Agent can accept the completion.
type auditInterruptedResponses struct {
	client          *openai.ResponsesClient
	boundary        func()
	mode            string
	calls           atomic.Int32
	deltas          atomic.Int32
	stateBoundaries atomic.Int32
}

func (m *auditInterruptedResponses) Provider() string { return m.client.Provider() }
func (m *auditInterruptedResponses) Model() string    { return m.client.Model() }
func (m *auditInterruptedResponses) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	comp, err := m.client.Invoke(ctx, req)
	if m.calls.Add(1) == 1 && err == nil {
		m.boundary()
		if m.mode == "buffered_error" {
			return comp, errors.New("fixture terminal invocation failure")
		}
	}
	return comp, err
}

type auditInterruptedResponsesStream struct{ base *auditInterruptedResponses }

func (m auditInterruptedResponsesStream) Provider() string { return m.base.Provider() }
func (m auditInterruptedResponsesStream) Model() string    { return m.base.Model() }
func (m auditInterruptedResponsesStream) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	return m.base.Invoke(ctx, req)
}
func (m auditInterruptedResponsesStream) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	source, err := m.base.client.InvokeStream(ctx, req)
	if err != nil || m.base.calls.Add(1) != 1 {
		return source, err
	}
	out := make(chan llm.StreamEvent)
	go func() {
		defer close(out)
		defer func() {
			for range source {
			}
		}()
		for event := range source {
			if _, ok := event.(llm.StreamToolCallDeltaEvent); ok {
				m.base.deltas.Add(1)
			}
			select {
			case out <- event:
			case <-ctx.Done():
				return
			}
			if _, ok := event.(llm.StreamProviderStateEvent); !ok {
				continue
			}
			m.base.stateBoundaries.Add(1)
			m.base.boundary()
			switch m.base.mode {
			case "stream_error":
				out <- llm.StreamErrorEvent{Err: errors.New("fixture error after provider state")}
			case "missing_done":
			default:
				<-ctx.Done()
			}
			return
		}
	}()
	return out, nil
}

func TestAuditUnacceptedResponsesToolStateDoesNotReplayNativeCall(t *testing.T) {
	for _, mode := range []string{"stream_error", "missing_done", "stream_cancel", "stream_steering", "stream_idle", "buffered_error", "buffered_cancel", "buffered_steering"} {
		t.Run(mode, func(t *testing.T) {
			var mu sync.Mutex
			var requests [][]byte
			streaming := mode != "buffered_error" && mode != "buffered_cancel" && mode != "buffered_steering"
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				mu.Lock()
				requests = append(requests, body)
				call := len(requests)
				mu.Unlock()
				if call == 1 {
					if streaming {
						writeOpaqueResponsesSSE(w, 1)
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, _ = io.WriteString(w, firstOpaqueResponsesJSON())
					}
					return
				}
				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"follow-up\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"after\",\"status\":\"completed\",\"output\":[]}}\n\n")
				} else {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"id":"after","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"follow-up"}]}]}`)
				}
			}))
			defer server.Close()
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			steering := make(chan agent.SteeringMsg, 1)
			var ag *agent.Agent
			wrapped := &auditInterruptedResponses{
				client: &openai.ResponsesClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "gpt-test", MaxRetries: 1},
				mode:   mode,
			}
			wrapped.boundary = func() {
				switch mode {
				case "stream_cancel", "buffered_cancel":
					cancel()
				case "stream_steering", "buffered_steering":
					steering <- agent.SteeringMsg{Content: "change direction"}
					if !streaming {
						ag.InterruptActiveStageForSteering()
					}
				}
			}
			var model llm.ChatModel = wrapped
			if streaming {
				model = auditInterruptedResponsesStream{base: wrapped}
			}
			var toolRuns atomic.Int32
			// Only the idle-specific fixture should trigger idle recovery. A
			// short timeout in the other modes can cancel the first invocation
			// before its HTTP request reaches the server and desynchronize the
			// wrapper's first-call boundary from the server's first response.
			var idleTimeout time.Duration
			idleRecoveries := 0
			if mode == "stream_idle" {
				idleTimeout = time.Second
				idleRecoveries = 1
			}
			var err error
			ag, err = agent.New(agent.Config{
				LLM: model, MaxIterations: 4, InvokeRetryMaxAttempts: 1,
				StreamIdleTimeout: idleTimeout, StreamIdleMaxRecoveries: idleRecoveries,
				Tools: []tools.Tool{tools.Func[struct {
					Message string `json:"message"`
				}]("echo", "fixture", func(context.Context, struct {
					Message string `json:"message"`
				}, *tools.Container) (any, error) {
					toolRuns.Add(1)
					return "must not execute", nil
				})},
			})
			if err != nil {
				t.Fatal(err)
			}
			observedIdleRecoveries := 0
			for event := range ag.QueryStreamWithSteering(ctx, llm.TextContent("start"), steering) {
				if warning, ok := event.(agent.WarnEvent); ok && warning.Kind == "stream_idle_recovery" {
					observedIdleRecoveries++
				}
			}
			if observedIdleRecoveries != idleRecoveries {
				t.Fatalf("idle recoveries=%d, want %d for mode %s", observedIdleRecoveries, idleRecoveries, mode)
			}
			if streaming && wrapped.stateBoundaries.Load() != 1 {
				t.Fatalf("fixture reached %d state-to-Done boundaries, want one", wrapped.stateBoundaries.Load())
			}
			continues := mode == "stream_steering" || mode == "buffered_steering" || mode == "stream_idle"
			if !continues {
				if _, err := ag.Query(t.Context(), "try again"); err != nil {
					t.Fatal(err)
				}
			}
			if toolRuns.Load() != 0 {
				t.Fatalf("unaccepted call executed %d times", toolRuns.Load())
			}
			if streaming && wrapped.deltas.Load() == 0 {
				t.Fatal("fixture did not emit tool-call metadata before opaque state")
			}
			mu.Lock()
			defer mu.Unlock()
			if len(requests) != 2 {
				t.Fatalf("requests=%d, want initial and resumed requests", len(requests))
			}
			var payload struct {
				Input []struct {
					Type   string `json:"type"`
					CallID string `json:"call_id"`
				} `json:"input"`
			}
			if err := json.Unmarshal(requests[1], &payload); err != nil {
				t.Fatal(err)
			}
			nativeCalls, nativeResults := 0, 0
			for _, item := range payload.Input {
				if item.Type == "function_call" {
					nativeCalls++
				}
				if item.Type == "function_call_output" {
					nativeResults++
				}
			}
			if nativeCalls != 0 || nativeResults != 0 {
				t.Fatalf("unaccepted provider state replayed %d native function_call items and %d outputs despite zero executions", nativeCalls, nativeResults)
			}
		})
	}
}
