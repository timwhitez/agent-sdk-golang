package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

const agentGatewayToolChoiceRejection = `{"error":{"code":"invalid_request_error","message":"Thinking mode does not support this tool_choice (request_id: req-fixture)","type":"invalid_request_error"}}`

// requireDoneGateway scripts a reasoning-model gateway that rejects any forced
// tool_choice. Accepted requests answer in order: a "work" call, then text,
// then (for every later accepted request) either text again or a "done" call.
type requireDoneGateway struct {
	api        string
	laterReply string // "text" or "done"

	mu       sync.Mutex
	accepted int
	sequence []string // "forced" or "auto" per HTTP request
}

func (g *requireDoneGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	forced := false
	switch tc := body["tool_choice"].(type) {
	case string:
		forced = tc == "required"
	case map[string]any:
		forced = true
	}
	g.mu.Lock()
	if forced {
		g.sequence = append(g.sequence, "forced")
		g.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, agentGatewayToolChoiceRejection)
		return
	}
	g.sequence = append(g.sequence, "auto")
	g.accepted++
	n := g.accepted
	g.mu.Unlock()

	reply := "text"
	switch {
	case n == 1:
		reply = "work"
	case n >= 3:
		reply = g.laterReply
	}
	stream, _ := body["stream"].(bool)
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
	_, _ = io.WriteString(w, requireDoneGatewayBody(g.api, stream, reply, n))
}

func requireDoneGatewayBody(api string, stream bool, reply string, n int) string {
	callName := reply // "work" or "done"
	callID := "call_" + reply
	switch {
	case api == "chat" && stream && reply == "text":
		return "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"finished\"}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	case api == "chat" && stream:
		return "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"" + callID + "\",\"type\":\"function\",\"function\":{\"name\":\"" + callName + "\",\"arguments\":\"{}\"}}]}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n"
	case api == "chat" && reply == "text":
		return `{"id":"chat","choices":[{"index":0,"message":{"role":"assistant","content":"finished"},"finish_reason":"stop"}]}`
	case api == "chat":
		return `{"id":"chat","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"` + callID + `","type":"function","function":{"name":"` + callName + `","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`
	case stream && reply == "text":
		return "data: {\"type\":\"response.output_text.delta\",\"delta\":\"finished\"}\n\ndata: [DONE]\n\n"
	case stream:
		return "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fc_" + reply + "\",\"call_id\":\"" + callID + "\",\"type\":\"function_call\",\"name\":\"" + callName + "\",\"arguments\":\"{}\"}}\n\ndata: [DONE]\n\n"
	case reply == "text":
		return `{"id":"resp","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"finished"}]}]}`
	default:
		return `{"id":"resp","status":"completed","output":[{"id":"fc_` + reply + `","call_id":"` + callID + `","type":"function_call","name":"` + callName + `","arguments":"{}"}]}`
	}
}

// The RequireDone recovery forces tool_choice; a reasoning-model gateway that
// rejects forced tool_choice must not end the run with a runtime error. The
// client downgrades that request to auto, and the Agent either receives the
// done call or reaches its existing bounded safety fallback.
func TestAgentRequireDoneSurvivesForcedToolChoiceRejection(t *testing.T) {
	t.Parallel()
	for _, api := range []string{"chat", "responses"} {
		for _, later := range []string{"done", "text"} {
			t.Run(api+"_"+later, func(t *testing.T) {
				t.Parallel()
				gw := &requireDoneGateway{api: api, laterReply: later}
				server := httptest.NewServer(gw)
				defer server.Close()

				var model llm.ChatModel
				if api == "chat" {
					model = &openai.ChatClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 3, Warningf: func(string, ...any) {}}
				} else {
					model = &openai.ResponsesClient{BaseURL: server.URL, ModelName: "test-model", MaxRetries: 3, Warningf: func(string, ...any) {}}
				}
				var workCalls, doneCalls atomic.Int32
				work := tools.Func[struct{}]("work", "fixture work", func(context.Context, struct{}, *tools.Container) (any, error) {
					workCalls.Add(1)
					return "work result", nil
				})
				done := tools.Func[struct{}]("done", "fixture done", func(context.Context, struct{}, *tools.Container) (any, error) {
					doneCalls.Add(1)
					return nil, tools.TaskComplete("all done")
				})
				ag, err := agent.New(agent.Config{LLM: model, Tools: []tools.Tool{work, done}, RequireDoneTool: true, MaxIterations: 20, Warningf: func(string, ...any) {}})
				if err != nil {
					t.Fatal(err)
				}

				var errs []agent.ErrorEvent
				var final *agent.FinalResponseEvent
				safety := 0
				for ev := range ag.QueryStream(context.Background(), llm.TextContent("do the task")) {
					switch e := ev.(type) {
					case agent.ErrorEvent:
						errs = append(errs, e)
					case agent.FinalResponseEvent:
						f := e
						final = &f
					case agent.WarnEvent:
						if e.Kind == "require_done_safety" {
							safety++
						}
					}
				}
				if len(errs) != 0 {
					t.Fatalf("run ended with error events: %#v", errs)
				}
				if final == nil {
					t.Fatal("run produced no final response")
				}
				if workCalls.Load() != 1 {
					t.Fatalf("work calls = %d, want 1", workCalls.Load())
				}

				gw.mu.Lock()
				seq := append([]string(nil), gw.sequence...)
				gw.mu.Unlock()
				// Each forced request is rejected once and immediately retried as auto.
				forced := 0
				for i, kind := range seq {
					if kind != "forced" {
						continue
					}
					forced++
					if i+1 >= len(seq) || seq[i+1] != "auto" {
						t.Fatalf("forced request %d not followed by auto retry: %v", i, seq)
					}
				}
				switch later {
				case "done":
					if doneCalls.Load() != 1 || safety != 0 || forced != 1 {
						t.Fatalf("done calls=%d safety=%d forced=%d seq=%v; want 1/0/1", doneCalls.Load(), safety, forced, seq)
					}
				case "text":
					if doneCalls.Load() != 0 || safety != 1 || forced != 2 {
						t.Fatalf("done calls=%d safety=%d forced=%d seq=%v; want 0/1/2", doneCalls.Load(), safety, forced, seq)
					}
					if final.Content != "finished" {
						t.Fatalf("final = %q, want finished", final.Content)
					}
				}
			})
		}
	}
}
