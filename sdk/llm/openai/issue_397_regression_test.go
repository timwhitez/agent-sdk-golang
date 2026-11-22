package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestMergedToolArgumentsAgreeWithOpaqueReplay(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, changed := range []bool{false, true} {
			t.Run(fmt.Sprintf("streaming_%t_changed_%t", streaming, changed), func(t *testing.T) {
				var mu sync.Mutex
				var requests [][]byte
				var finalState []json.RawMessage
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					body, err := io.ReadAll(r.Body)
					if err != nil {
						http.Error(w, err.Error(), http.StatusBadRequest)
						return
					}
					mu.Lock()
					requests = append(requests, body)
					n := len(requests)
					mu.Unlock()
					if n > 2 {
						if streaming {
							w.Header().Set("Content-Type", "text/event-stream")
							fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output_text\":\"done\"}}\n\n")
						} else {
							fmt.Fprint(w, `{"status":"completed","output_text":"done"}`)
						}
						return
					}
					args, itemID, status, eventType, details := `{"left":1}`, "fc_partial", "incomplete", "response.incomplete", `,"incomplete_details":{"reason":"max_output_tokens"}`
					if n == 2 {
						args, itemID, status, eventType, details = `{"left":1,"right":2}`, "fc_final", "completed", "response.completed", ""
						if changed {
							args = `{"right":2}`
						}
					}
					item := fmt.Sprintf(`{"id":%q,"type":"function_call","call_id":"call_merge","name":"combine","arguments":%q,"phase":"commentary"}`, itemID, args)
					message := fmt.Sprintf(`{"id":"msg_%d","type":"message","role":"assistant","content":[{"type":"output_text","text":"fragment %d"}]}`, n, n)
					response := fmt.Sprintf(`{"id":"resp_%d","status":%q,"output":[%s,%s]%s}`, n, status, message, item, details)
					if n == 2 {
						mu.Lock()
						finalState = []json.RawMessage{json.RawMessage(message), json.RawMessage(item)}
						mu.Unlock()
					}
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprintf(w, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":%s}\n\ndata: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":%s}\n\ndata: {\"type\":%q,\"response\":%s}\n\n", message, item, eventType, response)
					} else {
						w.Header().Set("Content-Type", "application/json")
						fmt.Fprint(w, response)
					}
				}))
				defer server.Close()
				client := &openai.ResponsesClient{BaseURL: server.URL, HTTPClient: server.Client(), ModelName: "fixture", MaxRetries: 1}
				var model llm.ChatModel = client
				if !streaming {
					model = invokeOnlyResponsesClient{client: client}
				}
				type args struct {
					Left  int `json:"left"`
					Right int `json:"right"`
				}
				var executed []args
				tool := tools.Func[args]("combine", "combine", func(_ context.Context, a args, _ *tools.Container) (any, error) {
					executed = append(executed, a)
					return "combined", nil
				})
				ag, err := agent.New(agent.Config{LLM: model, Tools: []tools.Tool{tool}, MaxIterations: 4, InvokeRetryMaxAttempts: 1})
				if err != nil {
					t.Fatal(err)
				}
				if result, err := ag.Query(t.Context(), "combine both parts"); err != nil || result != "done" {
					t.Fatalf("query=%q err=%v", result, err)
				}
				if len(executed) != 1 || executed[0] != (args{Left: 1, Right: 2}) {
					t.Fatalf("tool did not execute merged arguments: %#v", executed)
				}
				mu.Lock()
				defer mu.Unlock()
				accepted := 0
				for _, message := range ag.Messages() {
					if len(message.ToolCalls) == 0 {
						continue
					}
					accepted++
					var stored args
					if len(message.ToolCalls) != 1 || json.Unmarshal([]byte(message.ToolCalls[0].Function.Arguments), &stored) != nil || stored != executed[0] {
						t.Fatalf("generic history disagrees with executed call: %+v", message.ToolCalls)
					}
					if message.PlainText() != "fragment 2" {
						t.Fatalf("merge lost visible final-fragment text: %q", message.PlainText())
					}
					state, err := llm.ProviderStateFromContent(message.Content)
					if err != nil {
						t.Fatal(err)
					}
					if changed {
						if len(state) != 0 {
							t.Error("changed calls retained opaque provider state")
						}
					} else {
						if len(state) != len(finalState) {
							t.Fatalf("unchanged state count=%d, want %d", len(state), len(finalState))
						}
						for i := range state {
							if !bytes.Equal(state[i].Data, finalState[i]) {
								t.Error("unchanged merge rewrote opaque state bytes")
							}
						}
					}
				}
				if accepted != 1 {
					t.Fatalf("accepted tool histories=%d, want 1", accepted)
				}
				if len(requests) != 3 {
					t.Fatalf("HTTP requests=%d, want 3", len(requests))
				}
				var body struct {
					Input []struct {
						Type, ID, Phase, Arguments string
						CallID                     string `json:"call_id"`
					} `json:"input"`
				}
				if err := json.Unmarshal(requests[2], &body); err != nil {
					t.Fatal(err)
				}
				calls, outputs := 0, 0
				for _, item := range body.Input {
					if item.Type == "function_call_output" && item.CallID == "call_merge" {
						outputs++
					}
					if item.Type != "function_call" || item.CallID != "call_merge" {
						continue
					}
					calls++
					var replay args
					if err := json.Unmarshal([]byte(item.Arguments), &replay); err != nil || replay != executed[0] {
						t.Errorf("replayed arguments disagree with executed call: replay=%q executed=%+v err=%v", item.Arguments, executed[0], err)
					}
					if !changed && (item.ID != "fc_final" || item.Phase != "commentary") {
						t.Error("unchanged final call lost opaque provider fields")
					}
				}
				if calls != 1 || outputs != 1 {
					t.Fatalf("replayed native calls/outputs=%d/%d, want 1/1", calls, outputs)
				}
			})
		}
	}
}
