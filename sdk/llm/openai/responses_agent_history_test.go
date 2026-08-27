package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type invokeOnlyResponsesClient struct {
	client *openai.ResponsesClient
}

func (m invokeOnlyResponsesClient) Provider() string { return m.client.Provider() }
func (m invokeOnlyResponsesClient) Model() string    { return m.client.Model() }
func (m invokeOnlyResponsesClient) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	return m.client.Invoke(ctx, req)
}

func TestAgentReplaysOpaqueResponsesItemsAcrossToolRoundTrip(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		streaming := streaming
		t.Run(map[bool]string{false: "buffered", true: "streaming"}[streaming], func(t *testing.T) {
			var (
				mu       sync.Mutex
				requests [][]byte
			)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				_ = r.Body.Close()
				mu.Lock()
				requests = append(requests, append([]byte(nil), body...))
				call := len(requests)
				mu.Unlock()

				if call == 2 {
					if err := validateOpaqueResponsesContinuation(body); err != nil {
						http.Error(w, err.Error(), http.StatusBadRequest)
						return
					}
				}
				if call > 2 {
					http.Error(w, fmt.Sprintf("unexpected request %d", call), http.StatusBadRequest)
					return
				}

				if streaming {
					writeOpaqueResponsesSSE(w, call)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				if call == 1 {
					_, _ = io.WriteString(w, firstOpaqueResponsesJSON())
				} else {
					_, _ = io.WriteString(w, secondOpaqueResponsesJSON())
				}
			}))
			defer server.Close()

			client := &openai.ResponsesClient{
				HTTPClient:      server.Client(),
				BaseURL:         server.URL,
				ModelName:       "gpt-test",
				ReasoningEffort: "max",
				MaxRetries:      1,
			}
			var model llm.ChatModel = client
			if !streaming {
				model = invokeOnlyResponsesClient{client: client}
			}
			echoTool := tools.Func[struct {
				Message string `json:"message"`
			}]("echo", "echo", func(_ context.Context, args struct {
				Message string `json:"message"`
			}, _ *tools.Container) (any, error) {
				return args.Message, nil
			})
			doneTool := tools.Func[struct {
				Message string `json:"message"`
			}]("done", "finish", func(_ context.Context, args struct {
				Message string `json:"message"`
			}, _ *tools.Container) (any, error) {
				return nil, tools.TaskComplete(args.Message)
			})
			ag, err := agent.New(agent.Config{
				LLM:             model,
				Tools:           []tools.Tool{echoTool, doneTool},
				ToolChoice:      llm.ToolChoice("auto"),
				RequireDoneTool: true,
				MaxIterations:   4,
			})
			if err != nil {
				t.Fatalf("agent.New: %v", err)
			}

			final := ""
			for event := range ag.QueryStream(context.Background(), llm.TextContent("do the work")) {
				switch event := event.(type) {
				case agent.FinalResponseEvent:
					final = event.Content
				case agent.ErrorEvent:
					t.Fatalf("agent error: %#v", event)
				}
			}
			if final != "finished" {
				t.Fatalf("final response = %q, want finished", final)
			}
			mu.Lock()
			requestCount := len(requests)
			mu.Unlock()
			if requestCount != 2 {
				t.Fatalf("request count = %d, want 2", requestCount)
			}
		})
	}
}

func TestAgentMaxTokenToolContinuationReplaysAllOpaqueItems(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		streaming := streaming
		t.Run(map[bool]string{false: "buffered", true: "streaming"}[streaming], func(t *testing.T) {
			var (
				mu       sync.Mutex
				requests [][]byte
			)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				mu.Lock()
				requests = append(requests, append([]byte(nil), body...))
				call := len(requests)
				mu.Unlock()
				if call == 2 {
					if err := validateMaxTokenOpaqueContinuation(body); err != nil {
						http.Error(w, err.Error(), http.StatusBadRequest)
						return
					}
				}
				if call > 2 {
					http.Error(w, fmt.Sprintf("unexpected request %d", call), http.StatusBadRequest)
					return
				}
				if streaming {
					writeMaxTokenContinuationSSE(w, call)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				if call == 1 {
					_, _ = io.WriteString(w, maxTokenPartialResponsesJSON())
				} else {
					_, _ = io.WriteString(w, maxTokenFinalResponsesJSON())
				}
			}))
			defer server.Close()

			client := &openai.ResponsesClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "gpt-test", MaxRetries: 1}
			var model llm.ChatModel = client
			if !streaming {
				model = invokeOnlyResponsesClient{client: client}
			}
			doneTool := tools.Func[struct {
				Message string `json:"message"`
			}]("done", "finish", func(_ context.Context, args struct {
				Message string `json:"message"`
			}, _ *tools.Container) (any, error) {
				return nil, tools.TaskComplete(args.Message)
			})
			ag, err := agent.New(agent.Config{LLM: model, Tools: []tools.Tool{doneTool}, MaxIterations: 4})
			if err != nil {
				t.Fatal(err)
			}
			final, err := ag.Query(context.Background(), "finish")
			if err != nil || final != "finished" {
				t.Fatalf("query = %q, %v", final, err)
			}
			mu.Lock()
			requestCount := len(requests)
			mu.Unlock()
			if requestCount != 2 {
				t.Fatalf("request count = %d, want 2", requestCount)
			}
		})
	}
}

func TestAgentStreamingReplaysOpaqueStateWithoutVisibleDelta(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if calls == 1 {
			_, _ = io.WriteString(w, strings.Join([]string{
				`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_state_only","type":"reasoning","encrypted_content":"state-only-ciphertext","phase":"analysis","summary":[]}}`,
				"",
				`data: {"type":"response.completed","response":{"id":"resp_state_only","status":"completed","output":[{"id":"rs_state_only","type":"reasoning","encrypted_content":"state-only-ciphertext","phase":"analysis","summary":[]}]}}`,
				"",
			}, "\n"))
			return
		}
		if calls == 2 {
			if !strings.Contains(string(body), `"id":"rs_state_only"`) || !strings.Contains(string(body), `"encrypted_content":"state-only-ciphertext"`) {
				http.Error(w, "missing state-only reasoning item", http.StatusBadRequest)
				return
			}
			_, _ = io.WriteString(w, strings.Join([]string{
				`data: {"type":"response.completed","response":{"id":"resp_visible","status":"completed","output":[{"id":"msg_visible","type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}}`,
				"",
			}, "\n"))
			return
		}
		http.Error(w, fmt.Sprintf("unexpected request %d", calls), http.StatusBadRequest)
	}))
	defer server.Close()
	client := &openai.ResponsesClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "gpt-test", MaxRetries: 1}
	ag, err := agent.New(agent.Config{LLM: client, MaxIterations: 2})
	if err != nil {
		t.Fatal(err)
	}
	first, err := ag.Query(context.Background(), "first")
	if err != nil || first != "" {
		t.Fatalf("first query = %q, %v", first, err)
	}
	second, err := ag.Query(context.Background(), "second")
	if err != nil || second != "ok" {
		t.Fatalf("second query = %q, %v", second, err)
	}
	if calls != 2 {
		t.Fatalf("request count = %d, want 2", calls)
	}
}

func validateOpaqueResponsesContinuation(body []byte) error {
	var payload struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return fmt.Errorf("decode second request: %w", err)
	}
	reasoningIndex, callIndex, outputIndex := -1, -1, -1
	reasoningCount, callCount := 0, 0
	for index, raw := range payload.Input {
		var item map[string]any
		if err := json.Unmarshal(raw, &item); err != nil {
			return fmt.Errorf("decode input item %d: %w", index, err)
		}
		switch item["type"] {
		case "reasoning":
			reasoningCount++
			reasoningIndex = index
			if item["id"] != "rs_opaque_1" || item["encrypted_content"] != "encrypted-reasoning-sentinel" || item["phase"] != "analysis" {
				return fmt.Errorf("reasoning item was not replayed faithfully: %#v", item)
			}
		case "function_call":
			callCount++
			callIndex = index
			if item["id"] != "fc_opaque_1" || item["call_id"] != "call_opaque_1" || item["phase"] != "commentary" || item["arguments"] != `{"message":"work"}` {
				return fmt.Errorf("function-call item was not replayed faithfully: %#v", item)
			}
		case "function_call_output":
			if item["call_id"] == "call_opaque_1" {
				outputIndex = index
			}
		}
	}
	if reasoningCount != 1 || callCount != 1 {
		return fmt.Errorf("opaque output items were duplicated or omitted: reasoning=%d function_call=%d", reasoningCount, callCount)
	}
	if reasoningIndex < 0 || callIndex <= reasoningIndex || outputIndex <= callIndex {
		return fmt.Errorf("opaque item order is invalid: reasoning=%d call=%d output=%d", reasoningIndex, callIndex, outputIndex)
	}
	return nil
}

func validateMaxTokenOpaqueContinuation(body []byte) error {
	var payload struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return fmt.Errorf("decode continuation request: %w", err)
	}
	reasoningCount, callCount := 0, 0
	for _, raw := range payload.Input {
		var item map[string]any
		if err := json.Unmarshal(raw, &item); err != nil {
			return err
		}
		switch item["type"] {
		case "reasoning":
			reasoningCount++
			if item["id"] != "rs_partial" || item["encrypted_content"] != "partial-ciphertext" {
				return fmt.Errorf("reasoning item changed: %#v", item)
			}
		case "function_call":
			callCount++
			if item["id"] != "fc_partial" || item["call_id"] != "call_partial" || item["arguments"] != `{"message":"fin` {
				return fmt.Errorf("function-call item changed: %#v", item)
			}
		}
	}
	if reasoningCount != 1 || callCount != 1 {
		return fmt.Errorf("continuation replay counts = reasoning:%d function_call:%d", reasoningCount, callCount)
	}
	return nil
}

func firstOpaqueResponsesItems() string {
	return `[{"id":"rs_opaque_1","type":"reasoning","encrypted_content":"encrypted-reasoning-sentinel","phase":"analysis","summary":[]},{"id":"fc_opaque_1","call_id":"call_opaque_1","type":"function_call","name":"echo","arguments":"{\"message\":\"work\"}","phase":"commentary","status":"completed"}]`
}

func firstOpaqueResponsesJSON() string {
	return `{"id":"resp_opaque_1","status":"completed","output":` + firstOpaqueResponsesItems() + `,"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`
}

func secondOpaqueResponsesJSON() string {
	return `{"id":"resp_opaque_2","status":"completed","output":[{"id":"fc_done_1","call_id":"call_done_1","type":"function_call","name":"done","arguments":"{\"message\":\"finished\"}","status":"completed"}],"usage":{"input_tokens":20,"output_tokens":3,"total_tokens":23}}`
}

func maxTokenPartialItems() string {
	return `[{"id":"rs_partial","type":"reasoning","encrypted_content":"partial-ciphertext","summary":[]},{"id":"fc_partial","call_id":"call_partial","type":"function_call","name":"done","arguments":"{\"message\":\"fin","status":"in_progress"}]`
}

func maxTokenPartialResponsesJSON() string {
	return `{"id":"resp_partial","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":` + maxTokenPartialItems() + `}`
}

func maxTokenFinalResponsesJSON() string {
	return `{"id":"resp_final","status":"completed","output":[{"id":"fc_final","call_id":"call_partial","type":"function_call","name":"done","arguments":"ished\"}","status":"completed"}]}`
}

func writeOpaqueResponsesSSE(w http.ResponseWriter, call int) {
	w.Header().Set("Content-Type", "text/event-stream")
	if call == 1 {
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_opaque_1","type":"reasoning","encrypted_content":"encrypted-reasoning-sentinel","phase":"analysis","summary":[]}}`,
			"",
			`data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_opaque_1","call_id":"call_opaque_1","type":"function_call","name":"echo","arguments":"{\"message\":\"work\"}","phase":"commentary","status":"completed"}}`,
			"",
			`data: {"type":"response.completed","response":{"id":"resp_opaque_1","status":"completed","output":` + firstOpaqueResponsesItems() + `,"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`,
			"",
		}, "\n"))
		return
	}
	_, _ = io.WriteString(w, strings.Join([]string{
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_done_1","call_id":"call_done_1","type":"function_call","name":"done","arguments":"{\"message\":\"finished\"}","status":"completed"}}`,
		"",
		`data: {"type":"response.completed","response":` + secondOpaqueResponsesJSON() + `}`,
		"",
	}, "\n"))
}

func writeMaxTokenContinuationSSE(w http.ResponseWriter, call int) {
	w.Header().Set("Content-Type", "text/event-stream")
	if call == 1 {
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_partial","type":"reasoning","encrypted_content":"partial-ciphertext","summary":[]}}`,
			"",
			`data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fc_partial","call_id":"call_partial","type":"function_call","name":"done","arguments":"{\"message\":\"fin","status":"in_progress"}}`,
			"",
			`data: {"type":"response.incomplete","response":` + maxTokenPartialResponsesJSON() + `}`,
			"",
		}, "\n"))
		return
	}
	_, _ = io.WriteString(w, strings.Join([]string{
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_final","call_id":"call_partial","type":"function_call","name":"done","arguments":"ished\"}","status":"completed"}}`,
		"",
		`data: {"type":"response.completed","response":` + maxTokenFinalResponsesJSON() + `}`,
		"",
	}, "\n"))
}
