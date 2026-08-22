package anthropic_test

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
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestAgentStreamingReplaysAnthropicThinkingSignatureAfterToolUse(t *testing.T) {
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

		w.Header().Set("Content-Type", "text/event-stream")
		switch call {
		case 1:
			_, _ = io.WriteString(w, strings.Join([]string{
				`data: {"type":"message_start","message":{"id":"msg_stream_1","usage":{"input_tokens":10}}}`,
				"",
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
				"",
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"inspect exact state"}}`,
				"",
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_stream_abc"}}`,
				"",
				`data: {"type":"content_block_stop","index":0}`,
				"",
				`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"echo_1","name":"echo","input":{}}}`,
				"",
				`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"message\":\"work\"}"}}`,
				"",
				`data: {"type":"content_block_stop","index":1}`,
				"",
				`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}`,
				"",
				`data: {"type":"message_stop"}`,
				"",
			}, "\n"))
		case 2:
			_, _ = io.WriteString(w, strings.Join([]string{
				`data: {"type":"message_start","message":{"id":"msg_stream_2","usage":{"input_tokens":20}}}`,
				"",
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"done_1","name":"done","input":{}}}`,
				"",
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"message\":\"finished\"}"}}`,
				"",
				`data: {"type":"content_block_stop","index":0}`,
				"",
				`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}`,
				"",
				`data: {"type":"message_stop"}`,
				"",
			}, "\n"))
		default:
			http.Error(w, fmt.Sprintf("unexpected request %d", call), http.StatusBadRequest)
		}
	}))
	defer server.Close()

	client := &anthropic.Client{
		HTTPClient:     server.Client(),
		BaseURL:        server.URL,
		ModelName:      "claude-opus-4-8",
		MaxTokens:      1024,
		ThinkingMode:   "adaptive",
		ThinkingEffort: "max",
		MaxRetries:     1,
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
		LLM:             client,
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
	captured := append([][]byte(nil), requests...)
	mu.Unlock()
	if len(captured) != 2 {
		t.Fatalf("request count = %d, want 2", len(captured))
	}
	var payload struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(captured[1], &payload); err != nil {
		t.Fatalf("decode second request: %v", err)
	}
	var assistant []map[string]any
	for _, message := range payload.Messages {
		if message.Role == "assistant" {
			assistant = message.Content
			break
		}
	}
	if len(assistant) < 2 {
		t.Fatalf("assistant content missing thinking/tool_use blocks: %#v", payload.Messages)
	}
	if assistant[0]["type"] != "thinking" || assistant[0]["thinking"] != "inspect exact state" || assistant[0]["signature"] != "sig_stream_abc" {
		t.Fatalf("replayed thinking block = %#v", assistant[0])
	}
	if assistant[1]["type"] != "tool_use" || assistant[1]["id"] != "echo_1" {
		t.Fatalf("replayed tool_use block = %#v", assistant[1])
	}
}
