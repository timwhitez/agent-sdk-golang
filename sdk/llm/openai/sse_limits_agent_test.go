package openai_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// A stream that exceeds the local SSE parser budget after a partial tool call
// must end the Query with an error: the incomplete call never runs and the
// Agent does not transparently re-send the request.
func TestIssue152AgentDoesNotExecuteOrRetryResourceLimitedStream(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"effect","arguments":"{\"path\":"}}]}}]}`+"\n\n")
		_, _ = fmt.Fprint(w, strings.Repeat("data: {\"choices\":\n\n", 40))
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	var effects atomic.Int32
	type effectArgs struct {
		Path string `json:"path"`
	}
	effect := tools.Func[effectArgs]("effect", "fixture effect", func(context.Context, effectArgs, *tools.Container) (any, error) {
		effects.Add(1)
		return "ok", nil
	})
	client := &openai.ChatClient{HTTPClient: server.Client(), BaseURL: server.URL, ModelName: "test-model", MaxRetries: 1}
	a, err := agent.New(agent.Config{LLM: client, Tools: []tools.Tool{effect}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = a.Query(context.Background(), "hello")
	if err == nil {
		t.Fatal("resource-limited stream completed without error")
	}
	if !strings.Contains(err.Error(), "SSE parser resource budget") {
		t.Fatalf("err = %v, want the SSE resource-limit terminal", err)
	}
	if got := effects.Load(); got != 0 {
		t.Fatalf("incomplete tool call executed %d times", got)
	}
	if got := requests.Load(); got != 1 {
		t.Fatalf("HTTP requests = %d, want 1 (no transparent retry)", got)
	}
}
