package agent

import (
	"context"
	"encoding/json"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"testing"
)

func issue211Call(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
}
func TestD1PreHandlerFailureReceipt(t *testing.T) {
	handled := 0
	lookup := tools.Tool{Name: "lookup", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		handled++
		return llm.TextContent("ok"), nil
	}}
	model := &turnModel{turns: []*llm.Completion{{ToolCalls: []llm.ToolCall{issue211Call("one", "lookup", `[]`)}}}}
	a, err := New(Config{LLM: model, Tools: []tools.Tool{lookup}})
	if err != nil {
		t.Fatal(err)
	}
	out, receipt := a.QueryStreamEnvelopedWithReceipt(t.Context(), llm.TextContent("look it up"), nil)
	var result ToolResultEvent
	for env := range out {
		if r, ok := env.Event.(ToolResultEvent); ok {
			result = r
		}
	}
	summary, _ := receipt.Summary()
	if handled != 0 || !result.IsError {
		t.Fatalf("fixture expected rejection before handler, got %d and %+v", handled, result)
	}
	if result.HandlerFailed || summary.HandlerFailures != 0 {
		t.Fatalf("handler invocations=%d, HandlerFailed=%v, receipt.HandlerFailures=%d; want false/0", handled, result.HandlerFailed, summary.HandlerFailures)
	}
}
