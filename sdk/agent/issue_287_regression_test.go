package agent

import (
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func issue287Call(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
}
func TestD1IterationCapAbandonsToolContinuation(t *testing.T) {
	model := &turnModel{turns: []*llm.Completion{{Content: llm.TextContent("draft"), StopReason: "max_tokens", ToolCalls: []llm.ToolCall{issue287Call("t1", "lookup", `{"id":`)}}}}
	a, err := New(Config{LLM: model, MaxIterations: 1})
	if err != nil {
		t.Fatal(err)
	}
	_, err = a.Query(t.Context(), "Look up the item")
	if err == nil {
		t.Fatal("expected the iteration limit")
	}
	messages := a.Messages()
	if start := llm.OpenToolCallBlockStart(messages); start >= 0 {
		t.Fatalf("closed query has an open tool block at history[%d], id=%q, following reminder=%q; want no pending call", start, messages[start].ToolCalls[0].ID, messages[start+1].Name)
	}
	if last := messages[len(messages)-1]; last.Name == messageorigin.Name(messageorigin.KindToolCallContinuation) || last.PlainText() != "draft" {
		t.Fatalf("abandoned continuation retained its reminder or lost text: %+v", last)
	}
}
