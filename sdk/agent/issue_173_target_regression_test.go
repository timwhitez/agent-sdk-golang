package agent

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"sync/atomic"
	"testing"
	"time"
)

func issue173TargetCall(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
}

type issue173TargetPathArgs struct {
	Path string `json:"path"`
}

func TestD1SharedTargetAcrossReadAndGrep(t *testing.T) {
	var active, peak atomic.Int32
	paired := make(chan struct{})
	handler := func(context.Context, issue173TargetPathArgs, *tools.Container) (any, error) {
		n := active.Add(1)
		defer active.Add(-1)
		if n == 2 {
			peak.Store(2)
			close(paired)
		} else {
			peak.CompareAndSwap(0, 1)
		}
		select {
		case <-paired:
		case <-time.After(100 * time.Millisecond):
		}
		return "ok", nil
	}
	model := &turnModel{turns: []*llm.Completion{{ToolCalls: []llm.ToolCall{issue173TargetCall("one", "read", `{"path":"report.txt"}`), issue173TargetCall("two", "grep", `{"path":"./report.txt"}`)}}}}
	a, err := New(Config{LLM: model, Tools: []tools.Tool{tools.Func("read", "read", handler), tools.Func("grep", "grep", handler)}, ToolParallelism: &ToolParallelism{MaxWorkers: 2, Plan: func(ToolCallPlanInput) BlockCallPlan { return BlockCallPlan{Concurrent: true} }}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Query(t.Context(), "inspect report"); err != nil {
		t.Fatal(err)
	}
	if got := peak.Load(); got != 1 {
		t.Fatalf("peak handlers for the same cleaned target=%d; want 1", got)
	}
}
