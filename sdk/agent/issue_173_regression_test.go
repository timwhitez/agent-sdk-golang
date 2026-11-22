package agent

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"testing"
)

func issue173Call(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
}

type issue173PathArgs struct {
	Path string `json:"path"`
}

func TestD1PlanPanicOnlyConsultedOnce(t *testing.T) {
	counts := map[int]int{}
	read := tools.Func("read", "read", func(context.Context, issue173PathArgs, *tools.Container) (any, error) { return "ok", nil })
	model := &turnModel{turns: []*llm.Completion{{ToolCalls: []llm.ToolCall{issue173Call("one", "read", `{"path":"one.txt"}`), issue173Call("two", "read", `{"path":"two.txt"}`)}}}}
	a, err := New(Config{LLM: model, Tools: []tools.Tool{read}, ToolParallelism: &ToolParallelism{MaxWorkers: 2, Plan: func(in ToolCallPlanInput) BlockCallPlan {
		counts[in.Ordinal]++
		if in.Ordinal == 1 {
			panic("unavailable")
		}
		return BlockCallPlan{Concurrent: true}
	}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Query(t.Context(), "inspect files"); err != nil {
		t.Fatal(err)
	}
	if counts[1] != 1 {
		t.Fatalf("plan calls for ordinal 1=%d; want 1", counts[1])
	}
}
