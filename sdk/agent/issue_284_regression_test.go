package agent

import (
	"context"
	"errors"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
	"testing"
)

func issue284Call(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
}

type issue284PathArgs struct {
	Path string `json:"path"`
}

func TestD1EvidenceAdmissionHonorsCancellation(t *testing.T) {
	observed := make(chan context.Context, 1)
	release := make(chan struct{})
	deps := tools.NewContainer()
	tools.Provide(deps, sandbox.Key, func(ctx context.Context) (*sandbox.Sandbox, error) {
		observed <- ctx
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-release:
			return nil, errors.New("fixture released")
		}
	})
	read := tools.Func("read", "read", func(context.Context, issue284PathArgs, *tools.Container) (any, error) { return "ok", nil })
	model := &turnModel{turns: []*llm.Completion{{ToolCalls: []llm.ToolCall{issue284Call("one", "read", `{"path":"report.txt"}`)}}}}
	a, err := New(Config{LLM: model, Tools: []tools.Tool{read}, Deps: deps})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	out := a.QueryStream(ctx, llm.TextContent("read report"))
	depCtx := <-observed
	cancel()
	ended := false
	select {
	case <-depCtx.Done():
		ended = true
	default:
	}
	close(release)
	for range out {
	}
	if !ended {
		t.Fatalf("query context canceled but sandbox dependency context Done is nil=%v; resolver needed manual release", depCtx.Done() == nil)
	}
	if _, err := a.Query(t.Context(), "next query"); err != nil {
		t.Fatalf("agent remains unavailable after canceled evidence admission: %v", err)
	}
}
