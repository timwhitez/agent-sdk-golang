package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
)

type todoPromptModel struct {
	calls int
	reqs  []llm.InvokeRequest
}

func (m *todoPromptModel) Provider() string { return "stub" }
func (m *todoPromptModel) Model() string    { return "stub" }

func (m *todoPromptModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	m.reqs = append(m.reqs, req)
	if m.calls == 1 {
		return &llm.Completion{Content: llm.TextContent("draft"), StopReason: "stop"}, nil
	}
	return &llm.Completion{Content: llm.TextContent("final"), StopReason: "stop"}, nil
}

func TestAgentDoesNotDependOnTodoPromptForEarlyStop(t *testing.T) {
	root := t.TempDir()
	sb, err := sandbox.New(root)
	if err != nil {
		t.Fatalf("new sandbox: %v", err)
	}
	sb.ReplaceTodos([]sandbox.TodoItem{
		{Content: "fix bug", Status: "pending"},
		{Content: "done", Status: "completed"},
	})

	deps := tools.NewContainer()
	tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) { return sb, nil })

	model := &todoPromptModel{}
	ag, err := New(Config{LLM: model, Deps: deps, MaxIterations: 3})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	if model.calls != 1 {
		t.Fatalf("expected single model call without todo-coupled reminder, got %d", model.calls)
	}

	final := ""
	for _, ev := range events {
		if f, ok := ev.(FinalResponseEvent); ok {
			final = f.Content
		}
	}
	if final != "draft" {
		t.Fatalf("expected final response %q, got %q", "draft", final)
	}
	if len(model.reqs) != 1 {
		t.Fatalf("expected exactly 1 request, got %d", len(model.reqs))
	}

	for _, msg := range model.reqs[0].Messages {
		if msg.Role == llm.RoleUser && strings.Contains(msg.PlainText(), "Incomplete todos") {
			t.Fatalf("did not expect todo-coupled reminder, got %q", msg.PlainText())
		}
	}
}
