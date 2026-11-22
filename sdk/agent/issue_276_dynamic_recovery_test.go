package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type independentDynamicRouter struct {
	bound    int
	requests []llm.InvokeRequest
	root     *independentDynamicRouter
	ordinal  int
}

func (*independentDynamicRouter) Provider() string { return "fixture" }
func (*independentDynamicRouter) Model() string    { return "dynamic" }
func (m *independentDynamicRouter) BindFrameModel(context.Context) (llm.ChatModel, bool, error) {
	m.bound++
	return &independentDynamicRouter{root: m, ordinal: m.bound}, true, nil
}
func (m *independentDynamicRouter) ThinkingCapabilities() (llm.ThinkingCapabilities, bool) {
	return llm.ThinkingCapabilities{SupportsDisableThinking: m.ordinal < 4, SupportsForcedToolChoice: true}, true
}
func (m *independentDynamicRouter) Invoke(_ context.Context, r llm.InvokeRequest) (*llm.Completion, error) {
	if m.root == nil {
		panic("unbound model")
	}
	m.root.requests = append(m.root.requests, r)
	if m.ordinal == 2 {
		return &llm.Completion{Content: llm.TextContent("answer"), StopReason: "stop"}, nil
	}
	name, args, id := "work", "{}", "one"
	if m.ordinal == 3 {
		id = "three"
	}
	if m.ordinal == 4 {
		name, args, id = "done", `{"message":"complete"}`, "four"
	}
	return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}}}, nil
}
func TestIndependentDynamicRecoveryWarningMatchesWire(t *testing.T) {
	m := &independentDynamicRouter{}
	work := tools.Func[struct{}]("work", "", func(context.Context, struct{}, *tools.Container) (any, error) { return "ok", nil })
	type doneArgs struct {
		Message string `json:"message"`
	}
	done := tools.Func[doneArgs]("done", "", func(_ context.Context, a doneArgs, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(a.Message)
	})
	a, err := New(Config{LLM: m, Tools: []tools.Tool{work, done}, ToolChoice: "auto", RequireDoneTool: true, MaxIterations: 6, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	warning := ""
	for env := range a.QueryStreamEnveloped(t.Context(), llm.TextContent("run")) {
		if e, ok := env.Event.(WarnEvent); ok && e.Kind == "require_done_capability_degraded" {
			warning = e.Message
		}
		if e, ok := env.Event.(ErrorEvent); ok {
			t.Fatalf("query error: %+v", e)
		}
	}
	if len(m.requests) != 4 {
		t.Fatalf("requests=%d", len(m.requests))
	}
	if m.requests[2].ToolChoice != "required" || !m.requests[2].DisableThinking || m.requests[3].ToolChoice != "auto" || m.requests[3].DisableThinking {
		t.Fatalf("unexpected controls %#v", m.requests)
	}
	if strings.Contains(warning, "with forced tool_choice") {
		t.Fatalf("last request tool_choice=%q but warning=%q", m.requests[3].ToolChoice, warning)
	}
}
