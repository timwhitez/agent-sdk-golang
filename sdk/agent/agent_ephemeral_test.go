package agent

import (
	"context"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type ephemeralRetentionModel struct {
	calls              int
	secondCallMessages []llm.Message
}

func (m *ephemeralRetentionModel) Provider() string { return "stub" }
func (m *ephemeralRetentionModel) Model() string    { return "stub" }

func (m *ephemeralRetentionModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	switch m.calls {
	case 1:
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: llm.FunctionCall{
						Name:      "echo",
						Arguments: `{"text":"first"}`,
					},
				},
				{
					ID:   "call_2",
					Type: "function",
					Function: llm.FunctionCall{
						Name:      "echo",
						Arguments: `{"text":"second"}`,
					},
				},
			},
		}, nil
	case 2:
		for _, msg := range req.Messages {
			if msg.Role == llm.RoleTool && msg.ToolName == "echo" {
				m.secondCallMessages = append(m.secondCallMessages, msg)
			}
		}
		return &llm.Completion{Content: llm.TextContent("done")}, nil
	default:
		return &llm.Completion{Content: llm.TextContent("done")}, nil
	}
}

func TestEphemeralToolOutputsRespectKeepCountOnNextCall(t *testing.T) {
	type echoArgs struct {
		Text string `json:"text"`
	}
	echoTool := tools.Func[echoArgs]("echo", "echo", func(_ context.Context, args echoArgs, _ *tools.Container) (any, error) {
		return args.Text, nil
	}).WithEphemeralKeep(1)

	model := &ephemeralRetentionModel{}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{echoTool}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	_ = collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	if model.calls < 2 {
		t.Fatalf("expected at least 2 model calls, got %d", model.calls)
	}
	if len(model.secondCallMessages) != 2 {
		t.Fatalf("expected 2 tool messages on second call, got %d", len(model.secondCallMessages))
	}
	if got := model.secondCallMessages[0].PlainText(); got != "<removed to save context>" {
		t.Fatalf("expected first tool output to be compacted, got %q", got)
	}
	if got := model.secondCallMessages[1].PlainText(); got != "second" {
		t.Fatalf("expected second tool output, got %q", got)
	}
	if !model.secondCallMessages[0].Destroyed {
		t.Fatalf("expected older tool output to be marked destroyed")
	}
	if model.secondCallMessages[1].Destroyed {
		t.Fatalf("expected newest tool output to be retained")
	}
}
