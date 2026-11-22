package main

import (
	"bytes"
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"testing"
)

func TestAuditFinalSuffixOfEarlierProgress(t *testing.T) {
	check := tools.Func[struct{}]("check", "", func(context.Context, struct{}, *tools.Container) (any, error) { return "ok", nil })
	type args struct {
		Message string `json:"message"`
	}
	done := tools.Func[args]("done", "", func(_ context.Context, a args, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(a.Message)
	})
	model := &scriptedStreamer{turns: [][]llm.StreamEvent{toolCallTurn("Earlier guess: not ready", "c1", "check", `{}`), toolCallTurn("", "c2", "done", `{"message":"ready"}`)}}
	a, err := agent.New(agent.Config{LLM: model, Tools: []tools.Tool{check, done}})
	if err != nil {
		t.Fatal(err)
	}
	var out, diag bytes.Buffer
	if err := consumeAgentEnvelopes(a.QueryStreamEnveloped(context.Background(), llm.TextContent("test")), &out, &diag); err != nil {
		t.Fatal(err)
	}
	if want := "Earlier guess: not ready\n" + finalMarker + "ready\n"; out.String() != want {
		t.Fatalf("display=%q; want %q", out.String(), want)
	}
}
