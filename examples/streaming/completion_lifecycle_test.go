package main

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestAgentWhitespaceStopContinuationThenReminder(t *testing.T) {
	work := tools.Func[struct{}]("work", "work", func(context.Context, struct{}, *tools.Container) (any, error) { return "ok", nil })
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "finish", func(_ context.Context, x struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(x.Message)
	})
	for _, required := range []bool{false, true} {
		for _, leading := range []string{"", " ", "\n"} {
			t.Run(leading+map[bool]string{false: "generic", true: "required"}[required], func(t *testing.T) {
				model := &scriptedStreamer{turns: [][]llm.StreamEvent{
					toolCallTurn("", "w", "work", `{}`),
					{llm.StreamTextDeltaEvent{Delta: "First answer."}, llm.StreamDoneEvent{StopReason: "max_tokens"}},
					{llm.StreamTextDeltaEvent{Delta: leading}, llm.StreamDoneEvent{StopReason: "stop"}},
					toolCallTurn("Revised answer.", "d", "done", `{"message":"Saved."}`),
				}}
				a, e := agent.New(agent.Config{LLM: model, Tools: []tools.Tool{work, done}, RequireDoneTool: required, Warningf: func(string, ...any) {}})
				if e != nil {
					t.Fatal(e)
				}
				var out, diag bytes.Buffer
				e = consumeAgentEnvelopes(a.QueryStreamEnveloped(context.Background(), llm.TextContent("test")), &out, &diag)
				t.Logf("out=%q err=%v", out.String(), e)
				if e != nil {
					t.Fatal(e)
				}
				if strings.Count(out.String(), "Revised answer.") != 1 {
					t.Fatalf("new answer repeated: %q", out.String())
				}
			})
		}
	}
}

type completionLifecycleStream struct {
	mu     sync.Mutex
	n      int
	second chan struct{}
}

func (*completionLifecycleStream) Provider() string { return "offline" }
func (*completionLifecycleStream) Model() string    { return "lifecycle" }
func (*completionLifecycleStream) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("unexpected buffered")
}
func (m *completionLifecycleStream) InvokeStream(ctx context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.mu.Lock()
	m.n++
	n := m.n
	m.mu.Unlock()
	if n == 2 {
		close(m.second)
		ch := make(chan llm.StreamEvent)
		go func() { <-ctx.Done(); close(ch) }()
		return ch, nil
	}
	var ev []llm.StreamEvent
	if n == 1 {
		ev = []llm.StreamEvent{llm.StreamTextDeltaEvent{Delta: "Abandoned answer."}, llm.StreamDoneEvent{StopReason: "max_tokens"}}
	} else if n == 3 {
		ev = toolCallTurn("Revised answer.", "done", "done", `{"message":"Saved."}`)
	} else {
		return nil, errors.New("unexpected request")
	}
	ch := make(chan llm.StreamEvent, len(ev))
	for _, e := range ev {
		ch <- e
	}
	close(ch)
	return ch, nil
}
func TestAgentContinuationSteeringAfterMaxTokens(t *testing.T) {
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "finish", func(_ context.Context, x struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(x.Message)
	})
	model := &completionLifecycleStream{second: make(chan struct{})}
	a, e := agent.New(agent.Config{LLM: model, Tools: []tools.Tool{done}, Warningf: func(string, ...any) {}})
	if e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	steering := make(chan agent.SteeringMsg, 1)
	stream := a.QueryStreamEnvelopedWithSteering(ctx, llm.TextContent("test"), steering)
	var out, diag bytes.Buffer
	var observed []string
	e = consumeAgentOutput(func() (agent.EventEnvelope, bool) {
		env, ok := <-stream
		if !ok {
			return agent.EventEnvelope{}, false
		}
		switch ev := env.Event.(type) {
		case agent.AutoContinueEvent:
			select {
			case <-model.second:
			case <-ctx.Done():
				t.Fatal("second stream not entered")
			}
			steering <- agent.SteeringMsg{Content: "Replace with revised answer."}
			observed = append(observed, "auto_continue")
		case agent.SteeringReceivedEvent:
			observed = append(observed, "steering")
		case agent.FinalResponseEvent:
			observed = append(observed, "final="+ev.Content)
		}
		return env, true
	}, &out, &diag)
	t.Logf("observed=%q output=%q err=%v", observed, out.String(), e)
	if e != nil {
		t.Fatal(e)
	}
	if strings.Count(out.String(), "Revised answer.") != 1 {
		t.Fatalf("revised answer duplicated: %q", out.String())
	}
}
func TestAgentContinuationLimit(t *testing.T) {
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "finish", func(_ context.Context, x struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(x.Message)
	})
	var turns [][]llm.StreamEvent
	for i := 0; i < 4; i++ {
		ev := toolCallTurn("", "partial", "done", `{"message":"`)
		if i == 0 {
			ev = toolCallTurn("Abandoned answer.", "partial", "done", `{"message":"`)
		}
		ev[len(ev)-1] = llm.StreamDoneEvent{StopReason: "max_tokens"}
		turns = append(turns, ev)
	}
	turns = append(turns, toolCallTurn("Revised answer.", "done", "done", `{"message":"Saved."}`))
	a, e := agent.New(agent.Config{LLM: &scriptedStreamer{turns: turns}, Tools: []tools.Tool{done}, Warningf: func(string, ...any) {}})
	if e != nil {
		t.Fatal(e)
	}
	var out, diag bytes.Buffer
	stream := a.QueryStreamEnveloped(context.Background(), llm.TextContent("test"))
	var limits int
	e = consumeAgentOutput(func() (agent.EventEnvelope, bool) {
		env, ok := <-stream
		if warn, yes := env.Event.(agent.WarnEvent); yes && warn.Kind == "continuation_limit" {
			limits++
		}
		return env, ok
	}, &out, &diag)
	t.Logf("limits=%d output=%q err=%v", limits, out.String(), e)
	if e != nil {
		t.Fatal(e)
	}
	if limits == 0 {
		t.Fatal("limit not reached")
	}
	if strings.Count(out.String(), "Revised answer.") != 1 {
		t.Fatalf("revised answer duplicated: %q", out.String())
	}
}

func TestAgentAbandonedArgumentsReplacement(t *testing.T) {
	for _, required := range []bool{false, true} {
		done := tools.Func[struct {
			Message string `json:"message"`
		}]("done", "complete", func(_ context.Context, args struct {
			Message string `json:"message"`
		}, _ *tools.Container) (any, error) {
			return nil, tools.TaskComplete(args.Message)
		})
		first := toolCallTurn("ABANDONED", "partial", "done", `{"message":"`)
		first[len(first)-1] = llm.StreamDoneEvent{StopReason: "max_tokens"}
		model := &scriptedStreamer{turns: [][]llm.StreamEvent{
			first,
			{llm.StreamTextDeltaEvent{Delta: "REPLACEMENT"}, llm.StreamDoneEvent{StopReason: "stop"}},
			toolCallTurn("", "d1", "done", `{"message":"Saved."}`),
		}}
		a, err := agent.New(agent.Config{LLM: model, Tools: []tools.Tool{done}, RequireDoneTool: required, Warningf: func(string, ...any) {}})
		if err != nil {
			t.Fatal(err)
		}
		var out, diag bytes.Buffer
		err = consumeAgentEnvelopes(a.QueryStreamEnveloped(context.Background(), llm.TextContent("fixture")), &out, &diag)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Count(out.String(), "REPLACEMENT") != 1 || strings.Count(out.String(), "Saved.") != 1 {
			t.Fatalf("required=%t output=%q", required, out.String())
		}
	}
}

func TestAgentGenericCompletionPayload(t *testing.T) {
	for _, payload := range []string{"Answer.", "Saved."} {
		finish := tools.Func[struct{}]("finish", "complete", func(context.Context, struct{}, *tools.Container) (any, error) {
			return nil, tools.TaskComplete(payload)
		})
		a, err := agent.New(agent.Config{LLM: &scriptedStreamer{turns: [][]llm.StreamEvent{toolCallTurn("Answer.", "f1", "finish", `{}`)}}, Tools: []tools.Tool{finish}, Warningf: func(string, ...any) {}})
		if err != nil {
			t.Fatal(err)
		}
		var out, diag bytes.Buffer
		if err := consumeAgentEnvelopes(a.QueryStreamEnveloped(context.Background(), llm.TextContent("fixture")), &out, &diag); err != nil {
			t.Fatal(err)
		}
		if strings.Count(out.String(), "Answer.") != 1 || strings.Count(out.String(), payload) != 1 {
			t.Fatalf("payload=%q output=%q", payload, out.String())
		}
	}
}
