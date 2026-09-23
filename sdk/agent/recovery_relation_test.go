package agent

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// #83: a stream-idle recovery changes the next logical request (it appends
// a recovery reminder); that Frame's events name the stalled Frame as the
// producer, once, and no other Frame carries the relation.
func TestStreamIdleRecoveryRecordsRequestRelation(t *testing.T) {
	origTimeout, origRecoveries := agentStreamIdleTimeout, agentStreamIdleMaxRecoveries
	agentStreamIdleTimeout, agentStreamIdleMaxRecoveries = 20*time.Millisecond, 2
	t.Cleanup(func() { agentStreamIdleTimeout, agentStreamIdleMaxRecoveries = origTimeout, origRecoveries })

	ag, err := New(Config{LLM: &streamIdleRecoveryModel{}, StreamIdleMaxRecoveries: -1})
	if err != nil {
		t.Fatal(err)
	}
	var stalledFrame, finalFrame string
	relations := map[string]string{} // frame → recovery source
	for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("hello")) {
		if _, ok := env.Event.(FinalResponseEvent); ok {
			finalFrame = env.FrameID
		}
		if env.RequestRecoveryRelation != "" || env.RequestRecoverySourceFrameID != "" {
			if env.RequestRecoveryRelation != RequestRecoveryStreamIdle {
				t.Fatalf("relation=%q", env.RequestRecoveryRelation)
			}
			relations[env.FrameID] = env.RequestRecoverySourceFrameID
		}
	}
	// The first Frame stalled; the recovery request is the second.
	if strings.HasSuffix(finalFrame, "/frame/2") {
		stalledFrame = strings.TrimSuffix(finalFrame, "/frame/2") + "/frame/1"
	}
	if stalledFrame == "" || finalFrame == "" || stalledFrame == finalFrame {
		t.Fatalf("frames stalled=%q final=%q", stalledFrame, finalFrame)
	}
	if len(relations) != 1 || relations[finalFrame] != stalledFrame {
		t.Fatalf("relations=%v, want only %s → %s", relations, finalFrame, stalledFrame)
	}
}

// A query without a recovery reports no recovery relation.
func TestNoRecoveryNoRequestRelation(t *testing.T) {
	ag, err := New(Config{LLM: plainAnswerModel{}})
	if err != nil {
		t.Fatal(err)
	}
	for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("hello")) {
		if env.RequestRecoveryRelation != "" || env.RequestRecoverySourceFrameID != "" {
			t.Fatalf("unexpected relation: %+v", env)
		}
	}
}

type plainAnswerModel struct{}

func (plainAnswerModel) Provider() string { return "fixture" }
func (plainAnswerModel) Model() string    { return "plain" }
func (plainAnswerModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("answer")}, nil
}

// stallThenToolModel stalls once, then calls a tool, then answers.
type stallThenToolModel struct{ calls atomic.Int32 }

func (*stallThenToolModel) Provider() string { return "fixture" }
func (*stallThenToolModel) Model() string    { return "stall-then-tool" }
func (*stallThenToolModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}
func (m *stallThenToolModel) InvokeStream(ctx context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent, 3)
	n := m.calls.Add(1)
	go func() {
		defer close(ch)
		switch n {
		case 1:
			<-ctx.Done()
		case 2:
			ch <- llm.StreamToolCallDeltaEvent{Index: 0, ID: "noop-1", NameDelta: "noop", ArgumentsDelta: `{}`}
			ch <- llm.StreamDoneEvent{StopReason: "tool_calls"}
		default:
			ch <- llm.StreamTextDeltaEvent{Delta: "done"}
			ch <- llm.StreamDoneEvent{StopReason: "stop"}
		}
	}()
	return ch, nil
}

// The relation belongs to the first Frame after the recovery only.
func TestRecoveryRelationIsConsumedByOneFrame(t *testing.T) {
	origTimeout, origRecoveries := agentStreamIdleTimeout, agentStreamIdleMaxRecoveries
	agentStreamIdleTimeout, agentStreamIdleMaxRecoveries = 20*time.Millisecond, 2
	t.Cleanup(func() { agentStreamIdleTimeout, agentStreamIdleMaxRecoveries = origTimeout, origRecoveries })
	noop := tools.Func[struct{}]("noop", "noop", func(context.Context, struct{}, *tools.Container) (any, error) { return "ok", nil })
	ag, err := New(Config{LLM: &stallThenToolModel{}, Tools: []tools.Tool{noop}, StreamIdleMaxRecoveries: -1})
	if err != nil {
		t.Fatal(err)
	}
	related := map[string]bool{}
	frames := map[string]bool{}
	for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("hello")) {
		if env.FrameID != "" {
			frames[env.FrameID] = true
		}
		if env.RequestRecoveryRelation != "" {
			related[env.FrameID] = true
		}
	}
	// The stalled first Frame emits nothing; frames 2 and 3 do.
	hasThird := false
	for frame := range frames {
		hasThird = hasThird || strings.HasSuffix(frame, "/frame/3")
	}
	if len(related) != 1 || !hasThird {
		t.Fatalf("related=%v frames=%v", related, frames)
	}
	for frame := range related {
		if !strings.HasSuffix(frame, "/frame/2") {
			t.Fatalf("relation on %s, want only frame 2", frame)
		}
	}
}
