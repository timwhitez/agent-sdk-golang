package agent

import (
	"context"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type blockingModel struct {
	started chan struct{}
}

func (m *blockingModel) Provider() string { return "stub" }
func (m *blockingModel) Model() string    { return "stub" }

func (m *blockingModel) Invoke(ctx context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	if m.started != nil {
		select {
		case <-m.started:
			// already closed
		default:
			close(m.started)
		}
	}
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestSteeringChannelOwnership_keepsCallerChannelOpenOnContextCancel(t *testing.T) {
	model := &blockingModel{started: make(chan struct{})}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	steering := make(chan SteeringMsg, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events := ag.QueryStreamWithSteering(ctx, llm.TextContent("hi"), steering)
	<-model.started
	cancel()

	for range events {
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("expected caller-owned steering channel to remain open, panic=%v", r)
		}
	}()

	steering <- SteeringMsg{Content: "still-open"}
	select {
	case msg := <-steering:
		if msg.Content != "still-open" {
			t.Fatalf("unexpected steering message: %#v", msg)
		}
	default:
		t.Fatalf("expected steering message send/receive to succeed on open channel")
	}
}
