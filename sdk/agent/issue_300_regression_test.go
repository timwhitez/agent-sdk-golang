package agent

import (
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"sync/atomic"
	"testing"
	"time"
)

func TestD1SteeringAcknowledgementAfterHistoryCommit(t *testing.T) {
	gate := make(chan struct{})
	var clockCalls atomic.Int32
	a, err := New(Config{LLM: &turnModel{}, EventClock: func() time.Time {
		if clockCalls.Add(1) == 2 {
			<-gate
		}
		return time.Now()
	}})
	if err != nil {
		t.Fatal(err)
	}
	steer := make(chan SteeringMsg, 2)
	steer <- SteeringMsg{Content: "change to first"}
	steer <- SteeringMsg{Content: "change to second"}
	out := a.QueryStreamWithSteering(t.Context(), llm.TextContent("original"), steer)
	e := <-out
	msg, ok := e.(SteeringReceivedEvent)
	if !ok {
		close(gate)
		for range out {
		}
		t.Fatalf("expected steering, got %T", e)
	}
	seen := countText(a.Messages(), msg.Content)
	close(gate)
	for range out {
	}
	if seen != 1 {
		t.Fatalf("after SteeringReceivedEvent(%q), history contained %d copies; want 1", msg.Content, seen)
	}
}
