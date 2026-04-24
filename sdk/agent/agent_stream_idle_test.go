package agent

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type streamIdleRecoveryModel struct {
	calls        atomic.Int32
	partialFirst bool
}

func (m *streamIdleRecoveryModel) Provider() string { return "stub" }
func (m *streamIdleRecoveryModel) Model() string    { return "stub" }

func (m *streamIdleRecoveryModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}

func (m *streamIdleRecoveryModel) InvokeStream(ctx context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent, 2)
	call := m.calls.Add(1)
	go func() {
		defer close(ch)
		if call == 1 {
			if m.partialFirst {
				ch <- llm.StreamTextDeltaEvent{Delta: "partial "}
			}
			<-ctx.Done()
			return
		}
		ch <- llm.StreamTextDeltaEvent{Delta: "recovered"}
		ch <- llm.StreamDoneEvent{StopReason: "stop"}
	}()
	return ch, nil
}

type streamIdleForeverModel struct {
	calls atomic.Int32
}

func (m *streamIdleForeverModel) Provider() string { return "stub" }
func (m *streamIdleForeverModel) Model() string    { return "stub" }

func (m *streamIdleForeverModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}

func (m *streamIdleForeverModel) InvokeStream(ctx context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent)
	m.calls.Add(1)
	go func() {
		defer close(ch)
		<-ctx.Done()
	}()
	return ch, nil
}

func TestQueryStreamAutoRecoversFromIdleStream(t *testing.T) {
	origTimeout := agentStreamIdleTimeout
	origRecoveries := agentStreamIdleMaxRecoveries
	agentStreamIdleTimeout = 20 * time.Millisecond
	agentStreamIdleMaxRecoveries = 2
	t.Cleanup(func() {
		agentStreamIdleTimeout = origTimeout
		agentStreamIdleMaxRecoveries = origRecoveries
	})

	model := &streamIdleRecoveryModel{}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hello")))
	for _, ev := range events {
		if _, ok := ev.(ErrorEvent); ok {
			t.Fatalf("expected idle stream to recover without error, got events=%#v", events)
		}
	}
	var final FinalResponseEvent
	var found bool
	for _, ev := range events {
		if f, ok := ev.(FinalResponseEvent); ok {
			final = f
			found = true
		}
	}
	if !found {
		t.Fatalf("expected final response after idle recovery, got events=%#v", events)
	}
	if final.Content != "recovered" {
		t.Fatalf("expected recovered final content, got %q", final.Content)
	}
	if final.StallRecoveries != 1 {
		t.Fatalf("expected stall_recoveries=1 on final event, got %d", final.StallRecoveries)
	}
	if got := model.calls.Load(); got != 2 {
		t.Fatalf("expected 2 streaming attempts (idle recover + success), got %d", got)
	}
}

func TestQueryStreamAutoRecoversFromIdleStreamWithPartialText(t *testing.T) {
	origTimeout := agentStreamIdleTimeout
	origRecoveries := agentStreamIdleMaxRecoveries
	agentStreamIdleTimeout = 20 * time.Millisecond
	agentStreamIdleMaxRecoveries = 2
	t.Cleanup(func() {
		agentStreamIdleTimeout = origTimeout
		agentStreamIdleMaxRecoveries = origRecoveries
	})

	model := &streamIdleRecoveryModel{partialFirst: true}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hello")))
	for _, ev := range events {
		if _, ok := ev.(ErrorEvent); ok {
			t.Fatalf("expected partial idle stream to recover without error, got events=%#v", events)
		}
	}

	msgs := ag.Messages()
	foundPartial := false
	for _, msg := range msgs {
		if msg.Role == llm.RoleAssistant && msg.PlainText() == "partial " {
			foundPartial = true
			break
		}
	}
	if !foundPartial {
		t.Fatalf("expected partial assistant text to be preserved in history, got %#v", msgs)
	}
	var final FinalResponseEvent
	var foundFinal bool
	for _, ev := range events {
		if f, ok := ev.(FinalResponseEvent); ok {
			final = f
			foundFinal = true
		}
	}
	if !foundFinal {
		t.Fatalf("expected final response after partial idle recovery, got events=%#v", events)
	}
	if final.StallRecoveries != 1 {
		t.Fatalf("expected stall_recoveries=1 on final event, got %d", final.StallRecoveries)
	}
}

func TestQueryStreamIdleRecoveryEventuallySurfacesError(t *testing.T) {
	origTimeout := agentStreamIdleTimeout
	origRecoveries := agentStreamIdleMaxRecoveries
	agentStreamIdleTimeout = 20 * time.Millisecond
	agentStreamIdleMaxRecoveries = 1
	t.Cleanup(func() {
		agentStreamIdleTimeout = origTimeout
		agentStreamIdleMaxRecoveries = origRecoveries
	})

	model := &streamIdleForeverModel{}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hello")))
	var errEvent ErrorEvent
	var found bool
	for _, ev := range events {
		if e, ok := ev.(ErrorEvent); ok {
			errEvent = e
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected idle stream error after exhausting recoveries, got events=%#v", events)
	}
	if errEvent.Kind != "timeout" {
		t.Fatalf("expected timeout error kind, got %#v", errEvent)
	}
	if errEvent.StallRecoveries != 1 {
		t.Fatalf("expected stall_recoveries=1 on final error, got %d", errEvent.StallRecoveries)
	}
	if got := model.calls.Load(); got != 2 {
		t.Fatalf("expected one silent recovery before final error, got %d calls", got)
	}
}
