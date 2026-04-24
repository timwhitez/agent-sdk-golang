package agent

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// slowStreamingModel simulates a model that streams text slowly,
// allowing steering messages to interrupt mid-stream.
type slowStreamingModel struct {
	emitCount       *atomic.Int32
	eventsToEmit    int
	delayPerEvent   time.Duration
	emitStarted     chan struct{}
	emitFinished    chan struct{}
	blockOnSteering bool
}

func (m *slowStreamingModel) Provider() string { return "stub" }
func (m *slowStreamingModel) Model() string    { return "stub" }

func (m *slowStreamingModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}

func (m *slowStreamingModel) InvokeStream(_ context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent, 1)
	if m.emitStarted != nil {
		select {
		case <-m.emitStarted:
		default:
			close(m.emitStarted)
		}
	}
	go func() {
		defer close(ch)
		if m.emitFinished != nil {
			defer func() {
				select {
				case <-m.emitFinished:
				default:
					close(m.emitFinished)
				}
			}()
		}
		for i := 0; i < m.eventsToEmit; i++ {
			if m.delayPerEvent > 0 {
				time.Sleep(m.delayPerEvent)
			}
			ch <- llm.StreamTextDeltaEvent{Delta: "word "}
			if m.emitCount != nil {
				m.emitCount.Add(1)
			}
			// If blocking on steering, yield to allow steering to be processed
			if m.blockOnSteering {
				time.Sleep(10 * time.Millisecond)
			}
		}
		ch <- llm.StreamDoneEvent{StopReason: "stop"}
	}()
	return ch, nil
}

// TestSteeringInterruptMidStream verifies that a steering message sent
// during streaming interrupts the LLM stream and incorporates the message.

func TestSteeringInterruptDrainsProviderStream(t *testing.T) {
	emitCount := atomic.Int32{}
	model := &slowStreamingModel{
		emitCount:     &emitCount,
		eventsToEmit:  64,
		delayPerEvent: time.Millisecond,
		emitStarted:   make(chan struct{}),
		emitFinished:  make(chan struct{}),
	}

	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	steering := make(chan SteeringMsg, 1)
	events := ag.QueryStreamWithSteering(ctx, llm.TextContent("hello"), steering)

	select {
	case <-model.emitStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for stream to start")
	}

	time.Sleep(10 * time.Millisecond)
	steering <- SteeringMsg{Content: "interrupt"}

	steeringSeen := false
	timeout := time.After(5 * time.Second)
	for !steeringSeen {
		select {
		case ev, ok := <-events:
			if !ok {
				t.Fatal("event stream closed before steering was observed")
			}
			if _, ok := ev.(SteeringReceivedEvent); ok {
				steeringSeen = true
			}
		case <-timeout:
			t.Fatal("timeout waiting for steering event")
		}
	}

	cancel()
	for range events {
	}

	select {
	case <-model.emitFinished:
	case <-time.After(2 * time.Second):
		t.Fatal("provider stream did not finish after steering interrupt")
	}
}

func TestSteeringInterruptMidStream(t *testing.T) {
	emitCount := atomic.Int32{}
	model := &slowStreamingModel{
		emitCount:     &emitCount,
		eventsToEmit:  100, // Many events - would take a while
		delayPerEvent: 30 * time.Millisecond,
		emitStarted:   make(chan struct{}),
	}

	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	steering := make(chan SteeringMsg, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start the query in background
	events := ag.QueryStreamWithSteering(ctx, llm.TextContent("tell me a long story"), steering)

	// Wait for streaming to start
	select {
	case <-model.emitStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for stream to start")
	}

	// Wait a bit to ensure some text has been streamed
	time.Sleep(80 * time.Millisecond)

	// Check that some events were emitted
	eventsEmitted := emitCount.Load()
	if eventsEmitted == 0 {
		t.Fatal("expected some events to be emitted before steering")
	}

	// Send steering message to interrupt
	steering <- SteeringMsg{Content: "stop that, tell me about Go instead"}

	// Collect events and verify steering was received
	var steeringReceived bool
	var textDeltaCount int
	timeout := time.After(10 * time.Second)

loop:
	for {
		select {
		case ev, ok := <-events:
			if !ok {
				// Channel closed
				break loop
			}
			switch e := ev.(type) {
			case SteeringReceivedEvent:
				if e.Content == "stop that, tell me about Go instead" {
					steeringReceived = true
				}
			case FinalResponseEvent:
				// Stream ended - steering was processed
				break loop
			case TextDeltaEvent:
				textDeltaCount++
			}
		case <-timeout:
			t.Fatal("timeout waiting for events")
		}
	}

	if !steeringReceived {
		t.Errorf("expected steering received event, textDeltaCount=%d, eventsEmitted=%d", textDeltaCount, eventsEmitted)
	}

	// The test passes if we received the steering event.
	// We don't strictly verify interruption timing because the select
	// might process a stream event before the steering message arrives.
	// The key is that the steering message IS processed and an event is emitted.
}

// TestSteeringInterruptWithPartialCompletion verifies that when a stream
// is interrupted, the partial completion is saved to history.
func TestSteeringInterruptWithPartialCompletion(t *testing.T) {
	emitCount := atomic.Int32{}
	model := &slowStreamingModel{
		emitCount:     &emitCount,
		eventsToEmit:  50,
		delayPerEvent: 10 * time.Millisecond,
		emitStarted:   make(chan struct{}),
	}

	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	steering := make(chan SteeringMsg, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events := ag.QueryStreamWithSteering(ctx, llm.TextContent("hello"), steering)

	// Wait for streaming to start
	select {
	case <-model.emitStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for stream to start")
	}

	// Wait a bit for some text
	time.Sleep(50 * time.Millisecond)

	// Send steering
	steering <- SteeringMsg{Content: "interrupt"}

	// Drain events
	for range events {
	}

	// Check history - should contain partial assistant message
	messages := ag.Messages()
	if len(messages) < 2 {
		t.Fatalf("expected at least 2 messages in history, got %d", len(messages))
	}

	// Find the assistant message with partial content
	foundPartial := false
	for _, msg := range messages {
		if msg.Role == llm.RoleAssistant {
			text := msg.PlainText()
			if text != "" {
				foundPartial = true
				// Should have some text but not the full 50 "word " chunks
				if len(text) > 500 {
					t.Errorf("partial text too long: %d chars", len(text))
				}
			}
		}
	}

	if !foundPartial {
		t.Error("expected to find partial assistant message in history")
	}
}

// TestSteeringInterruptNilChannel verifies backward compatibility:
// when steering channel is nil, behavior is unchanged.
func TestSteeringInterruptPersistsUserMessageInHistory(t *testing.T) {
	emitCount := atomic.Int32{}
	model := &slowStreamingModel{
		emitCount:     &emitCount,
		eventsToEmit:  50,
		delayPerEvent: 10 * time.Millisecond,
		emitStarted:   make(chan struct{}),
	}

	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	steering := make(chan SteeringMsg, 1)
	events := ag.QueryStreamWithSteering(context.Background(), llm.TextContent("hello"), steering)

	select {
	case <-model.emitStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for stream to start")
	}

	time.Sleep(50 * time.Millisecond)
	steering <- SteeringMsg{Content: "switch topics now"}
	for range events {
	}

	messages := ag.Messages()
	found := false
	for _, msg := range messages {
		if msg.Role == llm.RoleUser && msg.PlainText() == "switch topics now" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected steering message to be persisted in history, messages=%#v", messages)
	}
}

func TestSteeringInterruptNilChannel(t *testing.T) {
	emitCount := atomic.Int32{}
	model := &slowStreamingModel{
		emitCount:     &emitCount,
		eventsToEmit:  10,
		delayPerEvent: 5 * time.Millisecond,
		emitStarted:   make(chan struct{}),
	}

	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	ctx := context.Background()

	// Use QueryStream (which passes nil steering channel)
	events := ag.QueryStream(ctx, llm.TextContent("hello"))

	// Collect all events
	collected := collectEvents(events)

	// Should complete normally without interruption
	var foundFinal bool
	for _, ev := range collected {
		if _, ok := ev.(FinalResponseEvent); ok {
			foundFinal = true
			break
		}
	}

	if !foundFinal {
		t.Error("expected final response event")
	}

	// All events should have been emitted
	if emitCount.Load() != int32(model.eventsToEmit) {
		t.Errorf("expected all %d events to be emitted, got %d", model.eventsToEmit, emitCount.Load())
	}
}

// TestSteeringInterruptNoMessage verifies that empty steering messages
// don't cause interruptions.
func TestSteeringInterruptNoMessage(t *testing.T) {
	emitCount := atomic.Int32{}
	model := &slowStreamingModel{
		emitCount:     &emitCount,
		eventsToEmit:  10,
		delayPerEvent: 5 * time.Millisecond,
		emitStarted:   make(chan struct{}),
	}

	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	steering := make(chan SteeringMsg, 1)
	ctx := context.Background()

	events := ag.QueryStreamWithSteering(ctx, llm.TextContent("hello"), steering)

	// Wait for streaming to start
	select {
	case <-model.emitStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for stream to start")
	}

	// Send empty steering message - should be ignored
	steering <- SteeringMsg{Content: ""}

	// Drain events
	for range events {
	}

	// All events should have been emitted (no interruption)
	if emitCount.Load() != int32(model.eventsToEmit) {
		t.Errorf("expected all %d events, got %d (empty steering should not interrupt)", model.eventsToEmit, emitCount.Load())
	}
}

// TestInvokeCompletionWithSteeringNilChannel verifies backward compatibility
// at the invokeCompletion level.
func TestInvokeCompletionWithSteeringNilChannel(t *testing.T) {
	model := &streamingResponseIDModel{}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	// Test with nil steering channel (backward compatible path)
	comp, streamedText, err := ag.invokeCompletionWithSteering(context.Background(), llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}, nil, nil)

	if err != nil {
		t.Fatalf("invoke completion with nil steering: %v", err)
	}
	if !streamedText {
		t.Fatal("expected streamed text=true")
	}
	if comp.ResponseID != "msg_stream_123" {
		t.Fatalf("expected response id msg_stream_123, got %q", comp.ResponseID)
	}
	if comp.PlainText() != "hello" {
		t.Fatalf("expected streamed text hello, got %q", comp.PlainText())
	}
}

// TestSteeringInterruptErrorIsNotRetryable verifies that steering
// interrupt errors are not retried.
func TestSteeringInterruptErrorIsNotRetryable(t *testing.T) {
	ag, err := New(Config{
		LLM:                    &slowStreamingModel{},
		InvokeRetryMaxAttempts: 5,
		InvokeRetryBackoff:     time.Millisecond,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	steering := make(chan SteeringMsg, 1)
	ctx := context.Background()

	// Send steering immediately
	steering <- SteeringMsg{Content: "stop"}

	// invokeCompletionWithRetryAndSteering should return immediately
	// without retrying
	start := time.Now()
	_, _, err = ag.invokeCompletionWithRetryAndSteering(ctx, llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hi")}},
	}, nil, steering)
	elapsed := time.Since(start)

	// Should return quickly (not after multiple retries)
	if elapsed > 200*time.Millisecond {
		t.Errorf("expected quick return on steering interrupt, got %v", elapsed)
	}

	// Error should be SteeringInterruptError
	var steerErr *llm.SteeringInterruptError
	if !errors.As(err, &steerErr) {
		t.Errorf("expected SteeringInterruptError, got %T: %v", err, err)
	}
}
