package agent

import (
	"bytes"
	"context"
	"errors"
	"log"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type burstStreamingModel struct {
	deltaCount int
	streamDone chan struct{}
}

func (m *burstStreamingModel) Provider() string { return "stub" }
func (m *burstStreamingModel) Model() string    { return "stub" }

func (m *burstStreamingModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}

func (m *burstStreamingModel) InvokeStream(_ context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent, 1)
	go func() {
		defer close(ch)
		if m.streamDone != nil {
			defer close(m.streamDone)
		}
		for i := 0; i < m.deltaCount; i++ {
			ch <- llm.StreamTextDeltaEvent{Delta: "x"}
		}
		ch <- llm.StreamDoneEvent{StopReason: "stop"}
	}()
	return ch, nil
}

func TestQueryStreamBackpressureDropsWithoutDeadlock(t *testing.T) {
	var logBuf bytes.Buffer
	origOut := log.Writer()
	origFlags := log.Flags()
	log.SetOutput(&logBuf)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(origOut)
		log.SetFlags(origFlags)
	})

	model := &burstStreamingModel{
		deltaCount: 256,
		streamDone: make(chan struct{}),
	}
	ag, err := New(Config{
		LLM:               model,
		EventBufferSize:   4,
		EventSendTimeout:  2 * time.Millisecond,
		EventDropLogEvery: 1,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := ag.QueryStream(context.Background(), llm.TextContent("hi"))

	// Do not consume events immediately to simulate a stalled UI consumer.
	select {
	case <-model.streamDone:
	case <-time.After(3 * time.Second):
		t.Fatal("stream producer blocked by event backpressure")
	}

	deadline := time.After(3 * time.Second)
	for {
		select {
		case _, ok := <-events:
			if !ok {
				if drops := ag.eventDropCount.Load(); drops == 0 {
					t.Fatal("expected dropped events under backpressure")
				}
				if !strings.Contains(logBuf.String(), "dropping agent event") {
					t.Fatalf("expected backpressure warning log, got %q", logBuf.String())
				}
				return
			}
		case <-deadline:
			t.Fatal("event channel did not close after producer finished")
		}
	}
}

func TestQueryStreamUsesConfiguredEventBufferSize(t *testing.T) {
	ag, err := New(Config{
		LLM:             &streamingResponseIDModel{},
		EventBufferSize: 9,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := ag.QueryStream(context.Background(), llm.TextContent("hello"))
	if got := cap(events); got != 9 {
		t.Fatalf("expected event buffer size 9, got %d", got)
	}
	for range events {
	}
}

func TestQueryStreamBackpressureDoesNotLeakOnTerminalEvent(t *testing.T) {
	baseline := runtime.NumGoroutine()
	const runs = 20
	models := make([]*burstStreamingModel, 0, runs)
	channels := make([]<-chan Event, 0, runs)

	for i := 0; i < runs; i++ {
		model := &burstStreamingModel{deltaCount: 64, streamDone: make(chan struct{})}
		ag, err := New(Config{
			LLM:               model,
			EventBufferSize:   1,
			EventSendTimeout:  5 * time.Millisecond,
			EventDropLogEvery: 1,
		})
		if err != nil {
			t.Fatalf("new agent: %v", err)
		}
		models = append(models, model)
		channels = append(channels, ag.QueryStream(context.Background(), llm.TextContent("hi")))
	}

	for _, model := range models {
		select {
		case <-model.streamDone:
		case <-time.After(3 * time.Second):
			t.Fatal("stream producer blocked by event backpressure")
		}
	}

	time.Sleep(300 * time.Millisecond)
	runtime.GC()
	time.Sleep(100 * time.Millisecond)

	after := runtime.NumGoroutine()
	if delta := after - baseline; delta > 8 {
		t.Fatalf("expected no large goroutine leak after terminal backpressure, baseline=%d after=%d delta=%d", baseline, after, delta)
	}

	_ = channels
}

func TestQueryStreamBackpressureStillDeliversFinalEvent(t *testing.T) {
	model := &burstStreamingModel{
		deltaCount: 256,
		streamDone: make(chan struct{}),
	}
	ag, err := New(Config{
		LLM:               model,
		EventBufferSize:   4,
		EventSendTimeout:  2 * time.Millisecond,
		EventDropLogEvery: 1,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := ag.QueryStream(context.Background(), llm.TextContent("hi"))

	select {
	case <-model.streamDone:
	case <-time.After(3 * time.Second):
		t.Fatal("stream producer blocked by event backpressure")
	}

	deadline := time.After(3 * time.Second)
	finalCount := 0
	for {
		select {
		case ev, ok := <-events:
			if !ok {
				if finalCount != 1 {
					t.Fatalf("expected exactly one FinalResponseEvent, got %d", finalCount)
				}
				return
			}
			if _, ok := ev.(FinalResponseEvent); ok {
				finalCount++
			}
		case <-deadline:
			t.Fatal("timed out waiting for terminal event under backpressure")
		}
	}
}

type completionOnlyModel struct{}

func (m *completionOnlyModel) Provider() string { return "stub" }
func (m *completionOnlyModel) Model() string    { return "stub" }

func (m *completionOnlyModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("done"), StopReason: "stop"}, nil
}

type errorStreamingModel struct {
	deltaCount int
	streamDone chan struct{}
}

func (m *errorStreamingModel) Provider() string { return "stub" }
func (m *errorStreamingModel) Model() string    { return "stub" }

func (m *errorStreamingModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("invoke should not be called")
}

func (m *errorStreamingModel) InvokeStream(_ context.Context, _ llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent, 1)
	go func() {
		defer close(ch)
		if m.streamDone != nil {
			defer close(m.streamDone)
		}
		for i := 0; i < m.deltaCount; i++ {
			ch <- llm.StreamTextDeltaEvent{Delta: "x"}
		}
		ch <- llm.StreamErrorEvent{Err: errors.New("boom")}
	}()
	return ch, nil
}

func TestQueryStreamBackpressureBlocksUntilFinalEventIsConsumed(t *testing.T) {
	ag, err := New(Config{
		LLM:               &completionOnlyModel{},
		EventBufferSize:   1,
		EventSendTimeout:  2 * time.Millisecond,
		EventDropLogEvery: 1,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := ag.QueryStream(context.Background(), llm.TextContent("hi"))
	time.Sleep(25 * time.Millisecond)

	deadline := time.After(2 * time.Second)
	finalCount := 0
	for {
		select {
		case ev, ok := <-events:
			if !ok {
				if finalCount != 1 {
					t.Fatalf("expected exactly one FinalResponseEvent, got %d", finalCount)
				}
				return
			}
			if _, ok := ev.(FinalResponseEvent); ok {
				finalCount++
			}
		case <-deadline:
			t.Fatal("timed out waiting for final response event")
		}
	}
}

func TestQueryStreamBackpressureBlocksUntilErrorEventIsConsumed(t *testing.T) {
	model := &errorStreamingModel{deltaCount: 1, streamDone: make(chan struct{})}
	ag, err := New(Config{
		LLM:               model,
		EventBufferSize:   1,
		EventSendTimeout:  2 * time.Millisecond,
		EventDropLogEvery: 1,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := ag.QueryStream(context.Background(), llm.TextContent("hi"))
	select {
	case <-model.streamDone:
	case <-time.After(2 * time.Second):
		t.Fatal("stream producer blocked before emitting error")
	}
	time.Sleep(25 * time.Millisecond)

	deadline := time.After(2 * time.Second)
	errorCount := 0
	for {
		select {
		case ev, ok := <-events:
			if !ok {
				if errorCount != 1 {
					t.Fatalf("expected exactly one ErrorEvent, got %d", errorCount)
				}
				return
			}
			if errEv, ok := ev.(ErrorEvent); ok {
				errorCount++
				if errEv.Message == "" {
					t.Fatal("expected error event message")
				}
			}
		case <-deadline:
			t.Fatal("timed out waiting for error event")
		}
	}
}

type diagnosticCompletionModel struct{}

func (m *diagnosticCompletionModel) Provider() string { return "stub" }
func (m *diagnosticCompletionModel) Model() string    { return "stub" }
func (m *diagnosticCompletionModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{
		Content: llm.TextContent("done"),
		Diagnostics: []llm.Diagnostic{
			{Kind: "provider_compatibility_downgrade", Message: "retrying without unsupported option"},
			{Message: "diagnostic without kind"},
			{Kind: "empty", Message: "   "},
		},
	}, nil
}

func TestQueryStreamEmitsCompletionDiagnosticsAsWarnings(t *testing.T) {
	ag, err := New(Config{LLM: &diagnosticCompletionModel{}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := ag.QueryStream(context.Background(), llm.TextContent("hi"))
	warnings := []WarnEvent{}
	finalSeen := false
	for ev := range events {
		switch e := ev.(type) {
		case WarnEvent:
			warnings = append(warnings, e)
		case FinalResponseEvent:
			finalSeen = true
		}
	}
	if !finalSeen {
		t.Fatal("expected final response")
	}
	if len(warnings) != 2 {
		t.Fatalf("warning count = %d, want 2 (%#v)", len(warnings), warnings)
	}
	if warnings[0].Kind != "provider_compatibility_downgrade" || !strings.Contains(warnings[0].Message, "unsupported option") {
		t.Fatalf("unexpected first warning: %#v", warnings[0])
	}
	if warnings[1].Kind != "provider_diagnostic" || warnings[1].Message != "diagnostic without kind" {
		t.Fatalf("unexpected default-kind warning: %#v", warnings[1])
	}
}

func TestTerminalEventPriorityKeepsErrorsAboveFinalResponses(t *testing.T) {
	if terminalEventPriority(ErrorEvent{}) <= terminalEventPriority(FinalResponseEvent{}) {
		t.Fatalf("ErrorEvent priority should be higher than FinalResponseEvent")
	}
}
