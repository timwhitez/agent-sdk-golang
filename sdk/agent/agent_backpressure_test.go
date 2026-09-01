package agent

import (
	"bytes"
	"context"
	"errors"
	"fmt"
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

// REG: emitEvent takes no ctx, so a query whose caller has already gone away used
// to pay the full criticalEventSendTimeoutFloor for every critical event even
// though nobody was left to read the channel. The floor itself is intentional and
// must stay in place for a live turn.
func TestCriticalEventFloorIsSkippedForCanceledTurn(t *testing.T) {
	ag, err := New(Config{
		LLM:              &completionOnlyModel{},
		EventBufferSize:  1,
		EventSendTimeout: 5 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	canceledOut := make(chan Event, 1)
	canceledDelivery := wrapLegacyEventOutput(canceledOut)
	canceledOut <- WarnEvent{Message: "filler"} // channel is now full
	canceledCtx, cancel := context.WithCancel(context.Background())
	defer ag.registerTurnCancellation(canceledDelivery, canceledCtx)()
	cancel()

	start := time.Now()
	if ag.emitEvent(canceledDelivery, ToolResultEvent{Tool: "read"}) {
		t.Fatal("expected the send into a full channel to fail")
	}
	if elapsed := time.Since(start); elapsed >= criticalEventSendTimeoutFloor {
		t.Fatalf("canceled turn paid the critical-event floor: %v >= %v", elapsed, criticalEventSendTimeoutFloor)
	}

	start = time.Now()
	for i := 0; i < 6; i++ {
		ag.emitEvent(canceledDelivery, ToolResultEvent{Tool: "read"})
	}
	if elapsed := time.Since(start); elapsed >= criticalEventSendTimeoutFloor {
		t.Fatalf("6 critical events on a canceled turn cost %v; expected far below one %v floor",
			elapsed, criticalEventSendTimeoutFloor)
	}

	liveOut := make(chan Event, 1)
	liveDelivery := wrapLegacyEventOutput(liveOut)
	liveOut <- WarnEvent{Message: "filler"}
	liveCtx, liveCancel := context.WithCancel(context.Background())
	defer liveCancel()
	defer ag.registerTurnCancellation(liveDelivery, liveCtx)()
	start = time.Now()
	ag.emitEvent(liveDelivery, ToolResultEvent{Tool: "read"})
	if elapsed := time.Since(start); elapsed < criticalEventSendTimeoutFloor {
		t.Fatalf("live turn lost the deliberate critical-event floor: %v < %v", elapsed, criticalEventSendTimeoutFloor)
	}
}

// REG: critical drops returned before the dropped%logEvery gate, so sustained
// backpressure emitted one warn line per drop. The exact count still reaches the
// consumer through FinalResponseEvent.DroppedCriticalEvents, so the log line is
// sampled like every other drop (with the first one always reported).
func TestCriticalEventDropLoggingIsSampled(t *testing.T) {
	var warnings lockedBuffer
	ag, err := New(Config{
		LLM:               &completionOnlyModel{},
		EventDropLogEvery: 5,
		Warningf: func(format string, args ...any) {
			_, _ = warnings.Write([]byte(fmt.Sprintf(format, args...) + "\n"))
		},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	const drops = 20
	for i := 0; i < drops; i++ {
		ag.logDroppedEvent(ToolResultEvent{Tool: "read"}, "channel_full")
	}

	lines := 0
	for _, line := range strings.Split(warnings.String(), "\n") {
		if strings.TrimSpace(line) != "" {
			lines++
		}
	}
	if lines >= drops {
		t.Fatalf("critical drop logging is not sampled: %d drops produced %d warn lines", drops, lines)
	}
	if lines == 0 {
		t.Fatal("expected at least the first critical drop to be logged")
	}
	if got := ag.criticalEventDropCount.Load(); got != drops {
		t.Fatalf("critical drop counter = %d, want %d; sampling must not lose the exact count", got, drops)
	}
	if got := ag.eventDropCount.Load(); got != drops {
		t.Fatalf("total drop counter = %d, want %d", got, drops)
	}
}

// REG (R4-CC-005): isCriticalAgentEvent covers per-step kinds (StepStart,
// ToolCall, ToolResult, StepComplete plus Accounting/Usage), i.e. roughly seven
// critical events per tool call. Paying criticalEventSendTimeoutFloor for each
// of them without a per-turn cap made a tool-heavy turn against a stalled
// consumer cost the floor times the event count: 700 events (~100 tool calls)
// measured ~175s of pure waiting. The floor must stay bounded per turn while the
// drop accounting stays exact.
func TestCriticalEventFloorIsBoundedPerTurn(t *testing.T) {
	ag, err := New(Config{
		LLM:             &completionOnlyModel{},
		EventBufferSize: 1,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	out := make(chan Event, 1)
	delivery := wrapLegacyEventOutput(out)
	out <- WarnEvent{Message: "filler"} // full channel, and nothing drains it
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	defer ag.registerTurnCancellation(delivery, ctx)()

	const events = 700 // ~100 tool calls x ~7 critical events
	start := time.Now()
	for i := 0; i < events; i++ {
		ag.emitEvent(delivery, ToolResultEvent{Tool: "read"})
	}
	elapsed := time.Since(start)

	// Without the cap this is events * criticalEventSendTimeoutFloor (~175s).
	uncapped := time.Duration(events) * criticalEventSendTimeoutFloor
	if elapsed >= uncapped/2 {
		t.Fatalf("%d critical events cost %v; the per-turn floor budget (%v) is not bounding the wait (uncapped would be ~%v)",
			events, elapsed, criticalEventFloorTurnBudget, uncapped)
	}
	// The budget must actually be spendable, not skipped outright.
	if elapsed < criticalEventFloorTurnBudget {
		t.Fatalf("%d critical events cost %v, below the %v floor budget: the deliberate floor was dropped entirely",
			events, elapsed, criticalEventFloorTurnBudget)
	}
	// ISS-129b: drops must still be counted exactly, never silently discarded.
	if got := ag.criticalEventDropCount.Load(); got != events {
		t.Fatalf("critical drop counter = %d, want %d; capping the floor must not lose the exact count", got, events)
	}
}

// REG (R4-CC-005): the per-turn cap must not eat the floor for the events that
// arrive before the budget is spent, and a terminal event keeps the floor even
// afterwards - there is at most one per turn, so it cannot multiply the cost, and
// losing the turn's outcome is the worst possible drop.
func TestCriticalEventFloorBudgetStillPaysTheFirstEvents(t *testing.T) {
	ag, err := New(Config{
		LLM:              &completionOnlyModel{},
		EventBufferSize:  1,
		EventSendTimeout: 5 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	out := make(chan Event, 1)
	delivery := wrapLegacyEventOutput(out)
	out <- WarnEvent{Message: "filler"}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	defer ag.registerTurnCancellation(delivery, ctx)()

	start := time.Now()
	ag.emitEvent(delivery, ToolResultEvent{Tool: "read"})
	if elapsed := time.Since(start); elapsed < criticalEventSendTimeoutFloor {
		t.Fatalf("the first critical event of a turn lost the deliberate floor: %v < %v",
			elapsed, criticalEventSendTimeoutFloor)
	}

	// Spend the rest of the turn's budget.
	deadline := time.Now().Add(2 * criticalEventFloorTurnBudget)
	for time.Now().Before(deadline) {
		before := time.Now()
		ag.emitEvent(delivery, ToolResultEvent{Tool: "read"})
		if time.Since(before) < criticalEventSendTimeoutFloor {
			break // budget spent, critical events fell back to the ordinary budget
		}
	}
	start = time.Now()
	ag.emitEvent(delivery, ToolResultEvent{Tool: "read"})
	if elapsed := time.Since(start); elapsed >= criticalEventSendTimeoutFloor {
		t.Fatalf("a critical event still paid the full floor after the turn budget was spent: %v", elapsed)
	}

	// A fresh turn gets a fresh budget.
	freshOut := make(chan Event, 1)
	freshDelivery := wrapLegacyEventOutput(freshOut)
	freshOut <- WarnEvent{Message: "filler"}
	freshCtx, freshCancel := context.WithCancel(context.Background())
	defer freshCancel()
	defer ag.registerTurnCancellation(freshDelivery, freshCtx)()
	start = time.Now()
	ag.emitEvent(freshDelivery, ToolResultEvent{Tool: "read"})
	if elapsed := time.Since(start); elapsed < criticalEventSendTimeoutFloor {
		t.Fatalf("a new turn did not get a fresh floor budget: %v < %v", elapsed, criticalEventSendTimeoutFloor)
	}
}
