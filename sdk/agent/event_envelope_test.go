package agent

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type envelopeFinalModel struct{}

type envelopeToolLoopModel struct{}

type envelopeErrorModel struct {
	calls atomic.Int64
	err   error
}

func (m *envelopeErrorModel) Provider() string { return "fixture" }
func (m *envelopeErrorModel) Model() string    { return "fixture" }
func (m *envelopeErrorModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls.Add(1)
	return nil, m.err
}

func (envelopeToolLoopModel) Provider() string { return "fixture" }
func (envelopeToolLoopModel) Model() string    { return "fixture" }
func (envelopeToolLoopModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "call-loop", Type: "function", Function: llm.FunctionCall{Name: "missing", Arguments: `{}`}}}, StopReason: "tool_calls"}, nil
}

func wrapLegacyEventOutput(ch chan Event) *eventOutput {
	return &eventOutput{legacy: ch, queryID: "test-query", clock: time.Now}
}

func (envelopeFinalModel) Provider() string { return "fixture" }
func (envelopeFinalModel) Model() string    { return "fixture" }
func (envelopeFinalModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("done"), StopReason: "stop"}, nil
}

func collectEnvelopes(ch <-chan EventEnvelope) []EventEnvelope {
	var envelopes []EventEnvelope
	for envelope := range ch {
		envelopes = append(envelopes, envelope)
	}
	return envelopes
}

func TestEventEnvelopeQuerySequenceAndMetadata(t *testing.T) {
	fixed := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	agent, err := New(Config{
		LLM:              envelopeFinalModel{},
		QueryIDGenerator: func() string { return "query-test" },
		EventClock:       func() time.Time { return fixed },
	})
	if err != nil {
		t.Fatal(err)
	}

	envelopes := collectEnvelopes(agent.QueryStreamEnveloped(context.Background(), llm.TextContent("secret prompt")))
	if len(envelopes) != 2 {
		t.Fatalf("envelope count=%d want 2: %#v", len(envelopes), envelopes)
	}
	wantKinds := []EventKind{EventKindText, EventKindFinalResponse}
	for i, envelope := range envelopes {
		if envelope.SchemaVersion != EventEnvelopeSchemaVersion || envelope.QueryID != "query-test" {
			t.Fatalf("envelope[%d] identity=%#v", i, envelope)
		}
		if envelope.Sequence != uint64(i+1) || envelope.Kind != wantKinds[i] || !envelope.Timestamp.Equal(fixed) {
			t.Fatalf("envelope[%d] ordering metadata=%#v", i, envelope)
		}
	}
	if envelopes[0].Origin != EventOriginModel || envelopes[1].Origin != EventOriginSDKDriver {
		t.Fatalf("origins=%q/%q", envelopes[0].Origin, envelopes[1].Origin)
	}
}

func TestEventEnvelopeClassifiesAllTypedEvents(t *testing.T) {
	tests := []struct {
		event  Event
		kind   EventKind
		origin EventOrigin
	}{
		{TextEvent{}, EventKindText, EventOriginModel},
		{TextDeltaEvent{}, EventKindTextDelta, EventOriginModel},
		{ThinkingEvent{}, EventKindThinking, EventOriginModel},
		{ThinkingDeltaEvent{}, EventKindThinkingDelta, EventOriginModel},
		{ErrorEvent{}, EventKindError, EventOriginSDKDriver},
		{ErrorEvent{Provider: "fixture"}, EventKindError, EventOriginProvider},
		{ErrorEvent{Provider: "fixture", Kind: "canceled"}, EventKindError, EventOriginProvider},
		{ErrorEvent{Provider: "fixture", Kind: "max_iterations"}, EventKindError, EventOriginSDKDriver},
		{WarnEvent{}, EventKindWarning, EventOriginSDKDriver},
		{HiddenUserMessageEvent{}, EventKindHiddenUserMessage, EventOriginSDKDriver},
		{StepStartEvent{}, EventKindStepStart, EventOriginToolRuntime},
		{StepCompleteEvent{}, EventKindStepComplete, EventOriginToolRuntime},
		{ToolCallEvent{}, EventKindToolCall, EventOriginToolRuntime},
		{ToolResultEvent{}, EventKindToolResult, EventOriginToolRuntime},
		{FinalResponseEvent{}, EventKindFinalResponse, EventOriginSDKDriver},
		{UsageEvent{}, EventKindUsage, EventOriginProvider},
		{CompactionEvent{}, EventKindCompaction, EventOriginCompaction},
		{AccountingEvent{}, EventKindAccounting, EventOriginSDKDriver},
		{SteeringReceivedEvent{}, EventKindSteeringReceived, EventOriginHost},
		{AutoContinueEvent{}, EventKindAutoContinue, EventOriginSDKDriver},
	}
	for _, test := range tests {
		kind, origin := classifyEvent(test.event)
		if kind != test.kind || origin != test.origin {
			t.Fatalf("%T classified as %q/%q want %q/%q", test.event, kind, origin, test.kind, test.origin)
		}
	}
}

func TestEventEnvelopeTerminalPriorityRejectsOnce(t *testing.T) {
	agent := &Agent{eventSendTimeout: time.Millisecond}
	out := newEventOutput(1, true, "query-test", time.Now)
	errorEnvelope := out.next(ErrorEvent{Provider: "fixture", Kind: "provider", Message: "failed"})
	if !out.trySend(errorEnvelope) {
		t.Fatal("failed to fill envelope channel")
	}

	if agent.emitEvent(out, FinalResponseEvent{Content: "must not follow error"}) {
		t.Fatal("lower-priority final was delivered after terminal error")
	}
	if got := agent.eventDropCount.Load(); got != 1 {
		t.Fatalf("drop count=%d want exactly 1", got)
	}
	buffered := <-out.enveloped
	if buffered.Sequence != 1 || buffered.Kind != EventKindError {
		t.Fatalf("buffered terminal changed: %#v", buffered)
	}
	select {
	case extra := <-out.enveloped:
		t.Fatalf("lower-priority terminal remained deliverable: %#v", extra)
	default:
	}
}

func TestEventEnvelopeSDKTerminalErrorsKeepSDKOrigin(t *testing.T) {
	canceledAgent, err := New(Config{LLM: envelopeFinalModel{}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	canceled := collectEnvelopes(canceledAgent.QueryStreamEnveloped(ctx, llm.TextContent("run")))
	assertEnvelopeErrorOrigin(t, canceled, "canceled", EventOriginSDKDriver)

	maxAgent, err := New(Config{LLM: envelopeToolLoopModel{}, MaxIterations: 1})
	if err != nil {
		t.Fatal(err)
	}
	maximum := collectEnvelopes(maxAgent.QueryStreamEnveloped(context.Background(), llm.TextContent("run")))
	assertEnvelopeErrorOrigin(t, maximum, "max_iterations", EventOriginSDKDriver)
}

func TestEventEnvelopeCancellationOriginUsesEmissionBoundary(t *testing.T) {
	rootModel := &envelopeErrorModel{err: context.DeadlineExceeded}
	rootAgent, err := New(Config{LLM: rootModel})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	rootTimeout := collectEnvelopes(rootAgent.QueryStreamEnveloped(ctx, llm.TextContent("run")))
	assertEnvelopeErrorOrigin(t, rootTimeout, "timeout", EventOriginSDKDriver)
	if calls := rootModel.calls.Load(); calls != 0 {
		t.Fatalf("expired root deadline invoked provider %d times", calls)
	}

	for _, test := range []struct {
		name string
		err  error
		kind string
	}{
		{name: "deadline", err: context.DeadlineExceeded, kind: "timeout"},
		{name: "canceled", err: context.Canceled, kind: "canceled"},
	} {
		t.Run(test.name, func(t *testing.T) {
			providerModel := &envelopeErrorModel{err: test.err}
			providerAgent, err := New(Config{LLM: providerModel})
			if err != nil {
				t.Fatal(err)
			}
			providerError := collectEnvelopes(providerAgent.QueryStreamEnveloped(context.Background(), llm.TextContent("run")))
			assertEnvelopeErrorOrigin(t, providerError, test.kind, EventOriginProvider)
			if calls := providerModel.calls.Load(); calls == 0 {
				t.Fatalf("provider %s did not invoke provider", test.name)
			}
		})
	}
}

func assertEnvelopeErrorOrigin(t *testing.T, envelopes []EventEnvelope, kind string, want EventOrigin) {
	t.Helper()
	for _, envelope := range envelopes {
		event, ok := envelope.Event.(ErrorEvent)
		if ok && event.Kind == kind {
			if envelope.Origin != want {
				t.Fatalf("%s origin=%q want %q: %#v", kind, envelope.Origin, want, envelope)
			}
			return
		}
	}
	t.Fatalf("missing %s error envelope: %#v", kind, envelopes)
}

func TestEventEnvelopeLegacyProjectionPreservesPayloadOrder(t *testing.T) {
	legacyAgent, err := New(Config{LLM: envelopeFinalModel{}})
	if err != nil {
		t.Fatal(err)
	}
	envelopedAgent, err := New(Config{LLM: envelopeFinalModel{}})
	if err != nil {
		t.Fatal(err)
	}

	legacy := collectEvents(legacyAgent.QueryStream(context.Background(), llm.TextContent("run")))
	envelopes := collectEnvelopes(envelopedAgent.QueryStreamEnveloped(context.Background(), llm.TextContent("run")))
	payloads := make([]Event, len(envelopes))
	for i, envelope := range envelopes {
		payloads[i] = envelope.Event
	}
	if !reflect.DeepEqual(payloads, legacy) {
		t.Fatalf("legacy payloads changed\nlegacy=%#v\nenveloped=%#v", legacy, payloads)
	}
}

func TestEventEnvelopeAllocatesSequenceBeforeDrop(t *testing.T) {
	agent, err := New(Config{LLM: envelopeFinalModel{}, EventBufferSize: 1, EventSendTimeout: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}

	events := agent.QueryStreamEnveloped(context.Background(), llm.TextContent("run"))
	time.Sleep(25 * time.Millisecond)
	envelopes := collectEnvelopes(events)
	if len(envelopes) != 1 {
		t.Fatalf("delivered envelopes=%d want terminal only: %#v", len(envelopes), envelopes)
	}
	final, ok := envelopes[0].Event.(FinalResponseEvent)
	if !ok || envelopes[0].Kind != EventKindFinalResponse {
		t.Fatalf("terminal envelope=%#v", envelopes[0])
	}
	if envelopes[0].Sequence != 2 {
		t.Fatalf("terminal sequence=%d want 2 to expose dropped sequence 1", envelopes[0].Sequence)
	}
	if final.DroppedEvents != 1 {
		t.Fatalf("terminal dropped events=%d want 1", final.DroppedEvents)
	}

	legacyAgent, err := New(Config{LLM: envelopeFinalModel{}, EventBufferSize: 1, EventSendTimeout: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	legacyStream := legacyAgent.QueryStream(context.Background(), llm.TextContent("run"))
	time.Sleep(25 * time.Millisecond)
	legacy := collectEvents(legacyStream)
	if len(legacy) != 1 {
		t.Fatalf("legacy delivered events=%d want terminal only: %#v", len(legacy), legacy)
	}
	legacyFinal, ok := legacy[0].(FinalResponseEvent)
	if !ok || legacyFinal.DroppedEvents != final.DroppedEvents {
		t.Fatalf("legacy/enveloped drop parity failed: legacy=%#v envelope=%#v", legacy, envelopes)
	}
}

func TestEventEnvelopeEmptyGeneratedQueryIDFallsBack(t *testing.T) {
	agent, err := New(Config{LLM: envelopeFinalModel{}, QueryIDGenerator: func() string { return "  " }})
	if err != nil {
		t.Fatal(err)
	}
	envelopes := collectEnvelopes(agent.QueryStreamEnveloped(context.Background(), llm.TextContent("secret prompt")))
	if len(envelopes) == 0 || envelopes[0].QueryID == "" || envelopes[0].QueryID == "secret prompt" {
		t.Fatalf("fallback query ID=%q", envelopes[0].QueryID)
	}
}

func TestEventEnvelopeQueryIDsArePerTurn(t *testing.T) {
	var ids atomic.Uint64
	agent, err := New(Config{
		LLM: envelopeFinalModel{},
		QueryIDGenerator: func() string {
			return fmt.Sprintf("query-%d", ids.Add(1))
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	first := collectEnvelopes(agent.QueryStreamEnveloped(context.Background(), llm.TextContent("first")))
	second := collectEnvelopes(agent.QueryStreamEnveloped(context.Background(), llm.TextContent("second")))
	if len(first) == 0 || len(second) == 0 || first[0].QueryID != "query-1" || second[0].QueryID != "query-2" {
		t.Fatalf("query IDs=%#v/%#v", first, second)
	}
	if first[0].Sequence != 1 || second[0].Sequence != 1 {
		t.Fatalf("per-query sequences did not restart: %d/%d", first[0].Sequence, second[0].Sequence)
	}
}

func TestEventEnvelopeBusyAdmissionIsEnveloped(t *testing.T) {
	model := &singleActiveTurnModel{firstStart: make(chan struct{}), firstFinish: make(chan struct{})}
	var idMu sync.Mutex
	nextID := 0
	agent, err := New(Config{
		LLM: model,
		QueryIDGenerator: func() string {
			idMu.Lock()
			defer idMu.Unlock()
			nextID++
			return fmt.Sprintf("query-%d", nextID)
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	first := agent.QueryStreamEnveloped(context.Background(), llm.TextContent("first"))
	select {
	case <-model.firstStart:
	case <-time.After(5 * time.Second):
		t.Fatal("first provider invocation did not start")
	}
	busy := collectEnvelopes(agent.QueryStreamEnveloped(context.Background(), llm.TextContent("second")))
	if len(busy) != 1 || busy[0].Sequence != 1 || busy[0].QueryID == "" || busy[0].Kind != EventKindError {
		t.Fatalf("busy envelope=%#v", busy)
	}
	if event, ok := busy[0].Event.(ErrorEvent); !ok || event.Kind != "agent_busy" {
		t.Fatalf("busy payload=%#v", busy[0].Event)
	}
	close(model.firstFinish)
	collectEnvelopes(first)
}
