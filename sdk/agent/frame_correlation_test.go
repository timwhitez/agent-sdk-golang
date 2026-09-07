package agent

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestFailedFrameMaterializationHasNoInvocationCorrelation(t *testing.T) {
	calls := 0
	model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) { calls++; return &llm.Completion{}, nil }}
	ag, err := New(Config{LLM: model, QueryIDGenerator: func() string { return "failed-frame" }, Warningf: func(string, ...any) {}, Tools: []tools.Tool{{Name: "work", Schema: map[string]any{"type": "object"}}}})
	if err != nil {
		t.Fatal(err)
	}
	ag.tools[0].Schema["PRIVATE_SCHEMA_FAILURE"] = make(chan int)
	count := 0
	for e := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("run")) {
		count++
		failure, ok := e.Event.(ErrorEvent)
		if !ok || failure.Kind != "invalid_request" || e.FrameID != "" || e.InvokeAttempt != 0 || e.Origin != EventOriginSDKDriver || strings.Contains(failure.Message, "PRIVATE_") {
			t.Errorf("invalid-frame event=%+v", e)
		}
	}
	if count != 1 || calls != 0 {
		t.Fatalf("events=%d model calls=%d", count, calls)
	}
}

func BenchmarkEventEnvelopeFrameCorrelation(b *testing.B) {
	for _, scoped := range []bool{false, true} {
		name := "absent"
		if scoped {
			name = "explicit"
		}
		b.Run(name, func(b *testing.B) {
			ag, err := New(Config{LLM: envelopeFinalModel{}})
			if err != nil {
				b.Fatal(err)
			}
			out := newEventOutput(1, true, "query", func() time.Time { return time.Unix(100, 0) })
			correlation := eventCorrelation{frameID: "query/frame/1", attempt: 2}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if scoped {
					ag.emitEvent(out, TextEvent{Content: "fixture"}, correlation)
				} else {
					ag.emitEvent(out, TextEvent{Content: "fixture"})
				}
				<-out.enveloped
			}
		})
	}
}

type correlatedStreamModel struct{ calls int }

func (*correlatedStreamModel) Provider() string { return "fixture" }
func (*correlatedStreamModel) Model() string    { return "stream" }
func (*correlatedStreamModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("unexpected buffered invocation")
}
func (m *correlatedStreamModel) InvokeStream(context.Context, llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.calls++
	if m.calls == 1 {
		return nil, &net.DNSError{Err: "fixture", IsTimeout: true}
	}
	ch := make(chan llm.StreamEvent, 5)
	ch <- llm.StreamRetryEvent{Attempt: 7, MaxRetries: 9, Message: "provider-reported retry"}
	ch <- llm.StreamThinkingDeltaEvent{Delta: "synthetic thinking fixture"}
	ch <- llm.StreamTextDeltaEvent{Delta: "done"}
	ch <- llm.StreamUsageEvent{Usage: llm.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12, PromptTokensValid: true, PromptTokensSource: llm.PromptTokensSourceProvider, PromptTokensSemantics: llm.PromptTokensSemanticsTotalInputV1}}
	ch <- llm.StreamDoneEvent{StopReason: "stop"}
	close(ch)
	return ch, nil
}

func TestStreamFrameCorrelationCountsSDKEntriesNotProviderRetryReports(t *testing.T) {
	m := &correlatedStreamModel{}
	ag, err := New(Config{LLM: m, InvokeRetryMaxAttempts: 2, QueryIDGenerator: func() string { return "stream-query" }, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	seen := map[EventKind]int{}
	for e := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("run")) {
		seen[e.Kind]++
		if e.FrameID != "stream-query/frame/1" || e.InvokeAttempt != 2 {
			t.Errorf("kind=%s frame=%s attempt=%d", e.Kind, e.FrameID, e.InvokeAttempt)
		}
	}
	if m.calls != 2 || seen[EventKindTextDelta] != 1 || seen[EventKindThinkingDelta] != 1 || seen[EventKindUsage] != 1 || seen[EventKindWarning] != 1 || seen[EventKindFinalResponse] != 1 {
		t.Fatalf("SDK calls=%d kinds=%v", m.calls, seen)
	}
}

func TestFrameCorrelationDoesNotInventAnInvocationOnCancellation(t *testing.T) {
	for _, beforeAdmission := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		calls := 0
		model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
			calls++
			return nil, &net.DNSError{Err: "fixture", IsTimeout: true}
		}}
		ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2, QueryIDGenerator: func() string { return "cancel-query" }, Warningf: func(string, ...any) { cancel() }})
		if err != nil {
			cancel()
			t.Fatal(err)
		}
		if beforeAdmission {
			cancel()
		}
		for e := range ag.QueryStreamEnveloped(ctx, llm.TextContent("run")) {
			failure, ok := e.Event.(ErrorEvent)
			if !ok || failure.Kind != "canceled" || e.InvokeAttempt != 0 {
				t.Errorf("cancel event=%+v", e)
			}
			wantFrame := "cancel-query/frame/1"
			if beforeAdmission {
				wantFrame = ""
			}
			if e.FrameID != wantFrame {
				t.Errorf("frame=%s want=%s", e.FrameID, wantFrame)
			}
		}
		cancel()
		wantCalls := 1
		if beforeAdmission {
			wantCalls = 0
		}
		if calls != wantCalls {
			t.Fatalf("calls=%d want=%d", calls, wantCalls)
		}
	}
}

func TestBackoffCancellationKeepsRetainedUsageOnItsActualInvoke(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls := 0
	model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
		calls++
		return &llm.Completion{ResponseID: "partial", Usage: &llm.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12, PromptTokensValid: true, PromptTokensSource: llm.PromptTokensSourceProvider, PromptTokensSemantics: llm.PromptTokensSemanticsTotalInputV1}}, &net.DNSError{Err: "fixture", IsTimeout: true}
	}}
	ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2, QueryIDGenerator: func() string { return "usage-query" }, Warningf: func(string, ...any) { cancel() }})
	if err != nil {
		t.Fatal(err)
	}
	seen := map[EventKind]int{}
	for e := range ag.QueryStreamEnveloped(ctx, llm.TextContent("run")) {
		seen[e.Kind]++
		if e.FrameID != "usage-query/frame/1" {
			t.Errorf("frame=%s", e.FrameID)
		}
		switch e.Kind {
		case EventKindUsage, EventKindAccounting:
			if e.InvokeAttempt != 1 {
				t.Errorf("retained usage mislabeled attempt=%d", e.InvokeAttempt)
			}
		case EventKindError:
			if e.InvokeAttempt != 0 {
				t.Error("backoff cancel blamed the finished invocation")
			}
		default:
			t.Errorf("unexpected event %s", e.Kind)
		}
	}
	if calls != 1 || seen[EventKindUsage] != 1 || seen[EventKindAccounting] != 1 || seen[EventKindError] != 1 {
		t.Fatalf("calls=%d events=%v", calls, seen)
	}
}

func TestExplicitFrameCorrelationDoesNotLeakIntoConcurrentUnscopedEvents(t *testing.T) {
	ag, err := New(Config{LLM: envelopeFinalModel{}})
	if err != nil {
		t.Fatal(err)
	}
	out := newEventOutput(200, true, "query", time.Now)
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			ag.emitEvent(out, TextEvent{Content: "text"}, eventCorrelation{frameID: "explicit", attempt: 3})
		}()
		go func() { defer wg.Done(); ag.emitEvent(out, WarnEvent{Kind: "host_observation"}) }()
	}
	wg.Wait()
	ag.emitCompactionWithAccounting(out, CompactionEvent{})
	ag.emitEvent(out, SteeringReceivedEvent{})
	out.close()
	seen := map[uint64]bool{}
	for e := range out.enveloped {
		if seen[e.Sequence] {
			t.Error("duplicate sequence")
		}
		seen[e.Sequence] = true
		if e.Kind == EventKindText {
			if e.FrameID != "explicit" || e.InvokeAttempt != 3 {
				t.Error("explicit correlation lost")
			}
		} else if e.FrameID != "" || e.InvokeAttempt != 0 {
			t.Error("unscoped event inherited an ambient frame")
		}
	}
	if len(seen) != 103 {
		t.Fatalf("events=%d", len(seen))
	}
}

func TestAbsentFrameCorrelationKeepsLegacyEnvelopeJSONShape(t *testing.T) {
	ag, err := New(Config{LLM: envelopeFinalModel{}})
	if err != nil {
		t.Fatal(err)
	}
	out := newEventOutput(2, true, "query", func() time.Time { return time.Unix(100, 0) })
	ag.emitEvent(out, TextEvent{Content: "fixture"}, eventCorrelation{attempt: 99})
	e := <-out.enveloped
	if e.FrameID != "" || e.InvokeAttempt != 0 {
		t.Fatal("orphan attempt correlation invented")
	}
	encoded, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "FrameID") || strings.Contains(string(encoded), "InvokeAttempt") {
		t.Fatal("unavailable correlation changed old envelope JSON fields")
	}
}

func TestFrameCorrelationRetainsOriginalTerminalBackpressureOwner(t *testing.T) {
	ag, err := New(Config{LLM: envelopeFinalModel{}, EventSendTimeout: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out := newEventOutput(1, true, "query", time.Now)
	unregister := ag.registerTurnCancellation(out, ctx)
	defer unregister()
	ag.emitEvent(out, TextEvent{Content: "evictable"}, eventCorrelation{frameID: "old", attempt: 1})
	if !ag.emitEvent(out, FinalResponseEvent{Content: "done"}, eventCorrelation{frameID: "final", attempt: 2}) {
		t.Fatal("terminal lost")
	}
	e := <-out.enveloped
	if e.FrameID != "final" || e.InvokeAttempt != 2 || e.Sequence != 2 || e.Kind != EventKindFinalResponse || ag.eventDropCount.Load() != 1 || ag.turnBackpressureState(out) == nil {
		t.Fatalf("terminal metadata or owner changed: %+v drops=%d", e, ag.eventDropCount.Load())
	}
}
