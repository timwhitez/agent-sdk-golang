package agent

import (
	"context"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// The receipt is published only when the stream closes and then matches the
// received range exactly for a drained, drop-free Query.
func TestQueryStreamReceiptPublishedAtClose(t *testing.T) {
	model := &singleActiveTurnModel{firstStart: make(chan struct{}), firstFinish: make(chan struct{})}
	agent, err := New(Config{LLM: model})
	if err != nil {
		t.Fatal(err)
	}
	events, receipt := agent.QueryStreamEnvelopedWithReceipt(context.Background(), llm.TextContent("run"), nil)
	select {
	case <-model.firstStart:
	case <-time.After(5 * time.Second):
		t.Fatal("provider invocation did not start")
	}
	if _, ok := receipt.Summary(); ok {
		t.Fatal("receipt published while the stream is open")
	}
	close(model.firstFinish)
	envelopes := collectEnvelopes(events)
	summary, ok := receipt.Summary()
	if !ok {
		t.Fatal("receipt missing after close")
	}
	last := envelopes[len(envelopes)-1]
	if last.Kind != EventKindFinalResponse {
		t.Fatalf("last envelope=%#v", last)
	}
	if summary.LastSequence != last.Sequence || summary.LastSequence != uint64(len(envelopes)) || summary.DroppedEvents != 0 || summary.DroppedCriticalEvents != 0 || summary.QueryID != last.QueryID || summary.QueryID == "" {
		t.Fatalf("summary=%+v received=%d last=%d query=%q", summary, len(envelopes), last.Sequence, last.QueryID)
	}
}

// A terminal rejected after an earlier terminal and a later undelivered
// event leave no visible gap: the consumer sees Sequence 1 and a closed
// channel. Only the receipt shows that two more envelopes were allocated.
func TestQueryStreamReceiptCountsDropsAfterTerminal(t *testing.T) {
	agent := &Agent{eventSendTimeout: time.Millisecond}
	out := newEventOutput(1, true, "query-test", time.Now)
	if !agent.emitEvent(out, ErrorEvent{Provider: "fixture", Kind: "provider", Message: "failed"}) {
		t.Fatal("terminal error not buffered")
	}
	if agent.emitEvent(out, FinalResponseEvent{Content: "lower priority"}) {
		t.Fatal("lower-priority final was delivered")
	}
	if agent.emitEvent(out, TextDeltaEvent{Delta: "late"}) {
		t.Fatal("late delta was delivered into a full channel")
	}
	out.close()
	received := collectEnvelopes(out.enveloped)
	if len(received) != 1 || received[0].Sequence != 1 {
		t.Fatalf("received=%#v", received)
	}
	summary, ok := out.receipt.Summary()
	if !ok || summary.LastSequence != 3 || summary.DroppedEvents != 2 || summary.DroppedCriticalEvents != 1 {
		t.Fatalf("summary=%+v ok=%v", summary, ok)
	}
	if uint64(len(received))+summary.DroppedEvents != summary.LastSequence {
		t.Fatalf("allocated range not accounted: received=%d summary=%+v", len(received), summary)
	}
}

// Drop counts belong to the stream, not the Agent: another stream's drops
// on the same Agent never appear in a clean stream's receipt.
func TestQueryStreamReceiptDropsArePerStream(t *testing.T) {
	agent := &Agent{eventSendTimeout: time.Millisecond}
	noisy := newEventOutput(1, true, "query-noisy", time.Now)
	agent.emitEvent(noisy, TextDeltaEvent{Delta: "a"})
	agent.emitEvent(noisy, TextDeltaEvent{Delta: "b"})
	if agent.eventDropCount.Load() != 1 {
		t.Fatalf("agent drop count=%d", agent.eventDropCount.Load())
	}
	clean := newEventOutput(4, true, "query-clean", time.Now)
	agent.emitEvent(clean, FinalResponseEvent{Content: "ok"})
	clean.close()
	collectEnvelopes(clean.enveloped)
	summary, ok := clean.receipt.Summary()
	if !ok || summary.LastSequence != 1 || summary.DroppedEvents != 0 || summary.DroppedCriticalEvents != 0 || summary.QueryID != "query-clean" {
		t.Fatalf("clean summary=%+v ok=%v", summary, ok)
	}
	noisy.close()
	collectEnvelopes(noisy.enveloped)
	if summary, _ := noisy.receipt.Summary(); summary.LastSequence != 2 || summary.DroppedEvents != 1 {
		t.Fatalf("noisy summary=%+v", summary)
	}
}

// The synchronous busy rejection also publishes its one-envelope range.
func TestQueryStreamReceiptBusyAdmission(t *testing.T) {
	model := &singleActiveTurnModel{firstStart: make(chan struct{}), firstFinish: make(chan struct{})}
	agent, err := New(Config{LLM: model})
	if err != nil {
		t.Fatal(err)
	}
	first, _ := agent.QueryStreamEnvelopedWithReceipt(context.Background(), llm.TextContent("first"), nil)
	select {
	case <-model.firstStart:
	case <-time.After(5 * time.Second):
		t.Fatal("provider invocation did not start")
	}
	busy, receipt := agent.QueryStreamEnvelopedWithReceipt(context.Background(), llm.TextContent("second"), nil)
	envelopes := collectEnvelopes(busy)
	summary, ok := receipt.Summary()
	if len(envelopes) != 1 || !ok || summary.LastSequence != 1 || summary.DroppedEvents != 0 || summary.QueryID != envelopes[0].QueryID {
		t.Fatalf("busy envelopes=%#v summary=%+v ok=%v", envelopes, summary, ok)
	}
	close(model.firstFinish)
	collectEnvelopes(first)
}
