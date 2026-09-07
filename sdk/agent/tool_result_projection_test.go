package agent

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestToolResultProjectionViewsAndDeliveryGate(t *testing.T) {
	content, err := llm.WithProviderState(llm.TextContent("visible"), []llm.ProviderState{{Provider: "fixture", Kind: "opaque", Data: json.RawMessage(`{"private":"state"}`)}})
	if err != nil {
		t.Fatal(err)
	}
	message := llm.Message{Role: llm.RoleTool, ToolCallID: "call", ToolName: "work", Content: content, IsError: true, Ephemeral: true}
	metadata := map[string]any{"result_truncated": true}
	result := projectToolResult(message, metadata, "original-long-result")
	if !reflect.DeepEqual(result.history, message) || result.original != "original-long-result" {
		t.Fatal("projection changed history or original result")
	}
	want := ToolResultEvent{Tool: "work", ToolCallID: "call", Result: "visible", IsError: true, Metadata: metadata}
	if !reflect.DeepEqual(result.event(), want) {
		t.Fatal("canonical identity/flags/visible metadata lost")
	}
	// Explicit compatibility view override does not rewrite history or identity.
	result.visible = "short event view"
	want.Result = "short event view"
	if !reflect.DeepEqual(result.event(), want) || !reflect.DeepEqual(result.history, message) {
		t.Fatal("event view rewrote history")
	}

	ag, err := New(Config{LLM: historyCloneModel{}, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	out := newEventOutput(2, false, "", nil)
	ag.emitToolResultWithAccounting(out, result, time.Millisecond)
	event := (<-out.legacy).(ToolResultEvent)
	accounting := (<-out.legacy).(AccountingEvent)
	if !reflect.DeepEqual(event, want) || accounting.ToolCallID != "call" || accounting.Payload.Status != "error" || accounting.DurationMS != 1 {
		t.Fatal("delivery projections diverged")
	}
	if *accounting.Payload.Measurements.OriginalBytes != int64(len(result.original)) || *accounting.Payload.Measurements.VisibleBytes != int64(len(result.visible)) {
		t.Fatal("measurement views conflated")
	}
	if len(ag.Messages()) != 0 {
		t.Fatal("publishing unexpectedly committed history")
	}

	// The existing publisher is delivery-gated: absent/abandoned output must
	// not allocate another accounting sequence or invent an accounting event.
	ag.emitToolResultWithAccounting(nil, result, 0)
	blocked := newEventOutput(1, false, "", nil)
	blocked.legacy <- WarnEvent{Message: "filler"}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	defer ag.registerTurnCancellation(blocked, ctx)()
	ag.eventSendTimeout = time.Nanosecond
	ag.emitToolResultWithAccounting(blocked, result, 0)
	if ag.accountingSequence.Load() != 1 || ag.eventDropCount.Load() != 1 {
		t.Fatal("undelivered result emitted accounting or lost drop evidence")
	}
	if len(blocked.legacy) != 1 {
		t.Fatal("abandoned output changed")
	}
}
