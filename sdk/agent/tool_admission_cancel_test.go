package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestToolCallDropRootCancellationPreventsHandlerAdmission(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var effects, providers atomic.Int32
	var once atomic.Bool
	dropped := make(chan struct{})
	states := make(chan []toolCallState, 1)
	model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
		if providers.Add(1) > 1 {
			return &llm.Completion{Content: llm.TextContent("next turn")}, nil
		}
		// No text/usage event before StepStart: it occupies the one-slot buffer.
		return &llm.Completion{ToolCalls: []llm.ToolCall{cancelBoundaryCall("first", "effect"), cancelBoundaryCall("tail", "effect")}}, nil
	}}
	ag, err := New(Config{LLM: model, EventBufferSize: 1, EventSendTimeout: time.Millisecond, EventDropLogEvery: 1, Warningf: func(format string, args ...any) {
		message := fmt.Sprintf(format, args...)
		if strings.Contains(message, "dropping consistency-critical agent event agent.ToolCallEvent") && once.CompareAndSwap(false, true) {
			cancel()
			close(dropped)
		}
	}, Tools: []tools.Tool{{Name: "effect", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		effects.Add(1)
		return llm.TextContent("effect occurred"), nil
	}}}})
	if err != nil {
		t.Fatal(err)
	}
	// Read-only final-state observation. Cancellation uses only public Warningf.
	ag.toolBlockStateObserved = func(b *toolBlockState) { states <- append([]toolCallState(nil), b.calls...) }
	stream := ag.QueryStreamEnveloped(ctx, llm.TextContent("run"))
	select {
	case <-dropped:
	case <-ctx.Done():
		if !once.Load() {
			t.Fatal("ToolCall drop did not trigger cancellation")
		}
	}
	resultEvents := 0
	for envelope := range stream {
		if _, ok := envelope.Event.(ToolResultEvent); ok {
			resultEvents++
		}
	}
	if effects.Load() != 0 || providers.Load() != 1 || resultEvents != 0 {
		t.Fatalf("effects=%d providers=%d result events=%d", effects.Load(), providers.Load(), resultEvents)
	}
	snapshot := <-states
	if len(snapshot) != 2 {
		t.Fatalf("accepted calls=%d", len(snapshot))
	}
	for _, call := range snapshot {
		if call.executionKnowledge != toolExecutionNotStarted || call.terminalCount != 1 || call.closure != "root_cancel_before_start" {
			t.Fatalf("wrong canceled closure: %+v", call)
		}
	}
	assertContiguousToolResults(t, ag.Messages())
	for _, id := range []string{"first", "tail"} {
		assertCanceledToolResult(t, ag.Messages(), id)
	}
	historyResults := 0
	for _, message := range ag.Messages() {
		if message.Role == llm.RoleTool {
			historyResults++
		}
	}
	if historyResults != 2 {
		t.Fatalf("history results=%d", historyResults)
	}
	// Closed history must be reusable without a repair or a handler replay.
	for event := range ag.QueryStream(context.Background(), llm.TextContent("next")) {
		if warning, ok := event.(WarnEvent); ok && warning.Kind == "tool_pairing_repaired" {
			t.Fatal("next turn needed history repair")
		}
	}
	if effects.Load() != 0 || providers.Load() != 2 {
		t.Fatalf("next turn effects=%d providers=%d", effects.Load(), providers.Load())
	}
}
