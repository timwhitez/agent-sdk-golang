package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestAcceptedToolCorrelationAndHistoryOnlyTails(t *testing.T) {
	for _, mode := range []string{"normal", "done", "cancel", "before_start_cancel", "invalid", "suppressed"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			providers, handled := 0, 0
			model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
				providers++
				if providers > 1 {
					return &llm.Completion{Content: llm.TextContent("done")}, nil
				}
				calls := make([]llm.ToolCall, 3)
				for i := range calls {
					id := fmt.Sprintf("call-%d", i+1)
					if mode == "invalid" {
						id = "duplicate"
					}
					calls[i] = llm.ToolCall{ID: id, Function: llm.FunctionCall{Name: "work", Arguments: "{}"}}
				}
				return &llm.Completion{ToolCalls: calls}, nil
			}}
			cfg := Config{LLM: model, QueryIDGenerator: func() string { return "tool-query" }, Warningf: func(string, ...any) {}, Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				handled++
				if mode == "cancel" {
					cancel()
				}
				if mode == "done" && handled == 2 {
					return llm.Content{}, &tools.TaskCompleteError{Message: "finished"}
				}
				return llm.TextContent("result"), nil
			}}}}
			if mode == "suppressed" {
				cfg.RepeatToolSignatureThreshold = 3
			}
			ag, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "before_start_cancel" {
				ag.toolBlockTestHook = func(*toolBlockState) { cancel() }
			}
			counts := map[uint64]map[EventKind]int{}
			var sequence uint64
			for envelope := range ag.QueryStreamEnveloped(ctx, llm.TextContent("run")) {
				sequence++
				if envelope.Sequence != sequence {
					t.Fatal("sequence changed")
				}
				known := false
				switch e := envelope.Event.(type) {
				case StepStartEvent, StepCompleteEvent, ToolCallEvent, ToolResultEvent:
					known = true
				case AccountingEvent:
					known = e.CorrelationKind == "tool_call"
				}
				if !known {
					if envelope.ToolBlockID != "" || envelope.ToolCallOrdinal != 0 || envelope.ToolBlockCallCount != 0 {
						t.Fatalf("ambient tool identity: %+v", envelope)
					}
					continue
				}
				ordinal := envelope.ToolCallOrdinal
				if envelope.ToolBlockID != "tool-query/frame/1/tool-block" || envelope.ToolBlockCallCount != 3 || ordinal < 1 || ordinal > 3 {
					t.Fatalf("wrong tool identity: %+v", envelope)
				}
				if counts[ordinal] == nil {
					counts[ordinal] = map[EventKind]int{}
				}
				counts[ordinal][envelope.Kind]++
			}
			wantEvents, wantHandled := 3, 3
			switch mode {
			case "done":
				wantEvents, wantHandled = 2, 2
			case "cancel":
				wantEvents, wantHandled = 1, 1
			case "before_start_cancel", "invalid":
				wantEvents, wantHandled = 0, 0
			case "suppressed":
				wantHandled = 2
			}
			if len(counts) != wantEvents || handled != wantHandled {
				t.Fatalf("counts=%v handled=%d", counts, handled)
			}
			for ordinal := uint64(1); ordinal <= uint64(wantEvents); ordinal++ {
				for _, kind := range []EventKind{EventKindStepStart, EventKindToolCall, EventKindToolResult, EventKindAccounting, EventKindStepComplete} {
					if counts[ordinal][kind] != 1 {
						t.Fatalf("ordinal=%d kind=%s counts=%v", ordinal, kind, counts)
					}
				}
			}
			historyResults := 0
			for _, message := range ag.Messages() {
				if message.Role == llm.RoleTool {
					historyResults++
				}
			}
			wantHistory := 3
			if mode == "invalid" {
				wantHistory = 0
			}
			if historyResults != wantHistory {
				t.Fatalf("history results=%d want=%d", historyResults, wantHistory)
			}
		})
	}
}

func TestUnknownToolEnvelopeFieldsAreOmitted(t *testing.T) {
	data, err := json.Marshal(EventEnvelope{Event: TextEvent{Content: "unchanged"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"ToolBlockID", "ToolCallOrdinal", "ToolBlockCallCount"} {
		if strings.Contains(string(data), key) {
			t.Fatalf("unknown field %s was serialized", key)
		}
	}
}
