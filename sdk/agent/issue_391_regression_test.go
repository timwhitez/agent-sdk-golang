package agent

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestCancellationAtContinuationBoundaryDiscardsUnacceptedCalls(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var calls, toolRuns atomic.Int32
	partial := mustProviderStateContent(t, llm.TextContent("partial text to preserve"), []llm.ProviderState{{
		Provider: "fixture", Kind: "opaque", Data: json.RawMessage(`{"native_call":"shared-call"}`),
	}})
	model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
		calls.Add(1)
		return &llm.Completion{
			Content: partial, StopReason: "max_tokens",
			ToolCalls: []llm.ToolCall{{ID: "shared-call", Function: llm.FunctionCall{Name: "work", Arguments: `{"x":`}}},
		}, nil
	}}
	boundary := make(chan struct{})
	var canceled atomic.Bool
	ag, err := New(Config{
		LLM: model, MaxIterations: 3, EventBufferSize: 1, EventSendTimeout: time.Millisecond, EventDropLogEvery: 1,
		Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
			toolRuns.Add(1)
			return llm.TextContent("unexpected"), nil
		}}},
		Warningf: func(format string, args ...any) {
			if !strings.Contains(format, "dropping consistency-critical") || len(args) == 0 {
				return
			}
			if _, ok := args[0].(AutoContinueEvent); ok && canceled.CompareAndSwap(false, true) {
				cancel()
				close(boundary)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	prior := []llm.Message{
		{Role: llm.RoleAssistant, Content: partial, ToolCalls: []llm.ToolCall{{ID: "shared-call", Function: llm.FunctionCall{Name: "work", Arguments: `{}`}}}},
		llm.NewToolMessage("shared-call", "work", llm.TextContent("earlier accepted result"), false),
	}
	if err := ag.ReplaceHistoryChecked(prior); err != nil {
		t.Fatal(err)
	}
	events := ag.QueryStream(ctx, llm.TextContent("continue"))
	select {
	case <-boundary:
	case <-time.After(3 * time.Second):
		t.Fatal("did not reach the continuation publication boundary")
	}
	for range events {
	}
	if calls.Load() != 1 || toolRuns.Load() != 0 {
		t.Fatalf("provider calls=%d tool runs=%d; expected 1/0", calls.Load(), toolRuns.Load())
	}
	history := ag.Messages()
	if len(history) < len(prior) || !reflect.DeepEqual(history[:len(prior)], prior) {
		t.Fatal("earlier accepted block was changed")
	}
	visible, reminders := 0, 0
	for _, msg := range history[len(prior):] {
		if len(msg.ToolCalls) != 0 || msg.Role == llm.RoleTool || llm.HasProviderState(msg.Content) {
			t.Fatalf("canceled continuation remained in idle history: role=%s calls=%d name=%s", msg.Role, len(msg.ToolCalls), msg.Name)
		}
		if msg.Role == llm.RoleAssistant && msg.PlainText() == "partial text to preserve" {
			visible++
		}
		if msg.Name == messageorigin.Name(messageorigin.KindToolCallContinuation) {
			reminders++
		}
	}
	if visible != 1 || reminders != 1 {
		t.Fatalf("visible partials/reminders=%d/%d, want 1/1", visible, reminders)
	}
	if _, changed, _ := repairToolCallPairsDetailed(history); changed {
		t.Fatal("idle history requires outbound repair")
	}
}

func TestContinuationCancellationBeforeReminderDiscardsCurrentAssistant(t *testing.T) {
	for _, invalidMerge := range []bool{false, true} {
		name := "first_partial"
		if invalidMerge {
			name = "invalid_merge"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			partial := mustProviderStateContent(t, llm.TextContent("partial text to preserve"), []llm.ProviderState{{
				Provider: "fixture", Kind: "opaque", Data: json.RawMessage(`{"native_call":"shared-call"}`),
			}})
			var calls, toolRuns atomic.Int32
			model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
				call := calls.Add(1)
				stop, args := "max_tokens", `{"x":`
				if invalidMerge && call == 2 {
					stop, args = "tool_calls", `{`
				}
				return &llm.Completion{Content: partial, StopReason: stop,
					ToolCalls: []llm.ToolCall{{ID: "shared-call", Function: llm.FunctionCall{Name: "work", Arguments: args}}}}, nil
			}}
			wantCalls := 1
			if invalidMerge {
				wantCalls = 2
			}
			var canceled atomic.Bool
			ag, err := New(Config{
				LLM: model, MaxIterations: 3,
				Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
					toolRuns.Add(1)
					return llm.TextContent("unexpected"), nil
				}}},
				Compaction: &compaction.Config{
					Enabled: true, ContextWindow: 100000,
					TokenEstimator: func(text string) int {
						// Compaction preparation estimates the pending reminder
						// (or the earlier reminder of an invalid merge). Cancel
						// before checkAndCompactWithGrowth appends the current
						// reminder, exercising its early-error return.
						if text == messageorigin.ResponseTruncatedContinuationText && calls.Load() == int32(wantCalls) && canceled.CompareAndSwap(false, true) {
							cancel()
						}
						return 1
					},
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			prior := []llm.Message{
				{Role: llm.RoleAssistant, Content: partial, ToolCalls: []llm.ToolCall{{ID: "shared-call", Function: llm.FunctionCall{Name: "work", Arguments: `{}`}}}},
				llm.NewToolMessage("shared-call", "work", llm.TextContent("earlier accepted result"), false),
			}
			if err := ag.ReplaceHistoryChecked(prior); err != nil {
				t.Fatal(err)
			}
			for range ag.QueryStream(ctx, llm.TextContent("continue")) {
			}
			if !canceled.Load() || calls.Load() != int32(wantCalls) || toolRuns.Load() != 0 {
				t.Fatalf("canceled=%t provider calls=%d tool runs=%d, want true/%d/0", canceled.Load(), calls.Load(), toolRuns.Load(), wantCalls)
			}
			history := ag.Messages()
			if len(history) < len(prior) || !reflect.DeepEqual(history[:len(prior)], prior) {
				t.Fatal("earlier accepted block was changed")
			}
			visible, reminders := 0, 0
			for _, msg := range history[len(prior):] {
				if len(msg.ToolCalls) != 0 || msg.Role == llm.RoleTool || llm.HasProviderState(msg.Content) {
					t.Fatalf("unpublished continuation retained topology: role=%s calls=%d name=%s", msg.Role, len(msg.ToolCalls), msg.Name)
				}
				if msg.Role == llm.RoleAssistant && msg.PlainText() == "partial text to preserve" {
					visible++
				}
				if msg.Name == messageorigin.Name(messageorigin.KindToolCallContinuation) {
					reminders++
				}
			}
			if visible != wantCalls || reminders != wantCalls-1 {
				t.Fatalf("visible partials/reminders=%d/%d, want %d/%d", visible, reminders, wantCalls, wantCalls-1)
			}
			if _, changed, _ := repairToolCallPairsDetailed(history); changed {
				t.Fatal("idle history requires outbound repair")
			}
		})
	}
}
