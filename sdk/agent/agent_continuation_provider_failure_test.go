package agent

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestProviderFailureDiscardsUnacceptedContinuation(t *testing.T) {
	for _, scenario := range []string{"provider_error", "provider_canceled_live_root", "root_canceled"} {
		t.Run(scenario, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			partial, err := llm.WithProviderState(llm.TextContent("visible partial text"), []llm.ProviderState{{Provider: "fixture", Kind: "partial", Data: json.RawMessage(`{"private":"opaque marker"}`)}})
			if err != nil {
				t.Fatal(err)
			}
			calls, handled := 0, 0
			model := &frameScriptModel{invoke: func(request llm.InvokeRequest) (*llm.Completion, error) {
				calls++
				switch calls {
				case 1:
					return &llm.Completion{Content: partial, StopReason: "max_tokens", ToolCalls: []llm.ToolCall{{ID: "call_0", Function: llm.FunctionCall{Name: "work", Arguments: `{"value":`}}}}, nil
				case 2:
					failure := errors.New("fixture provider failure")
					if scenario == "provider_canceled_live_root" {
						failure = context.Canceled
					}
					if scenario == "root_canceled" {
						cancel()
						failure = context.Canceled
					}
					return &llm.Completion{Content: llm.TextContent("failed response text"), ResponseID: "failed-response", Usage: &llm.Usage{PromptTokens: 17, CompletionTokens: 3, TotalTokens: 20}}, failure
				default:
					return &llm.Completion{Content: llm.TextContent("next turn")}, nil
				}
			}}
			a, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 1, Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				handled++
				return llm.TextContent("must not execute"), nil
			}}}})
			if err != nil {
				t.Fatal(err)
			}
			// An earlier completed block with the same ID must survive.
			prior := []llm.Message{
				{Role: llm.RoleAssistant, Content: partial, ToolCalls: []llm.ToolCall{{ID: "call_0", Function: llm.FunctionCall{Name: "work", Arguments: `{}`}}}},
				{Role: llm.RoleTool, ToolCallID: "call_0", ToolName: "work", Content: llm.TextContent("prior result")},
			}
			a.ReplaceHistory(prior)
			errorsSeen, usageSeen, accountingSeen := 0, 0, 0
			for envelope := range a.QueryStreamEnveloped(ctx, llm.TextContent("run")) {
				switch event := envelope.Event.(type) {
				case ErrorEvent:
					errorsSeen++
					wantOrigin := EventOriginProvider
					if scenario == "root_canceled" {
						wantOrigin = EventOriginSDKDriver
					}
					if envelope.Origin != wantOrigin {
						t.Errorf("error origin=%s want %s", envelope.Origin, wantOrigin)
					}
				case ToolCallEvent, ToolResultEvent, StepStartEvent, StepCompleteEvent:
					t.Errorf("unaccepted call emitted %T", event)
				case UsageEvent:
					usageSeen++
					if event.ResponseID != "failed-response" || event.Usage.PromptTokens != 17 || event.Usage.CompletionTokens != 3 {
						t.Error("partial usage changed")
					}
				case AccountingEvent:
					accountingSeen++
					if event.ToolCallID != "" || event.ResponseID != "failed-response" {
						t.Error("invented tool accounting")
					}
				}
			}
			if calls != 2 || handled != 0 || errorsSeen != 1 {
				t.Fatalf("calls/handled/errors=%d/%d/%d", calls, handled, errorsSeen)
			}
			if scenario != "root_canceled" && (usageSeen != 1 || accountingSeen != 1) {
				t.Fatalf("usage/accounting=%d/%d", usageSeen, accountingSeen)
			}
			history := a.Messages()
			if len(history) < 2 || !reflect.DeepEqual(history[:2], prior) {
				t.Fatal("cleanup changed an earlier accepted block with reused IDs")
			}
			visible, failedText := false, false
			for _, message := range history[2:] {
				if len(message.ToolCalls) != 0 || message.Role == llm.RoleTool {
					t.Fatal("failed unaccepted continuation retained tool topology")
				}
				if strings.Contains(message.Content.PlainText(), "visible partial text") {
					visible = true
					if llm.HasProviderState(message.Content) {
						t.Fatal("abandoned partial retained opaque provider state")
					}
				}
				if strings.Contains(message.Content.PlainText(), "failed response text") {
					failedText = true
				}
			}
			if !visible || !failedText {
				t.Fatal("cleanup lost visible text")
			}
			if _, changed, _ := repairToolCallPairsDetailed(history); changed {
				t.Fatal("history requires outbound repair")
			}
			for event := range a.QueryStream(context.Background(), llm.TextContent("next")) {
				if warning, ok := event.(WarnEvent); ok && warning.Kind == "tool_pairing_repaired" {
					t.Error("next turn required repair")
				}
			}
			if calls != 3 {
				t.Fatalf("provider calls=%d", calls)
			}
		})
	}
}
