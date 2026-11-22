package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestUnacceptedToolCompletionPreservesVisibleContentAndUsage(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, text := range []string{"", "visible partial"} {
			name := "buffered"
			if streaming {
				name = "streaming"
			}
			t.Run(name+"/"+text, func(t *testing.T) {
				state := []llm.ProviderState{{Provider: "fixture", Kind: "opaque", Data: json.RawMessage(`{"private":"provider-owned"}`)}}
				content, err := llm.WithProviderState(llm.TextContent(text), state)
				if err != nil {
					t.Fatal(err)
				}
				completion := &llm.Completion{
					Content: content, ResponseID: "partial-response", Usage: llm.NewProviderUsage(10, 2, 12),
					ToolCalls: []llm.ToolCall{{ID: "unaccepted", Function: llm.FunctionCall{Name: "work", Arguments: `{}`}}},
				}
				var model llm.ChatModel = &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
					return completion, errors.New("fixture failure")
				}}
				if streaming {
					model = &terminalContractModel{events: []llm.StreamEvent{
						llm.StreamTextDeltaEvent{Delta: text},
						llm.StreamToolCallDeltaEvent{Index: 0, ID: "unaccepted", NameDelta: "work", ArgumentsDelta: `{}`},
						llm.StreamProviderStateEvent{State: state},
						llm.StreamResponseEvent{ResponseID: "partial-response"},
						llm.StreamUsageEvent{Usage: *completion.Usage},
						llm.StreamErrorEvent{Err: errors.New("fixture failure")},
					}}
				}
				runs := 0
				a, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 1, Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
					runs++
					return llm.TextContent("unexpected"), nil
				}}}})
				if err != nil {
					t.Fatal(err)
				}
				usage, accounting, failures := 0, 0, 0
				for event := range a.QueryStream(t.Context(), llm.TextContent("run")) {
					switch event := event.(type) {
					case UsageEvent:
						usage++
						if event.ResponseID != "partial-response" || event.Usage.PromptTokens != 10 || event.Usage.CompletionTokens != 2 {
							t.Errorf("partial usage changed: %+v", event)
						}
					case AccountingEvent:
						accounting++
					case ErrorEvent:
						failures++
					case ToolCallEvent, ToolResultEvent, StepStartEvent, StepCompleteEvent:
						t.Errorf("unaccepted completion published %T", event)
					}
				}
				if runs != 0 || usage != 1 || accounting != 1 || failures != 1 {
					t.Fatalf("runs/usage/accounting/errors=%d/%d/%d/%d", runs, usage, accounting, failures)
				}
				assistants := 0
				for _, message := range a.Messages() {
					if llm.HasProviderState(message.Content) || len(message.ToolCalls) > 0 || message.Role == llm.RoleTool {
						t.Fatal("unaccepted call state entered history")
					}
					if message.Role == llm.RoleAssistant {
						assistants++
						if message.PlainText() != text {
							t.Fatal("visible content changed")
						}
					}
				}
				wantAssistants := 0
				if text != "" {
					wantAssistants = 1
				}
				if assistants != wantAssistants || !llm.HasProviderState(completion.Content) || len(completion.ToolCalls) != 1 {
					t.Fatalf("empty record appended or provider-owned completion mutated: assistants=%d", assistants)
				}
			})
		}
	}
}
