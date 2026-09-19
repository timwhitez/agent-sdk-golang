package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// The call event describes first-pass normalization. Typed repair happens later,
// inside Tool.Execute's adapter, so that event is not final execution authority.
func TestToolArgumentAdmissionViews(t *testing.T) {
	type args struct {
		Value string `json:"value"`
	}
	for _, tt := range []struct {
		name, raw, normalized, value, repair, failure string
		handlerCalls, businessCalls                   int
	}{
		{name: "valid bytes", raw: `{ "value" : "ok" }`, normalized: `{ "value" : "ok" }`, value: "ok", handlerCalls: 1, businessCalls: 1},
		{name: "string normalization", raw: `"ok"`, normalized: `{"value":"ok"}`, value: "ok", repair: "string_wrapped", handlerCalls: 1, businessCalls: 1},
		{name: "typed key repair", raw: `{"Value":"ok","extra":true}`, normalized: `{"Value":"ok","extra":true}`, value: "ok", repair: "schema_key", handlerCalls: 1, businessCalls: 1},
		{name: "invalid syntax", raw: `{"value":`, normalized: `{}`, repair: "decode_error", failure: "Invalid tool arguments"},
		{name: "invalid type", raw: `{"value":42}`, normalized: `{"value":42}`, handlerCalls: 1, failure: "Invalid tool arguments"},
		{name: "effect then unknown field error", raw: `{"value":"ok","extra":true}`, normalized: `{"value":"ok","extra":true}`, value: "ok", repair: "schema_key", handlerCalls: 1, businessCalls: 1, failure: "effect unknown field"},
		{name: "effect then panic", raw: `{"value":"ok"}`, normalized: `{"value":"ok"}`, value: "ok", handlerCalls: 1, businessCalls: 1, failure: "effect panic"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			handlers, business := 0, 0
			var handlerRaw, businessValue string
			tool := tools.Func[args]("admission", "fixture", func(_ context.Context, a args, _ *tools.Container) (any, error) {
				business++
				businessValue = a.Value
				switch tt.failure {
				case "effect unknown field":
					return nil, errors.New(tt.failure)
				case "effect panic":
					panic(tt.failure)
				}
				return "ok", nil
			})
			original := tool.Handler
			tool.Handler = func(ctx context.Context, raw json.RawMessage, deps *tools.Container) (llm.Content, error) {
				handlers++
				handlerRaw = string(raw)
				return original(ctx, raw, deps)
			}
			call := llm.ToolCall{ID: "args-call", Type: "function", Function: llm.FunctionCall{Name: "admission", Arguments: tt.raw}}
			model := &toolPlanScriptModel{toolCalls: []llm.ToolCall{call}}
			ag, err := New(Config{LLM: model, Tools: []tools.Tool{tool}, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			var states []toolCallState
			ag.toolBlockStateObserved = func(block *toolBlockState) { states = append(states, block.calls...) }
			events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
			if handlers != tt.handlerCalls || business != tt.businessCalls || businessValue != tt.value {
				t.Fatalf("handler/business/value=%d/%d/%q want %d/%d/%q", handlers, business, businessValue, tt.handlerCalls, tt.businessCalls, tt.value)
			}
			if handlers > 0 && handlerRaw != tt.normalized {
				t.Fatalf("handler raw=%q want %q", handlerRaw, tt.normalized)
			}
			calls, results := 0, 0
			for _, event := range events {
				switch event := event.(type) {
				case ToolCallEvent:
					calls++
					if string(event.ArgsJSON) != tt.normalized {
						t.Fatalf("call args=%s want %s", event.ArgsJSON, tt.normalized)
					}
					if tt.repair == "schema_key" && event.ArgsMeta["args_repaired"] == true {
						t.Fatal("first-pass event incorrectly claims typed repair")
					}
				case ToolResultEvent:
					results++
					if event.IsError != (tt.failure != "") || !strings.Contains(event.Result, tt.failure) {
						t.Fatalf("result=%+v", event)
					}
					repairKind, _ := event.Metadata["args_repair_kind"].(string)
					if tt.repair != "" && (event.Metadata["args_repaired"] != true || !strings.Contains(repairKind, tt.repair) || event.Metadata["args_raw"] != tt.raw) {
						t.Fatalf("repair metadata=%v want %s with original spelling", event.Metadata, tt.repair)
					}
				}
			}
			historyResults := 0
			for _, message := range ag.Messages() {
				if len(message.ToolCalls) > 0 && message.ToolCalls[0].Function.Arguments != tt.raw {
					t.Fatal("history changed provider arguments")
				}
				if message.Role == llm.RoleTool {
					historyResults++
					if message.IsError != (tt.failure != "") {
						t.Fatalf("history error=%t", message.IsError)
					}
				}
			}
			if calls != 1 || results != 1 || historyResults != 1 || model.calls != 2 {
				t.Fatalf("calls/results/history/provider=%d/%d/%d/%d", calls, results, historyResults, model.calls)
			}
			// The driver observes Execute returning even when normalization or
			// typed decode rejects before the business handler runs.
			if len(states) != 1 || states[0].executionKnowledge != toolExecutionOutcomeObserved || states[0].terminalCount != 1 {
				t.Fatalf("terminal states=%+v", states)
			}
			assertContiguousToolResults(t, ag.Messages())
		})
	}
}
