package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type onceEffectModel struct{ calls int }

func (*onceEffectModel) Provider() string { return "fixture" }
func (*onceEffectModel) Model() string    { return "fixture" }
func (m *onceEffectModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls == 1 {
		return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "effect", Function: llm.FunctionCall{Name: "effect", Arguments: `{"value":"ok","extra":true}`}}}}, nil
	}
	return &llm.Completion{Content: llm.TextContent("done")}, nil
}
func TestHandlerUnknownFieldErrorClosesOnceAfterEffect(t *testing.T) {
	effects := 0
	tool := tools.Tool{Name: "effect", Schema: map[string]any{"type": "object", "additionalProperties": false, "properties": map[string]any{"value": map[string]any{"type": "string"}}}, Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		effects++
		return llm.TextContent("effect happened"), errors.New("remote unknown field")
	}}
	a, err := New(Config{LLM: &onceEffectModel{}, Tools: []tools.Tool{tool}, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	results := 0
	var states []toolCallState
	a.toolBlockStateObserved = func(block *toolBlockState) { states = append(states, block.calls...) }
	for ev := range a.QueryStream(context.Background(), llm.TextContent("run")) {
		if result, ok := ev.(ToolResultEvent); ok {
			results++
			if !result.IsError {
				t.Fatal("execution error lost")
			}
		}
	}
	historyResults := 0
	for _, m := range a.Messages() {
		if m.Role == llm.RoleTool {
			historyResults++
			if !m.IsError || m.Content.PlainText() != "effect happened" {
				t.Fatal("history error lost")
			}
		}
	}
	if effects != 1 || results != 1 || historyResults != 1 {
		t.Fatal("effect or closure duplicated", effects, results, historyResults)
	}
	if len(states) != 1 || states[0].executionKnowledge != toolExecutionOutcomeObserved || states[0].terminalCount != 1 {
		t.Fatal("wrong execution knowledge", states)
	}
}
