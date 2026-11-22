package agent

import (
	"context"
	"encoding/json"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"testing"
)

func issue305Call(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
}
func TestD1InvalidFallbackPreservesRecordID(t *testing.T) {
	var recordID json.Number
	invalid := tools.Tool{Name: "invalid", Hidden: true, Handler: func(_ context.Context, raw json.RawMessage, _ *tools.Container) (llm.Content, error) {
		var in struct {
			RecordID json.Number `json:"record_id"`
		}
		if err := json.Unmarshal(raw, &in); err != nil {
			return llm.Content{}, err
		}
		recordID = in.RecordID
		return llm.TextContent("unknown tool"), nil
	}}
	model := &turnModel{turns: []*llm.Completion{{ToolCalls: []llm.ToolCall{issue305Call("one", "lookup_by_id", `{"record_id":9007199254740993}`)}}}}
	a, err := New(Config{LLM: model, Tools: []tools.Tool{invalid}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Query(t.Context(), "look it up"); err != nil {
		t.Fatal(err)
	}
	if recordID.String() != "9007199254740993" {
		t.Fatalf("invalid fallback received record_id=%s, want 9007199254740993", recordID)
	}
}
