package agent

import (
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestToolCallAccumulatorPreservesWhitespace(t *testing.T) {
	acc := &toolCallAccumulator{}
	acc.apply(llm.StreamToolCallDeltaEvent{Index: 0, NameDelta: "write", ArgumentsDelta: `{"content":"hello`})
	acc.apply(llm.StreamToolCallDeltaEvent{Index: 0, ArgumentsDelta: " "})
	acc.apply(llm.StreamToolCallDeltaEvent{Index: 0, ArgumentsDelta: `world"}`})

	calls := acc.finalize()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Arguments != `{"content":"hello world"}` {
		t.Fatalf("expected whitespace preserved, got %q", calls[0].Function.Arguments)
	}
}
