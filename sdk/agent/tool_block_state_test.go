package agent

import (
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestToolBlockStateTracksExecutedAndUnstartedClosure(t *testing.T) {
	block := newToolBlockState([]llm.ToolCall{{ID: "run"}, {ID: "skip"}})
	block.markRunning(0)
	block.markTerminal(0, toolCallRunning, "handler_return")
	block.markTerminal(1, toolCallAccepted, "turn_end")
	if err := block.validateClosed(); err != nil {
		t.Fatal(err)
	}
}

func TestToolBlockStateReportsInvariantViolations(t *testing.T) {
	tests := []struct {
		name string
		run  func(*toolBlockState)
		want string
	}{
		{name: "missing terminal", run: func(*toolBlockState) {}, want: "remains phase=\"accepted\""},
		{name: "duplicate terminal", run: func(block *toolBlockState) {
			block.markTerminal(0, toolCallAccepted, "first")
			block.markTerminal(0, toolCallTerminal, "second")
		}, want: "2 terminal transitions"},
		{name: "wrong phase", run: func(block *toolBlockState) {
			block.markTerminal(0, toolCallRunning, "handler_return")
		}, want: "want \"running\""},
		{name: "out of range", run: func(block *toolBlockState) {
			block.markRunning(1)
			block.markTerminal(0, toolCallAccepted, "turn_end")
		}, want: "outside block length"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			block := newToolBlockState([]llm.ToolCall{{ID: "call-1"}})
			test.run(block)
			err := block.validateClosed()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error=%v want substring %q", err, test.want)
			}
		})
	}
}

func TestToolBlockStateRejectsAmbiguousAcceptedIDs(t *testing.T) {
	block := newToolBlockState([]llm.ToolCall{{ID: ""}, {ID: "same"}, {ID: "same"}})
	block.markTerminalRange(0, toolCallAccepted, "closed")
	err := block.validateClosed()
	if err == nil || !strings.Contains(err.Error(), "empty id") || !strings.Contains(err.Error(), "duplicates the id") || strings.Contains(err.Error(), "same") {
		t.Fatalf("error=%v", err)
	}
}

func failOnToolBlockShadowWarning(t *testing.T) func(string, ...any) {
	t.Helper()
	return func(format string, args ...any) {
		message := fmt.Sprintf(format, args...)
		if strings.Contains(message, "tool block shadow invariant mismatch") {
			t.Errorf("unexpected shadow warning: %s", message)
		}
	}
}
