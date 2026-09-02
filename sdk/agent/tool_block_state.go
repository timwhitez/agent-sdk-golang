package agent

import (
	"fmt"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type toolCallPhase string

type toolExecutionKnowledge uint8

const (
	toolCallAccepted toolCallPhase = "accepted"
	toolCallRunning  toolCallPhase = "running"
	toolCallTerminal toolCallPhase = "terminal"

	toolExecutionUnknown toolExecutionKnowledge = iota
	toolExecutionNotStarted
	toolExecutionAttemptStarted
	toolExecutionOutcomeObserved
	toolExecutionIndeterminate
)

type toolCallState struct {
	phase              toolCallPhase
	executionKnowledge toolExecutionKnowledge
	terminalCount      int
	closure            string
}

func (k toolExecutionKnowledge) String() string {
	switch k {
	case toolExecutionNotStarted:
		return "not_started"
	case toolExecutionAttemptStarted:
		return "attempt_started"
	case toolExecutionOutcomeObserved:
		return "outcome_observed"
	case toolExecutionIndeterminate:
		return "indeterminate"
	default:
		return "unknown"
	}
}

// toolBlockState is an observe-only mirror of the sequential Tool Loop. It is
// deliberately query-local and does not construct history, events, or results.
type toolBlockState struct {
	calls      []toolCallState
	violations []string
}

func newToolBlockState(calls []llm.ToolCall) *toolBlockState {
	block := &toolBlockState{calls: make([]toolCallState, len(calls))}
	seen := make(map[string]int, len(calls))
	for i, call := range calls {
		id := strings.TrimSpace(call.ID)
		block.calls[i] = toolCallState{phase: toolCallAccepted, executionKnowledge: toolExecutionNotStarted}
		if id == "" {
			block.addViolation("call[%d] has empty id", i)
			continue
		}
		if first, ok := seen[id]; ok {
			block.addViolation("call[%d] duplicates the id from call[%d]", i, first)
			continue
		}
		seen[id] = i
	}
	return block
}

func (b *toolBlockState) markRunning(index int) {
	call, ok := b.call(index)
	if !ok {
		return
	}
	if call.phase != toolCallAccepted {
		b.addViolation("call[%d] cannot start from phase %q", index, call.phase)
		return
	}
	if call.executionKnowledge != toolExecutionNotStarted {
		b.addViolation("call[%d] cannot start from execution %q", index, call.executionKnowledge)
		return
	}
	call.phase = toolCallRunning
	call.executionKnowledge = toolExecutionAttemptStarted
}

func (b *toolBlockState) markAttemptReturned(index int, rootCanceled bool) {
	call, ok := b.call(index)
	if !ok {
		return
	}
	if call.phase != toolCallRunning || call.executionKnowledge != toolExecutionAttemptStarted {
		b.addViolation("call[%d] cannot observe attempt return from phase %q execution %q", index, call.phase, call.executionKnowledge)
		return
	}
	call.executionKnowledge = toolExecutionOutcomeObserved
	if rootCanceled {
		call.executionKnowledge = toolExecutionIndeterminate
	}
}

func (b *toolBlockState) markTerminal(index int, expected toolCallPhase, closure string) {
	call, ok := b.call(index)
	if !ok {
		return
	}
	call.terminalCount++
	if call.terminalCount != 1 {
		b.addViolation("call[%d] has %d terminal transitions", index, call.terminalCount)
		return
	}
	if call.phase != expected {
		b.addViolation("call[%d] closed by %q from phase %q, want %q", index, closure, call.phase, expected)
		return
	}
	if expected == toolCallAccepted && call.executionKnowledge != toolExecutionNotStarted {
		b.addViolation("call[%d] closed by %q from execution %q, want %q", index, closure, call.executionKnowledge, toolExecutionNotStarted)
	}
	if expected == toolCallRunning && call.executionKnowledge != toolExecutionOutcomeObserved && call.executionKnowledge != toolExecutionIndeterminate {
		b.addViolation("call[%d] closed by %q from execution %q, want terminal execution knowledge", index, closure, call.executionKnowledge)
	}
	call.phase = toolCallTerminal
	call.closure = closure
}

func (b *toolBlockState) markTerminalRange(start int, expected toolCallPhase, closure string) {
	if b == nil {
		return
	}
	if start < 0 || start > len(b.calls) {
		b.addViolation("terminal range start %d outside block length %d", start, len(b.calls))
		return
	}
	for i := start; i < len(b.calls); i++ {
		b.markTerminal(i, expected, closure)
	}
}

func (b *toolBlockState) validateClosed() error {
	if b == nil {
		return nil
	}
	violations := append([]string(nil), b.violations...)
	for i, call := range b.calls {
		if call.phase != toolCallTerminal || call.terminalCount != 1 || call.executionKnowledge == toolExecutionUnknown || call.executionKnowledge == toolExecutionAttemptStarted {
			violations = append(violations, fmt.Sprintf("call[%d] remains phase=%q execution=%q terminal_count=%d", i, call.phase, call.executionKnowledge, call.terminalCount))
		}
	}
	if len(violations) == 0 {
		return nil
	}
	return fmt.Errorf("%s", strings.Join(violations, "; "))
}

func (b *toolBlockState) call(index int) (*toolCallState, bool) {
	if b == nil {
		return nil, false
	}
	if index < 0 || index >= len(b.calls) {
		b.addViolation("call index %d outside block length %d", index, len(b.calls))
		return nil, false
	}
	return &b.calls[index], true
}

func (b *toolBlockState) addViolation(format string, args ...any) {
	if b == nil {
		return
	}
	b.violations = append(b.violations, fmt.Sprintf(format, args...))
}
