package agent

import (
	"fmt"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type toolCallPhase string
type toolExecutionKnowledge uint8

const (
	toolCallAccepted     toolCallPhase          = "accepted"
	toolCallRunning      toolCallPhase          = "running"
	toolCallTerminal     toolCallPhase          = "terminal"
	toolExecutionUnknown toolExecutionKnowledge = iota
	toolExecutionNotStarted
	toolExecutionAttemptStarted
	toolExecutionOutcomeObserved
	toolExecutionIndeterminate
)

type toolPublication uint8

const (
	toolPublicationNone toolPublication = iota // history-only terminal
	toolPublicationPending
	toolPublicationClaimed // not a claim of delivery
	toolPublicationAborted
)

type toolCallState struct {
	id, name           string
	phase              toolCallPhase
	executionKnowledge toolExecutionKnowledge
	terminalCount      int
	closure            string
	publication        toolPublication
	result             *toolResultProjection
}

type toolBlockTransitionError struct {
	index int
	code  string
}

func (e *toolBlockTransitionError) Error() string {
	return fmt.Sprintf("tool lifecycle call[%d]: %s", e.index, e.code)
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

// toolBlockState is query/block-local terminal and publication authority, owned
// by the sequential driver (not a concurrent scheduler or a durable ledger).
// It retains accepted identity, never arguments. Pending projection payloads
// are released on publication claim or abort; claim does not imply delivery.
type toolBlockState struct {
	calls        []toolCallState
	nextTerminal int
}

func newToolBlockState(calls []llm.ToolCall) (*toolBlockState, error) {
	b := &toolBlockState{calls: make([]toolCallState, len(calls))}
	seen := make(map[string]bool, len(calls))
	for i, call := range calls {
		id := strings.TrimSpace(call.ID)
		if id == "" {
			return nil, &toolBlockTransitionError{i, "empty_id"}
		}
		if seen[id] {
			return nil, &toolBlockTransitionError{i, "duplicate_id"}
		}
		seen[id] = true
		name := strings.TrimSpace(call.Function.Name)
		if name == "" {
			name = "unknown"
		}
		b.calls[i] = toolCallState{id: id, name: name, phase: toolCallAccepted, executionKnowledge: toolExecutionNotStarted}
	}
	return b, nil
}

func (b *toolBlockState) call(index int) (*toolCallState, error) {
	if b == nil || index < 0 || index >= len(b.calls) {
		return nil, &toolBlockTransitionError{index, "out_of_range"}
	}
	return &b.calls[index], nil
}

func (b *toolBlockState) markRunning(index int) error {
	call, err := b.call(index)
	if err != nil {
		return err
	}
	if call.phase != toolCallAccepted || call.executionKnowledge != toolExecutionNotStarted || call.terminalCount != 0 {
		return &toolBlockTransitionError{index, "cannot_start"}
	}
	call.phase = toolCallRunning
	call.executionKnowledge = toolExecutionAttemptStarted
	return nil
}

func (b *toolBlockState) markAttemptReturned(index int, rootCanceled bool) error {
	call, err := b.call(index)
	if err != nil {
		return err
	}
	if call.phase != toolCallRunning || call.executionKnowledge != toolExecutionAttemptStarted || call.terminalCount != 0 {
		return &toolBlockTransitionError{index, "cannot_observe_return"}
	}
	call.executionKnowledge = toolExecutionOutcomeObserved
	if rootCanceled {
		call.executionKnowledge = toolExecutionIndeterminate
	}
	return nil
}

// acceptResults validates the entire batch before accepting any result. The
// caller appends returned history exactly once; rejected proposals have no effect.
func (b *toolBlockState) acceptResults(start int, expected toolCallPhase, closure string, results []toolResultProjection) ([]llm.Message, error) {
	if b == nil || start < 0 || start > len(b.calls) || len(results) > len(b.calls)-start {
		return nil, &toolBlockTransitionError{start, "invalid_range"}
	}
	if expected != toolCallAccepted && expected != toolCallRunning {
		return nil, &toolBlockTransitionError{start, "invalid_expected_phase"}
	}
	if start != b.nextTerminal {
		if start < b.nextTerminal && len(results) > 0 {
			return nil, &toolBlockTransitionError{start, "duplicate_terminal"}
		}
		return nil, &toolBlockTransitionError{start, "out_of_order_terminal"}
	}
	for offset, result := range results {
		index := start + offset
		call := &b.calls[index]
		if call.phase == toolCallTerminal || call.terminalCount != 0 {
			return nil, &toolBlockTransitionError{index, "duplicate_terminal"}
		}
		if call.phase != expected {
			return nil, &toolBlockTransitionError{index, "wrong_phase"}
		}
		if (expected == toolCallAccepted && call.executionKnowledge != toolExecutionNotStarted) || (expected == toolCallRunning && call.executionKnowledge != toolExecutionOutcomeObserved && call.executionKnowledge != toolExecutionIndeterminate) {
			return nil, &toolBlockTransitionError{index, "invalid_execution_knowledge"}
		}
		if result.history.Role != llm.RoleTool || result.history.ToolCallID != call.id || len(result.history.ToolCalls) != 0 {
			return nil, &toolBlockTransitionError{index, "invalid_result_identity"}
		}
	}
	history := make([]llm.Message, len(results))
	for offset, result := range results {
		call := &b.calls[start+offset]
		result.history = llm.CloneMessage(result.history)
		if result.metadata != nil {
			result.metadata = cloneToolResultMetadata(result.metadata)
		}
		history[offset] = llm.CloneMessage(result.history)
		call.phase = toolCallTerminal
		call.terminalCount = 1
		call.closure = closure
		if result.publish {
			call.publication = toolPublicationPending
			call.result = &result
		}
	}
	b.nextTerminal += len(results)
	return history, nil
}

func (b *toolBlockState) takePublication(index int) (toolResultProjection, error) {
	call, err := b.call(index)
	if err != nil {
		return toolResultProjection{}, err
	}
	if call.phase != toolCallTerminal || call.terminalCount != 1 || index >= b.nextTerminal || call.publication != toolPublicationPending || call.result == nil {
		return toolResultProjection{}, &toolBlockTransitionError{index, "publication_unavailable"}
	}
	result := *call.result
	call.result = nil
	call.publication = toolPublicationClaimed
	return result, nil
}

// abortOpen is an idempotent, non-reentrant recovery path. It never rewrites
// terminal history. Observed execution and failed projection remain distinct.
func (b *toolBlockState) abortOpen() []llm.Message {
	if b == nil {
		return nil
	}
	var rows []llm.Message
	for i := range b.calls {
		call := &b.calls[i]
		call.result = nil
		if call.publication == toolPublicationPending {
			call.publication = toolPublicationAborted
		}
		if call.terminalCount > 0 {
			call.phase = toolCallTerminal
			continue
		}
		text := "[ERROR] Tool result unavailable after an internal lifecycle failure; result is indeterminate and execution may have occurred."
		if call.phase == toolCallAccepted && call.executionKnowledge == toolExecutionNotStarted {
			text = "[ERROR] Tool skipped before execution because of an internal lifecycle failure."
		} else if call.executionKnowledge != toolExecutionOutcomeObserved {
			call.executionKnowledge = toolExecutionIndeterminate
		}
		rows = append(rows, llm.NewToolMessage(call.id, call.name, llm.TextContent(text), true))
		call.phase = toolCallTerminal
		call.terminalCount = 1
		call.closure = "lifecycle_failure"
	}
	b.nextTerminal = len(b.calls)
	return rows
}

func (b *toolBlockState) validateClosed() error {
	if b == nil {
		return nil
	}
	if b.nextTerminal != len(b.calls) {
		return &toolBlockTransitionError{b.nextTerminal, "incomplete_block"}
	}
	for i, call := range b.calls {
		if call.phase != toolCallTerminal || call.terminalCount != 1 || call.executionKnowledge == toolExecutionUnknown || call.executionKnowledge == toolExecutionAttemptStarted || call.publication == toolPublicationPending || call.result != nil {
			return &toolBlockTransitionError{i, "incomplete_block"}
		}
	}
	return nil
}
