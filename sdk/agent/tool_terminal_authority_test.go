package agent

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func terminalFixture(t *testing.T, ids ...string) *toolBlockState {
	t.Helper()
	calls := make([]llm.ToolCall, len(ids))
	for i, id := range ids {
		calls[i] = llm.ToolCall{ID: id, Function: llm.FunctionCall{Name: "private-tool"}}
	}
	b, err := newToolBlockState(calls)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestTerminalDriverRejectsAndClosesRemainingCalls(t *testing.T) {
	for _, mode := range []string{"start", "return", "commit", "second_commit", "duplicate_commit", "steering_tail"} {
		t.Run(mode, func(t *testing.T) {
			var ag *Agent
			var block *toolBlockState
			handled, providers, injected := 0, 0, false
			steering := make(chan SteeringMsg, 1)
			model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
				providers++
				if providers > 1 {
					return &llm.Completion{Content: llm.TextContent("next")}, nil
				}
				return &llm.Completion{ToolCalls: []llm.ToolCall{
					{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: "{}"}},
					{ID: "b", Function: llm.FunctionCall{Name: "work", Arguments: "{}"}},
					{ID: "c", Function: llm.FunctionCall{Name: "work", Arguments: "{}"}},
				}}, nil
			}}
			var err error
			ag, err = New(Config{LLM: model, Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				handled++
				if mode == "steering_tail" {
					steering <- SteeringMsg{Content: "retained steering input"}
				}
				if mode == "return" {
					block.calls[0].phase = toolCallAccepted
					injected = true
				}
				return llm.TextContent("completed fixture effect"), nil
			}}}, ToolResultTokenEstimator: func(text string) int {
				if injected || block == nil {
					return len(text)
				}
				index := 0
				if mode == "second_commit" {
					index = 1
				}
				if handled != index+1 {
					return len(text)
				}
				if mode == "commit" || mode == "second_commit" {
					block.calls[index].phase = toolCallAccepted
					injected = true
				}
				if mode == "duplicate_commit" {
					p := terminalProjection("a")
					p.history.Content = llm.TextContent("already committed")
					rows, commitErr := block.acceptResults(0, toolCallRunning, "first", []toolResultProjection{p})
					if commitErr != nil {
						t.Error(commitErr)
					}
					ag.appendMessages(rows) // inject a completed first commit before the attempted duplicate
					injected = true
				}
				return len(text)
			}, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			ag.toolBlockTestHook = func(b *toolBlockState) {
				block = b
				if mode == "steering_tail" {
					b.calls[1].phase = toolCallRunning
					injected = true
				}
				if mode == "start" {
					b.calls[0].phase = toolCallRunning
					injected = true
				}
			}
			errorsSeen, resultEvents, accountEvents := 0, 0, 0
			for envelope := range ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("private prompt"), steering) {
				switch event := envelope.Event.(type) {
				case ErrorEvent:
					errorsSeen++
					if event.Kind != "invalid_tool_call_block" || envelope.Origin != EventOriginSDKDriver || strings.Contains(event.Message, "private") {
						t.Error("unsafe lifecycle failure diagnostic")
					}
				case ToolResultEvent:
					resultEvents++
				case AccountingEvent:
					if event.ToolCallID != "" {
						accountEvents++
					}
				}
			}
			wantHandled, wantEvents := 1, 0
			if mode == "start" {
				wantHandled = 0
			}
			if mode == "second_commit" {
				wantHandled = 2
				wantEvents = 1
			}
			if mode == "steering_tail" {
				wantEvents = 1
			}
			if !injected || providers != 1 || handled != wantHandled || errorsSeen != 1 || resultEvents != wantEvents || accountEvents != wantEvents {
				t.Fatalf("injected/provider/handler/errors/result/account=%v/%d/%d/%d/%d/%d", injected, providers, handled, errorsSeen, resultEvents, accountEvents)
			}
			var results []llm.Message
			for _, message := range ag.Messages() {
				if message.Role == llm.RoleTool {
					results = append(results, message)
				}
			}
			if len(results) != 3 {
				t.Fatalf("terminal history count=%d", len(results))
			}
			if mode == "steering_tail" {
				history := ag.Messages()
				if history[len(history)-1].Role != llm.RoleUser || history[len(history)-1].Content.PlainText() != "retained steering input" {
					t.Fatal("rejection lost consumed steering or appended it before closure")
				}
			}
			for i, id := range []string{"a", "b", "c"} {
				if results[i].ToolCallID != id || block.calls[i].terminalCount != 1 {
					t.Error("lost/duplicate terminal identity")
				}
			}
			if mode == "commit" && block.calls[0].executionKnowledge != toolExecutionOutcomeObserved {
				t.Error("projection rejection erased observed return")
			}
			if mode == "return" && block.calls[0].executionKnowledge != toolExecutionIndeterminate {
				t.Error("unconfirmed return not indeterminate")
			}
			if mode == "second_commit" && (results[0].IsError || !results[1].IsError || block.calls[1].executionKnowledge != toolExecutionOutcomeObserved) {
				t.Error("abort rewrote prior terminal or erased observed return")
			}
			if mode == "duplicate_commit" && results[0].Content.PlainText() != "already committed" {
				t.Error("duplicate rewrote first terminal")
			}
			if err := block.validateClosed(); err != nil {
				t.Fatal(err)
			}
			if _, changed, _ := repairToolCallPairsDetailed(ag.Messages()); changed {
				t.Fatal("abort left dangling topology")
			}
			ag.toolBlockTestHook = nil
			for event := range ag.QueryStream(context.Background(), llm.TextContent("next")) {
				if warning, ok := event.(WarnEvent); ok && warning.Kind == "tool_pairing_repaired" {
					t.Error("next query required repair")
				}
			}
			if providers != 2 {
				t.Fatal("next query failed")
			}
		})
	}
}

func terminalProjection(id string) toolResultProjection {
	return projectToolResult(llm.Message{Role: llm.RoleTool, ToolCallID: id, ToolName: "resolved-alias", Content: llm.TextContent("visible")}, map[string]any{"private": "metadata"}, "private original")
}

func acceptHistoryTerminal(t *testing.T, b *toolBlockState, index int, phase toolCallPhase, reason string) {
	t.Helper()
	if _, err := b.acceptResults(index, phase, reason, historyOnlyResults([]llm.Message{{Role: llm.RoleTool, ToolCallID: b.calls[index].id}})); err != nil {
		t.Fatal(err)
	}
}

func TestTerminalAcceptanceIsAtomicAndPublicationIsOneShot(t *testing.T) {
	b := terminalFixture(t, "a", "b", "c")
	proposals := []toolResultProjection{terminalProjection("a"), terminalProjection("wrong-private-id")}
	before := append([]toolCallState(nil), b.calls...)
	if _, err := b.acceptResults(0, toolCallAccepted, "guard", proposals); err == nil {
		t.Fatal("bad batch accepted")
	} else {
		var typed *toolBlockTransitionError
		if !errors.As(err, &typed) || strings.Contains(err.Error(), "private") {
			t.Fatal("error is untyped or leaks data")
		}
	}
	if !reflect.DeepEqual(before, b.calls) {
		t.Fatal("batch partially accepted")
	}
	proposals[1] = terminalProjection("b")
	history, err := b.acceptResults(0, toolCallAccepted, "guard", proposals)
	if err != nil || len(history) != 2 {
		t.Fatalf("history=%d err=%v", len(history), err)
	}
	if _, err = b.acceptResults(0, toolCallAccepted, "duplicate", proposals); err == nil {
		t.Fatal("duplicate accepted")
	}
	if b.calls[0].terminalCount != 1 || b.calls[1].terminalCount != 1 {
		t.Fatal("duplicate changed accepted state")
	}
	for i := 0; i < 2; i++ {
		p, err := b.takePublication(i)
		if err != nil || p.history.ToolCallID != history[i].ToolCallID || p.original != "private original" {
			t.Fatal("wrong publication")
		}
		if b.calls[i].result != nil {
			t.Fatal("claimed result retains raw payload")
		}
		if _, err := b.takePublication(i); err == nil {
			t.Fatal("duplicate publication allowed")
		}
	}
	tail := historyOnlyResults([]llm.Message{{Role: llm.RoleTool, ToolCallID: "c", Content: llm.TextContent("skipped")}})
	if _, err := b.acceptResults(2, toolCallAccepted, "tail", tail); err != nil {
		t.Fatal(err)
	}
	if _, err := b.takePublication(2); err == nil {
		t.Fatal("history-only tail published")
	}
	if err := b.validateClosed(); err != nil {
		t.Fatal(err)
	}
	if _, err := b.acceptResults(3, toolCallAccepted, "empty", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := b.acceptResults(4, toolCallAccepted, "invalid-empty", nil); err == nil {
		t.Fatal("invalid empty range accepted")
	}
}

func TestTerminalAbortPreservesClosedAndExecutionKnowledge(t *testing.T) {
	b := terminalFixture(t, "closed", "observed", "running", "unstarted")
	if _, err := b.acceptResults(0, toolCallAccepted, "guard", []toolResultProjection{terminalProjection("closed")}); err != nil {
		t.Fatal(err)
	}
	if _, err := b.takePublication(0); err != nil {
		t.Fatal(err)
	}
	if err := b.markRunning(1); err != nil {
		t.Fatal(err)
	}
	if err := b.markAttemptReturned(1, false); err != nil {
		t.Fatal(err)
	}
	if err := b.markRunning(2); err != nil {
		t.Fatal(err)
	}
	rows := b.abortOpen()
	if len(rows) != 3 || rows[0].ToolCallID != "observed" || rows[1].ToolCallID != "running" || rows[2].ToolCallID != "unstarted" {
		t.Fatal("abort lost ordinal identity or rewrote closed call")
	}
	if b.calls[1].executionKnowledge != toolExecutionOutcomeObserved || b.calls[2].executionKnowledge != toolExecutionIndeterminate || b.calls[3].executionKnowledge != toolExecutionNotStarted {
		t.Fatal("abort conflated execution evidence")
	}
	for i, row := range rows {
		if !row.IsError || (i < 2 && !strings.Contains(row.Content.PlainText(), "indeterminate")) {
			t.Fatal("unsafe abort representation")
		}
	}
	if len(b.abortOpen()) != 0 {
		t.Fatal("abort not idempotent")
	}
	if err := b.validateClosed(); err != nil {
		t.Fatal(err)
	}
}

func TestTerminalRejectionsDoNotMutateAndAcceptedRecordIsOwned(t *testing.T) {
	b := terminalFixture(t, "a", "b")
	if _, err := b.acceptResults(1, toolCallAccepted, "wrong_order", []toolResultProjection{terminalProjection("b")}); err == nil || b.nextTerminal != 0 {
		t.Fatal("out-of-order result accepted")
	}
	if err := b.markAttemptReturned(0, false); err == nil {
		t.Fatal("return before start accepted")
	}
	if err := b.markRunning(0); err != nil {
		t.Fatal(err)
	}
	if err := b.markRunning(0); err == nil {
		t.Fatal("duplicate start accepted")
	}
	p := terminalProjection("a")
	if _, err := b.acceptResults(0, toolCallRunning, "early", []toolResultProjection{p}); err == nil {
		t.Fatal("unobserved return accepted")
	}
	if err := b.markAttemptReturned(0, false); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []llm.Message{{Role: llm.RoleUser, ToolCallID: "a"}, {Role: llm.RoleTool, ToolCallID: "private-wrong"}, {Role: llm.RoleTool, ToolCallID: "a", ToolCalls: []llm.ToolCall{{ID: "nested"}}}} {
		candidate := p
		candidate.history = bad
		if _, err := b.acceptResults(0, toolCallRunning, "bad", []toolResultProjection{candidate}); err == nil {
			t.Fatal("invalid result accepted")
		}
		if b.nextTerminal != 0 || b.calls[0].terminalCount != 0 || b.calls[0].result != nil {
			t.Fatal("rejected proposal changed authority")
		}
	}
	rows, err := b.acceptResults(0, toolCallRunning, "returned", []toolResultProjection{p})
	if err != nil {
		t.Fatal(err)
	}
	p.history.ToolCallID = "mutated"
	p.metadata["private"] = "mutated"
	rows[0].Content = llm.TextContent("mutated")
	claimed, err := b.takePublication(0)
	if err != nil || claimed.history.ToolCallID != "a" || claimed.history.Content.PlainText() != "visible" || claimed.metadata["private"] != "metadata" {
		t.Fatal("accepted record aliases candidate or returned history")
	}
	if len(b.abortOpen()) != 1 {
		t.Fatal("unstarted sibling lost")
	}
	if err := b.validateClosed(); err != nil {
		t.Fatal(err)
	}
}

func TestTerminalDriverPublicationRejectionKeepsGuardAndClosesTail(t *testing.T) {
	var block *toolBlockState
	providers, handled := 0, 0
	model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
		providers++
		if providers > 1 {
			return &llm.Completion{Content: llm.TextContent("next")}, nil
		}
		var calls []llm.ToolCall
		for _, id := range []string{"a", "b", "c", "d"} {
			calls = append(calls, llm.ToolCall{ID: id, Function: llm.FunctionCall{Name: "work", Arguments: "{}"}})
		}
		return &llm.Completion{ToolCalls: calls}, nil
	}}
	ag, err := New(Config{LLM: model, RepeatToolSignatureThreshold: 3, LoopGuardUserMessage: "guard reminder", Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		handled++
		return llm.TextContent("ok"), nil
	}}}, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	ag.toolBlockTestHook = func(b *toolBlockState) { block = b }
	claimedEarly := false
	ag.eventClock = func() time.Time {
		if !claimedEarly && block != nil && block.calls[2].publication == toolPublicationPending {
			claimedEarly = true
			if _, err := block.takePublication(2); err != nil {
				t.Error(err)
			}
		}
		return time.Unix(0, 0)
	}
	errorsSeen, results, accounting := 0, 0, 0
	for envelope := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("run")) {
		switch event := envelope.Event.(type) {
		case ErrorEvent:
			errorsSeen++
			if event.Kind != "invalid_tool_call_block" || envelope.Origin != EventOriginSDKDriver {
				t.Error("wrong publication failure")
			}
		case ToolResultEvent:
			results++
		case AccountingEvent:
			if event.ToolCallID != "" {
				accounting++
			}
		}
	}
	if !claimedEarly || providers != 1 || handled != 2 || errorsSeen != 1 || results != 2 || accounting != 2 {
		t.Fatalf("claimed/provider/handler/errors/results/accounting=%v/%d/%d/%d/%d/%d", claimedEarly, providers, handled, errorsSeen, results, accounting)
	}
	var rows []llm.Message
	for _, m := range ag.Messages() {
		if m.Role == llm.RoleTool {
			rows = append(rows, m)
		}
	}
	if len(rows) != 4 || !strings.Contains(rows[2].Content.PlainText(), "loop guard") || !strings.Contains(rows[3].Content.PlainText(), "skipped before execution") {
		t.Fatal("publication rejection lost committed guard or open tail")
	}
	if block.calls[2].publication != toolPublicationClaimed || block.calls[2].result != nil || block.calls[3].executionKnowledge != toolExecutionNotStarted {
		t.Fatal("claim/delivery/abort evidence conflated")
	}
	if _, changed, _ := repairToolCallPairsDetailed(ag.Messages()); changed {
		t.Fatal("publication rejection left malformed history")
	}
	history := ag.Messages()
	if history[len(history)-1].Content.PlainText() != "guard reminder" {
		t.Fatal("guard reminder not preserved after closure")
	}
	if err := block.validateClosed(); err != nil {
		t.Fatal(err)
	}
}
