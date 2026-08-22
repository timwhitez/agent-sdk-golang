package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// emergencyTrimToolBlock builds one assistant tool_use message plus its tool
// results, i.e. one atomic block of the shapes below. resultWords sizes each
// result so a block can be made to fit or overflow a test budget on purpose.
func emergencyTrimToolBlock(prefix string, calls, resultWords int) []llm.Message {
	toolCalls := make([]llm.ToolCall, 0, calls)
	for i := 0; i < calls; i++ {
		toolCalls = append(toolCalls, llm.ToolCall{
			ID:       fmt.Sprintf("%s-%d", prefix, i),
			Type:     "function",
			Function: llm.FunctionCall{Name: "read", Arguments: `{"path":"f.go"}`},
		})
	}
	block := []llm.Message{llm.NewAssistantMessage("", toolCalls)}
	for _, call := range toolCalls {
		block = append(block, llm.NewToolMessage(
			call.ID, "read", llm.TextContent(strings.Repeat("result ", resultWords)), false))
	}
	return block
}

// assertEmergencyTrimPairing fails when the history is not a legal conversation:
// an orphan tool_result, a tool_use without its results, or a non-tool message
// spliced inside a parallel block. A trailing in-flight tool-call continuation is
// legitimately still unpaired and therefore exempt.
func assertEmergencyTrimPairing(t *testing.T, label string, messages []llm.Message) {
	t.Helper()
	inflight := pendingContinuationTailIndex(messages)
	for i := 0; i < len(messages); {
		m := messages[i]
		if m.Role == llm.RoleTool {
			t.Errorf("%s: orphan tool_result at index %d (call %q)", label, i, m.ToolCallID)
			i++
			continue
		}
		if m.Role != llm.RoleAssistant || len(m.ToolCalls) == 0 {
			i++
			continue
		}
		answered := make(map[string]bool, len(m.ToolCalls))
		for _, call := range m.ToolCalls {
			answered[call.ID] = false
		}
		j := i + 1
		for ; j < len(messages) && messages[j].Role == llm.RoleTool; j++ {
			id := messages[j].ToolCallID
			if _, ok := answered[id]; !ok {
				t.Errorf("%s: tool_result %q at index %d does not belong to the block at %d", label, id, j, i)
				continue
			}
			answered[id] = true
		}
		if i != inflight {
			for id, seen := range answered {
				if !seen {
					t.Errorf("%s: tool_use %q in the block at index %d has no tool_result", label, id, i)
				}
			}
		}
		i = j
	}
}

// REG (R4-CC-001): the backward walk retained the newest block unconditionally,
// and the budget re-check then rejected the WHOLE trim because that oversized
// block still overflowed - discarding the older content that provably would have
// fit. compactSyncOverflow turned the refusal into an ErrorEvent, so the turn
// aborted (and every later turn aborted identically) even though a legal
// in-budget history existed. That is precisely the failure mode the emergency
// trim exists to prevent.
func TestEmergencyTrimDropsOversizedNewestBlockInsteadOfRefusingWholeTrim(t *testing.T) {
	ag := emergencyTrimTestAgent(t, 1000)
	messages := []llm.Message{
		llm.NewSystemMessage("sys"),
		llm.NewUserMessage("please analyse the repository layout"),
	}
	for i := 0; i < 3; i++ {
		messages = append(messages, emergencyTrimToolBlock(fmt.Sprintf("small-%d", i), 1, 2)...)
	}
	messages = append(messages, emergencyTrimToolBlock("oversized", 1, 1500)...)

	budget := ag.compactor.ThresholdTokens()
	if estimate := ag.compactor.EstimateMessages(messages); estimate <= budget {
		t.Fatalf("history must start over budget: estimate=%d budget=%d", estimate, budget)
	}
	// A legal in-budget candidate provably exists: everything except the newest
	// (oversized) block.
	withoutNewest := messages[:len(messages)-2]
	if estimate := ag.compactor.EstimateMessages(withoutNewest); estimate > budget {
		t.Fatalf("fixture is wrong: dropping the newest block leaves %d tokens of a %d budget",
			estimate, budget)
	}

	trimmed, ok := ag.emergencyTrimHistory(messages)
	if !ok {
		t.Fatalf("emergency trim refused although a legal %d-token candidate exists for a %d budget",
			ag.compactor.EstimateMessages(withoutNewest), budget)
	}
	if estimate := ag.compactor.EstimateMessages(trimmed); estimate > budget {
		t.Fatalf("accepted trim is still over budget: estimate=%d budget=%d", estimate, budget)
	}
	if len(trimmed) >= len(messages) {
		t.Fatalf("accepted trim did not shrink history: %d -> %d", len(messages), len(trimmed))
	}
	assertEmergencyTrimPairing(t, "oversized-newest-block", trimmed)
	// The protected set must survive: system prefix and the real user request.
	if trimmed[0].Role != llm.RoleSystem {
		t.Fatalf("system prefix lost: first message role = %s", trimmed[0].Role)
	}
	foundRequest := false
	for _, m := range trimmed {
		if m.Role == llm.RoleUser && strings.Contains(m.Content.PlainText(), "analyse the repository layout") {
			foundRequest = true
		}
	}
	if !foundRequest {
		t.Fatal("the real user request was dropped by the trim")
	}
}

// REG (R4-CC-001): the same refusal at the realistic scale the review measured -
// a 200k window with a parallel block of large tool results. Pre-fix, parallel
// blocks of 8/14/16 reads all returned ok=false and terminated the turn.
func TestEmergencyTrimSurvivesOversizedParallelToolBlock(t *testing.T) {
	for _, parallel := range []int{8, 14, 16} {
		ag := emergencyTrimTestAgent(t, 170000)
		messages := []llm.Message{
			llm.NewSystemMessage("sys"),
			llm.NewUserMessage("read every file in the repository"),
		}
		messages = append(messages, emergencyTrimToolBlock("older", 3, 100)...)
		// One ~50KB tool result per parallel call, i.e. the default tool-result cap.
		messages = append(messages, emergencyTrimToolBlock("wide", parallel, 45000)...)

		budget := ag.compactor.ThresholdTokens()
		if estimate := ag.compactor.EstimateMessages(messages); estimate <= budget {
			t.Fatalf("parallel=%d: history must start over budget: estimate=%d budget=%d",
				parallel, estimate, budget)
		}
		trimmed, ok := ag.emergencyTrimHistory(messages)
		if !ok {
			t.Fatalf("parallel=%d: emergency trim refused; the turn would abort even though the protected prefix plus the older block fit", parallel)
		}
		if estimate := ag.compactor.EstimateMessages(trimmed); estimate > budget {
			t.Fatalf("parallel=%d: accepted trim is over budget: %d > %d", parallel, estimate, budget)
		}
		assertEmergencyTrimPairing(t, fmt.Sprintf("parallel-%d", parallel), trimmed)
	}
}

// REG (R4-CC-001): iterating towards smaller candidates must not become
// "trim to whatever is left". A history that cannot be reduced to a legal
// sendable in-budget conversation still has to report failure so the caller
// surfaces the original overflow error, and it must never be trimmed to a
// system-only history (Anthropic rejects empty messages, which would lose the
// user request and the session with it).
func TestEmergencyTrimStillRefusesGenuinelyIrreducibleHistory(t *testing.T) {
	ag := emergencyTrimTestAgent(t, 1000)
	// Only protected messages, and they alone blow the budget: there is nothing
	// legal left to give up.
	messages := []llm.Message{
		llm.NewSystemMessage("sys"),
		llm.NewUserMessage(strings.Repeat("word ", 3000)),
	}
	trimmed, ok := ag.emergencyTrimHistory(messages)
	if ok {
		t.Fatalf("emergency trim reported success for an irreducible history (%d tokens of a %d budget)",
			ag.compactor.EstimateMessages(trimmed), ag.compactor.ThresholdTokens())
	}
	if len(trimmed) != len(messages) {
		t.Fatalf("a refused trim must return the original history, got %d of %d messages",
			len(trimmed), len(messages))
	}

	// Unprotected content that cannot be reduced to a sendable history must not
	// be answered with a system-only result.
	systemOnly := []llm.Message{
		llm.NewSystemMessage("sys"),
		llm.NewAssistantMessage(strings.Repeat("x ", 3000), nil),
	}
	trimmed, ok = ag.emergencyTrimHistory(systemOnly)
	if ok {
		t.Fatalf("emergency trim accepted a result with no sendable non-system message: %d messages", len(trimmed))
	}
	if len(trimmed) != len(systemOnly) {
		t.Fatalf("a refused trim must return the original history, got %d of %d messages",
			len(trimmed), len(systemOnly))
	}
}

// REG (R4-CC-001): the iteration must keep every combination of framework
// interleavings a legal conversation. These are the parallel-block / loop-guard /
// steering / continuation-cap shapes the pairing invariant is defined over, run
// across budgets that force a trim and budgets that do not.
func TestEmergencyTrimKeepsPairingLegalAcrossFrameworkInterleavings(t *testing.T) {
	loopGuard := messageorigin.NewInternalUserMessage(messageorigin.KindLoopGuard, "loop guard reminder")
	continuationCap := messageorigin.NewInternalUserMessage(
		messageorigin.KindMaxTokensContinuation, messageorigin.ResponseTruncatedContinuationText)
	steering := llm.NewUserMessage("steering: change course")
	doneCall := llm.NewAssistantMessage("", []llm.ToolCall{{
		ID: "done-1", Type: "function", Function: llm.FunctionCall{Name: "done", Arguments: "{}"},
	}})
	doneResult := llm.NewToolMessage("done-1", "done", llm.TextContent("ok"), false)
	base := func(tail ...llm.Message) []llm.Message {
		out := []llm.Message{llm.NewSystemMessage("sys"), llm.NewUserMessage("do the work")}
		return append(out, tail...)
	}
	block := func(prefix string, calls int) []llm.Message {
		return emergencyTrimToolBlock(prefix, calls, 300)
	}
	inflight := llm.NewAssistantMessage("partial", []llm.ToolCall{{
		ID: "inflight", Type: "function",
		Function: llm.FunctionCall{Name: "write", Arguments: `{"path":"x.go","content":"package ma`},
	}})

	scenarios := map[string][]llm.Message{}
	scenarios["parallel_block_then_done"] = base(append(block("p", 2), doneCall, doneResult)...)
	scenarios["loop_guard_between_blocks"] = base(
		concatMessages(block("a", 2), []llm.Message{loopGuard}, block("b", 3))...)
	scenarios["steering_between_blocks"] = base(
		concatMessages(block("a", 2), []llm.Message{steering}, block("b", 2))...)
	scenarios["continuation_cap_between_blocks"] = base(
		concatMessages(block("a", 2), []llm.Message{continuationCap}, block("b", 2))...)
	scenarios["loop_guard_and_steering"] = base(concatMessages(
		block("a", 2), []llm.Message{loopGuard}, block("b", 2), []llm.Message{steering}, block("c", 2))...)
	scenarios["done_not_last_and_steering"] = base(concatMessages(
		[]llm.Message{doneCall, doneResult}, []llm.Message{steering}, block("a", 2))...)
	scenarios["loop_guard_and_done_not_last"] = base(concatMessages(
		block("a", 2), []llm.Message{loopGuard}, []llm.Message{doneCall, doneResult}, block("b", 2))...)
	scenarios["continuation_cap_and_loop_guard"] = base(concatMessages(
		block("a", 2), []llm.Message{continuationCap, loopGuard}, block("b", 2))...)
	scenarios["continuation_cap_and_steering"] = base(concatMessages(
		block("a", 3), []llm.Message{continuationCap}, block("b", 1), []llm.Message{steering}, block("c", 2))...)
	scenarios["continuation_cap_steering_inflight_tail"] = base(concatMessages(
		block("a", 3), []llm.Message{continuationCap}, block("b", 1), []llm.Message{steering},
		block("c", 2), []llm.Message{inflight})...)
	scenarios["all_three_interleavings"] = base(concatMessages(
		block("a", 2), []llm.Message{loopGuard}, block("b", 2), []llm.Message{continuationCap},
		block("c", 2), []llm.Message{steering}, []llm.Message{doneCall, doneResult})...)
	if len(scenarios) != 11 {
		t.Fatalf("expected 11 interleaving scenarios, built %d", len(scenarios))
	}

	for _, window := range []int{400, 1200, 2500} {
		for name, messages := range scenarios {
			ag := emergencyTrimTestAgent(t, window)
			trimmed, ok := ag.emergencyTrimHistory(messages)
			label := fmt.Sprintf("%s(window=%d,trimmed=%v)", name, window, ok)
			assertEmergencyTrimPairing(t, label, trimmed)
			if ok {
				if estimate := ag.compactor.EstimateMessages(trimmed); estimate > window {
					t.Errorf("%s: accepted trim is over budget: %d > %d", label, estimate, window)
				}
				if len(trimmed) >= len(messages) {
					t.Errorf("%s: reported a trim that did not shrink history: %d -> %d",
						label, len(messages), len(trimmed))
				}
			} else if len(trimmed) != len(messages) {
				t.Errorf("%s: a refused trim must return the original history, got %d of %d",
					label, len(trimmed), len(messages))
			}
			sendable := 0
			for _, m := range trimmed {
				if m.Role != llm.RoleSystem {
					sendable++
				}
			}
			if sendable == 0 {
				t.Errorf("%s: trimmed to a system-only history", label)
			}
			for _, m := range messages {
				if !emergencyTrimProtected(m) {
					continue
				}
				kept := false
				for _, candidate := range trimmed {
					if candidate.Role == m.Role && candidate.Name == m.Name &&
						candidate.Content.PlainText() == m.Content.PlainText() {
						kept = true
						break
					}
				}
				if !kept {
					t.Errorf("%s: protected message (role=%s name=%q) was dropped", label, m.Role, m.Name)
				}
			}
		}
	}
}

func concatMessages(groups ...[]llm.Message) []llm.Message {
	total := 0
	for _, g := range groups {
		total += len(g)
	}
	out := make([]llm.Message, 0, total)
	for _, g := range groups {
		out = append(out, g...)
	}
	return out
}

// REG (R4-CC-002): the '!res.Compacted && !allowSummary' branch dropped
// applyEmergencyTrim's bool and returned nil unconditionally, so a turn whose
// history could not be reduced reported success and then sent a request that was
// still far over the window. The sibling error branch already propagated; this
// one masked the overflow.
func TestCompactSyncOverflowPropagatesFailedEmergencyTrim(t *testing.T) {
	ag, err := New(Config{
		LLM: &countingCompactionModel{},
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  850,
			ThresholdRatio: 1.0,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Protected-only, irreducible history: nothing legal can be given up.
	ag.ReplaceHistory([]llm.Message{
		llm.NewSystemMessage("sys"),
		llm.NewUserMessage(strings.Repeat("word ", 3000)),
	})
	// Drive the failure streak so the summary tier is suppressed and the overflow
	// path takes the !allowSummary branch.
	ag.compactionFailureStreak.Store(compactionSummaryDisableStreak)
	if allowSummary, _ := ag.overflowSummaryPlan(); allowSummary {
		t.Fatal("fixture is wrong: the summary tier must be suppressed for this branch")
	}

	out := make(chan Event, 64)
	completion := &llm.Completion{Usage: &llm.Usage{TotalTokens: 4000, PromptTokens: 4000}}
	compactErr := ag.compactSyncOverflow(context.Background(), completion, completion.Usage, out)
	budget := ag.compactor.ThresholdTokens()
	estimate := ag.compactor.EstimateMessages(ag.Messages())
	if estimate <= budget {
		t.Fatalf("fixture is wrong: history was reduced to %d tokens of a %d budget", estimate, budget)
	}
	if compactErr == nil {
		t.Fatalf("compactSyncOverflow reported success while the history is still %d tokens over the %d budget",
			estimate-budget, budget)
	}
}
