package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestAutomaticCompactionAsyncPreservesSampledDescriptor(t *testing.T) {
	for _, initialTodo := range []bool{false, true} {
		t.Run(fmt.Sprint(initialTodo), func(t *testing.T) {
			model := &localCompactionCountingModel{}
			ag, err := New(Config{LLM: model, Compaction: &compaction.Config{
				Enabled: true, ContextWindow: 100, ThresholdRatio: 0.85,
				SessionID: "descriptor", LedgerStore: &agentLocalLedgerStore{ledger: compaction.NewLedger("descriptor")},
				ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
					return compaction.ArtifactResult{Path: "fixture/tool.txt"}, nil
				}),
				ProtectedRecentMessages: 1,
			}})
			if err != nil {
				t.Fatal(err)
			}
			ag.ReplaceHistory([]llm.Message{
				llm.NewUserMessage("search"),
				llm.NewAssistantMessage("searching", []llm.ToolCall{{ID: "search", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
				llm.NewToolMessage("search", "grep", llm.TextContent(strings.Repeat("hit\n", 300)), false),
				llm.NewUserMessage("latest"),
			})
			ag.todoCompactionPending.Store(initialTodo)
			admissions := 0
			ag.compactionAdmissionObserved = func() {
				admissions++
				ag.todoCompactionPending.Store(!initialTodo)
			}
			if err := ag.checkAndCompactWithGrowth(context.Background(), "", &llm.Completion{Usage: llm.WithPromptEstimate(nil, 70)}, nil, 0, 0); err != nil {
				t.Fatal(err)
			}
			waitFor(t, time.Second, func() bool { return !ag.compactionInFlight.Load() }, "async descriptor completion")
			if admissions != 1 || model.calls.Load() != 0 {
				t.Fatalf("admissions=%d summary calls=%d", admissions, model.calls.Load())
			}
			ag.pendingCompactionMu.Lock()
			defer ag.pendingCompactionMu.Unlock()
			wantTrigger := "usage"
			if initialTodo {
				wantTrigger = "todo_checkpoint"
			}
			if ag.pendingCompaction == nil || ag.pendingCompaction.result.Trigger != wantTrigger || ag.pendingCompaction.result.Watermark != "snip" {
				t.Fatalf("pending=%+v; want %s/snip", ag.pendingCompaction, wantTrigger)
			}
			if ag.todoCompactionPending.Load() != !initialTodo {
				t.Fatal("executor consumed a newly sampled trigger")
			}
		})
	}
}

func TestAutomaticCompactionDecisionAdmission(t *testing.T) {
	for _, test := range []struct {
		name       string
		tokens     int
		cooldown   bool
		want       compactionDecision
		admissions int
	}{
		{"below", 69, false, compactionDecision{trigger: "usage"}, 1},
		{"snip", 70, false, compactionDecision{true, "usage", "snip"}, 1},
		{"prune", 80, false, compactionDecision{true, "usage", "prune"}, 1},
		{"summary", 85, false, compactionDecision{true, "usage", "summarize"}, 1},
		{"cooldown", 85, true, compactionDecision{false, "usage", "summarize"}, 1},
		{"overflow", 100, true, compactionDecision{true, "overflow", "overflow"}, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			ag, err := New(Config{LLM: &countingCompactionModel{}, Compaction: &compaction.Config{
				Enabled: true, ContextWindow: 100, SnipThresholdRatio: 0.70, PruneThresholdRatio: 0.80, ThresholdRatio: 0.85,
			}})
			if err != nil {
				t.Fatal(err)
			}
			if test.cooldown {
				ag.compactionCooldownUntil.Store(time.Now().Add(time.Hour).UnixNano())
			}
			admissions := 0
			ag.compactionAdmissionObserved = func() { admissions++ }
			got := ag.automaticCompactionDecision(context.Background(), llm.WithPromptEstimate(nil, test.tokens))
			if got != test.want || admissions != test.admissions {
				t.Fatalf("decision=%+v admissions=%d; want %+v admissions=%d", got, admissions, test.want, test.admissions)
			}
		})
	}
}

func TestAutomaticCompactionDriverAdmissionBoundaries(t *testing.T) {
	for _, test := range []struct {
		name           string
		tokens         int
		canceled       bool
		wantAdmissions int
		wantError      bool
	}{
		{"below", 69, false, 1, false},
		{"cooldown", 85, false, 1, false},
		{"overflow", 4000, false, 0, true},
		{"canceled", 85, true, 0, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := &countingCompactionModel{}
			ag, err := New(Config{LLM: model, InitialMessages: []llm.Message{llm.NewSystemMessage("base"), llm.NewUserMessage(strings.Repeat("constraint ", 3000))}, Compaction: &compaction.Config{
				Enabled: true, ContextWindow: 100, ThresholdRatio: 0.85,
			}})
			if err != nil {
				t.Fatal(err)
			}
			ag.compactionFailureStreak.Store(compactionSummaryDisableStreak)
			ag.compactionCooldownUntil.Store(time.Now().Add(time.Hour).UnixNano())
			admissions := 0
			ag.compactionAdmissionObserved = func() { admissions++ }
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if test.canceled {
				cancel()
			}
			err = ag.checkAndCompactWithGrowth(ctx, "", &llm.Completion{Usage: llm.WithPromptEstimate(nil, test.tokens)}, nil, 0, 0)
			if (err != nil) != test.wantError || admissions != test.wantAdmissions {
				t.Fatalf("err=%v admissions=%d; want error=%v admissions=%d", err, admissions, test.wantError, test.wantAdmissions)
			}
			if model.Calls() != 0 || ag.compactionInFlight.Load() || ag.hasPendingCompaction() {
				t.Fatal("suppressed or irreducible request launched summary or queued publication")
			}
		})
	}
}
