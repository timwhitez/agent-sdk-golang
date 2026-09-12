package agent

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestOverflowSummaryAttemptBudget(t *testing.T) {
	for _, tt := range []struct {
		name      string
		cooldown  bool
		streak    uint64
		canceled  bool
		wantCalls int
	}{
		{name: "configured three retries", wantCalls: 4},
		{name: "cooldown allows one attempt", cooldown: true, wantCalls: 1},
		{name: "failure streak suppresses summary", streak: compactionSummaryDisableStreak},
		{name: "cancellation prevents summary", canceled: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model := &flakyCompactionModel{failFor: 100}
			// All history is protected and irreducible, so neither local tiers nor
			// emergency trim can hide the failed summary or allow a later request.
			original := []llm.Message{llm.NewSystemMessage("base"), llm.NewUserMessage(strings.Repeat("constraint ", 3000))}
			ag, err := New(Config{LLM: model, InitialMessages: original, Compaction: &compaction.Config{
				Enabled: true, ContextWindow: 100, ThresholdRatio: 0.85,
				CompactionRetries: 3, CompactionRetryBackoff: time.Millisecond,
			}})
			if err != nil {
				t.Fatal(err)
			}
			ag.compactionFailureStreak.Store(tt.streak)
			if tt.cooldown {
				ag.compactionCooldownUntil.Store(time.Now().Add(time.Hour).UnixNano())
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tt.canceled {
				cancel()
			}
			usage := llm.WithPromptEstimate(nil, 4000)
			err = ag.checkAndCompactWithGrowth(ctx, &llm.Completion{Usage: usage}, nil, 0, 0)
			if err == nil {
				t.Error("irreducible overflow must remain an error")
			}
			if tt.canceled && !errors.Is(err, context.Canceled) {
				t.Errorf("canceled error = %v", err)
			}
			if got := model.Calls(); got != tt.wantCalls {
				t.Errorf("summary Invoke calls = %d, want %d", got, tt.wantCalls)
			}
			if !reflect.DeepEqual(ag.Messages(), original) || ag.hasPendingCompaction() {
				t.Error("failed overflow published or queued replacement history")
			}
		})
	}
}

func TestPublicCompactionKeepsExplicitSummaryPolicyDuringFailureSuppression(t *testing.T) {
	for _, tt := range []struct {
		name      string
		manual    bool
		request   compaction.PipelineRequest
		wantCalls int
	}{
		{name: "CompactNow forces below threshold", manual: true, wantCalls: 4},
		{name: "explicit allow", request: compaction.PipelineRequest{Trigger: "preflight", TargetWatermark: "summarize", EstimatedTokens: 900000, AllowSummary: true}, wantCalls: 4},
		{name: "explicit deny", request: compaction.PipelineRequest{Trigger: "preflight", TargetWatermark: "summarize", EstimatedTokens: 900000}},
		{name: "explicit force overrides deny", request: compaction.PipelineRequest{Trigger: "manual", ForceSummary: true}, wantCalls: 4},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model := &flakyCompactionModel{failFor: 100}
			original := []llm.Message{llm.NewSystemMessage("base"), llm.NewUserMessage("small request")}
			ag, err := New(Config{LLM: model, InitialMessages: original, Compaction: &compaction.Config{
				Enabled: true, ContextWindow: 1000000, ThresholdRatio: 0.85,
				CompactionRetries: 3, CompactionRetryBackoff: time.Millisecond,
			}})
			if err != nil {
				t.Fatal(err)
			}
			ag.compactionFailureStreak.Store(compactionSummaryDisableStreak)
			ag.compactionCooldownUntil.Store(time.Now().Add(time.Hour).UnixNano())
			var result compaction.Result
			if tt.manual {
				result, err = ag.CompactNow(context.Background())
			} else {
				result, err = ag.CompactPipelineNow(context.Background(), tt.request)
			}
			if (err != nil) != (tt.wantCalls > 0) {
				t.Errorf("error = %v, want failure iff summary was requested", err)
			}
			if got := model.Calls(); got != tt.wantCalls {
				t.Errorf("summary Invoke calls = %d, want %d", got, tt.wantCalls)
			}
			if result.Compacted || !reflect.DeepEqual(ag.Messages(), original) {
				t.Error("failed or no-op public compaction replaced history")
			}
		})
	}
}
