package compaction

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type orderedPipelineModel struct {
	sequence *[]string
	calls    int
}

type failingPipelineSummaryModel struct{}

func (failingPipelineSummaryModel) Provider() string { return "mock" }
func (failingPipelineSummaryModel) Model() string    { return "mock" }
func (failingPipelineSummaryModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("injected summary failure")
}

func (m *orderedPipelineModel) Provider() string { return "mock" }
func (m *orderedPipelineModel) Model() string    { return "mock" }
func (m *orderedPipelineModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.sequence != nil {
		*m.sequence = append(*m.sequence, "summary")
	}
	return &llm.Completion{Content: llm.TextContent(structuredTestSummary("Completed Work", "pipeline summary with enough durable detail for the fixture"))}, nil
}

func TestAutoSummarizeRunsPruneBeforeSummaryWhenWatermarkIsSkipped(t *testing.T) {
	sequence := []string{}
	svc := newOrderedPipelineService(t, &sequence, false)
	model := &orderedPipelineModel{sequence: &sequence}

	_, res, err := svc.CompactPipeline(context.Background(), model, orderedPipelineMessages(), PipelineRequest{
		Trigger:         "usage",
		Usage:           &llm.Usage{PromptTokens: 180, TotalTokens: 180},
		TargetWatermark: "summarize",
		AllowSummary:    true,
	})
	if err != nil {
		t.Fatalf("CompactPipeline: %v", err)
	}
	if got, want := sequence, []string{"artifact:grep", "artifact:assistant", "summary"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("pipeline order = %#v, want %#v", got, want)
	}
	if got, want := res.TiersApplied, []string{"snip", "prune", "summarize"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("tiers = %#v, want %#v", got, want)
	}
}

func TestPipelineAppliesSnipThenPruneInOrder(t *testing.T) {
	sequence := []string{}
	svc := newOrderedPipelineService(t, &sequence, false)
	model := &orderedPipelineModel{sequence: &sequence}

	_, res, err := svc.CompactPipeline(context.Background(), model, orderedPipelineMessages(), PipelineRequest{
		Trigger:         "preflight",
		EstimatedTokens: 180,
		TargetWatermark: "summarize",
		AllowSummary:    false,
	})
	if err != nil {
		t.Fatalf("CompactPipeline: %v", err)
	}
	if model.calls != 0 {
		t.Fatalf("summary calls = %d, want 0 for local-only pipeline", model.calls)
	}
	if got, want := res.TiersApplied, []string{"snip", "prune"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("tiers = %#v, want %#v", got, want)
	}
}

func TestPipelineAppliesMicrocompactOnlyWhenEnabled(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("enabled=%t", enabled), func(t *testing.T) {
			sequence := []string{}
			svc := newOrderedPipelineService(t, &sequence, enabled)
			messages := []llm.Message{
				llm.NewUserMessage("old code\n```go\n" + strings.Repeat("fmt.Println(\"fixture\")\n", 180) + "```"),
				llm.NewUserMessage("latest protected"),
			}
			got, res, err := svc.CompactPipeline(context.Background(), &orderedPipelineModel{}, messages, PipelineRequest{
				Trigger:         "preflight",
				EstimatedTokens: 180,
				TargetWatermark: "prune",
				AllowSummary:    false,
			})
			if err != nil {
				t.Fatalf("CompactPipeline: %v", err)
			}
			if enabled {
				if !containsTier(res.TiersApplied, "microcompact") || got[0].Content.PlainText() == messages[0].Content.PlainText() {
					t.Fatalf("enabled microcompact result = %#v messages=%#v", res, got)
				}
				return
			}
			if containsTier(res.TiersApplied, "microcompact") || got[0].Content.PlainText() != messages[0].Content.PlainText() {
				t.Fatalf("disabled microcompact changed user content: %#v messages=%#v", res, got)
			}
		})
	}
}

func TestLocalReductionBelowSummaryTargetSkipsModelInvoke(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-local-enough")}
	svc := NewService(&Config{
		Enabled:        true,
		ContextWindow:  300,
		ThresholdRatio: 0.85,
		SessionID:      "sess-local-enough",
		LedgerStore:    store,
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
		}),
		ProtectedRecentMessages: 1,
	})
	model := &orderedPipelineModel{}

	_, res, err := svc.CompactPipeline(context.Background(), model, snipTestMessages(strings.Repeat("hit\n", 300)), PipelineRequest{
		Trigger:         "usage",
		Usage:           &llm.Usage{PromptTokens: 255, TotalTokens: 255},
		TargetWatermark: "summarize",
		AllowSummary:    true,
	})
	if err != nil {
		t.Fatalf("CompactPipeline: %v", err)
	}
	if model.calls != 0 {
		t.Fatalf("summary calls = %d, want 0 after sufficient local reduction", model.calls)
	}
	if !res.Compacted || !containsTier(res.TiersApplied, "snip") || containsTier(res.TiersApplied, "summarize") {
		t.Fatalf("result = %#v", res)
	}
}

func TestRuntimeCheckpointDefersLocalLedgerWrites(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-deferred-local")}
	svc := NewService(&Config{
		Enabled:          true,
		ContextWindow:    1000,
		ThresholdRatio:   0.85,
		SessionID:        "sess-deferred-local",
		LedgerStore:      store,
		CheckpointWriter: CompactionCheckpointWriterFunc(func(context.Context, CompactionCheckpoint) error { return nil }),
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
		}),
		ProtectedRecentMessages: 1,
	})
	messages := snipTestMessages(strings.Repeat("large tool result\n", 300))

	_, res, err := svc.CompactPipeline(context.Background(), nil, messages, PipelineRequest{
		Trigger:         "usage",
		Usage:           &llm.Usage{PromptTokens: 750, TotalTokens: 750},
		TargetWatermark: tierSnip,
		AllowSummary:    false,
	})
	if err != nil {
		t.Fatalf("CompactPipeline: %v", err)
	}
	if !res.Compacted || res.pendingLedger == nil {
		t.Fatalf("expected deferred local ledger transaction, got %#v", res)
	}
	if store.saves != 0 || len(store.ledger.Replacements) != 0 {
		t.Fatalf("local ledger escaped before runtime checkpoint: saves=%d ledger=%#v", store.saves, store.ledger)
	}
	if err := svc.CommitPendingLedger(context.Background(), &res); err != nil {
		t.Fatalf("CommitPendingLedger: %v", err)
	}
	if store.saves != 1 || len(store.ledger.Replacements) == 0 {
		t.Fatalf("deferred local ledger was not committed: saves=%d ledger=%#v", store.saves, store.ledger)
	}
	svc.FinalizePendingLedger(&res)
	if res.pendingLedger != nil || res.previousLedger != nil {
		t.Fatalf("pending ledger transaction not finalized: %#v", res)
	}
}

func TestDeferredLedgerTransactionSurvivesSummaryFailureForLocalFallback(t *testing.T) {
	sequence := []string{}
	svc := newOrderedPipelineService(t, &sequence, false)
	store := svc.Config.LedgerStore.(*memoryLedgerStore)
	svc.Config.CheckpointWriter = CompactionCheckpointWriterFunc(func(context.Context, CompactionCheckpoint) error { return nil })

	_, res, err := svc.CompactPipeline(context.Background(), failingPipelineSummaryModel{}, orderedPipelineMessages(), PipelineRequest{
		Trigger:         "overflow",
		EstimatedTokens: 180,
		TargetWatermark: "overflow",
		AllowSummary:    true,
		ForceSummary:    true,
	})
	if err == nil {
		t.Fatal("expected injected summary failure")
	}
	if !res.Compacted || res.pendingLedger == nil || res.previousLedger == nil {
		t.Fatalf("local fallback lost deferred ledger transaction: %#v", res)
	}
	if store.saves != 0 || len(store.ledger.Replacements) != 0 {
		t.Fatalf("fallback ledger escaped before runtime checkpoint: saves=%d ledger=%#v", store.saves, store.ledger)
	}
}

func TestMergedTelemetryPreservesActualLocalTiers(t *testing.T) {
	merged := mergePipelineResults(
		Result{Compacted: true, OriginalTokens: 900, NewTokens: 700, TiersApplied: []string{"snip"}},
		Result{Compacted: true, OriginalTokens: 700, NewTokens: 500, TiersApplied: []string{"prune", "microcompact"}},
		Result{Compacted: true, OriginalTokens: 500, NewTokens: 120, TiersApplied: []string{"summarize"}},
	)
	if got, want := merged.TiersApplied, []string{"snip", "prune", "microcompact", "summarize"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("tiers = %#v, want %#v", got, want)
	}
	if merged.OriginalTokens != 900 || merged.NewTokens != 120 {
		t.Fatalf("token telemetry = %#v", merged)
	}
}

func TestPipelineTelemetryKeepsProviderUsageSeparateFromComparableEstimates(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-token-units")}
	svc := NewService(&Config{
		Enabled:        true,
		ContextWindow:  1000,
		ThresholdRatio: 0.85,
		SessionID:      "sess-token-units",
		LedgerStore:    store,
		TokenEstimator: func(text string) int { return len(text) },
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
		}),
		ProtectedRecentMessages: 1,
	})
	messages := snipTestMessages(strings.Repeat("large provider-visible tool output\n", 120))
	providerUsage := &llm.Usage{
		PromptTokens:       750,
		TotalTokens:        750,
		PromptTokensValid:  true,
		PromptTokensSource: llm.PromptTokensSourceProvider,
	}
	wantBefore := svc.EstimateMessages(messages)

	got, res, err := svc.CompactPipeline(context.Background(), nil, messages, PipelineRequest{
		Trigger:         "usage",
		Usage:           providerUsage,
		TargetWatermark: tierSnip,
		AllowSummary:    false,
	})
	if err != nil {
		t.Fatalf("CompactPipeline: %v", err)
	}
	if !res.Compacted {
		t.Fatalf("result = %#v, want local compaction", res)
	}
	wantAfter := svc.EstimateMessages(got)
	if res.OriginalTokens != wantBefore || res.NewTokens != wantAfter {
		t.Fatalf("comparable estimate telemetry = %d -> %d, want %d -> %d", res.OriginalTokens, res.NewTokens, wantBefore, wantAfter)
	}
	if res.TokenCountSource != TokenCountSourceEstimate {
		t.Fatalf("token_count_source = %q, want %q", res.TokenCountSource, TokenCountSourceEstimate)
	}
	if res.Usage == nil || res.Usage.TotalTokens != providerUsage.TotalTokens || res.Usage.PromptTokensSource != llm.PromptTokensSourceProvider {
		t.Fatalf("trigger provider usage was not preserved separately: %#v", res.Usage)
	}
	if res.NewTokens >= res.OriginalTokens {
		t.Fatalf("local compaction did not reduce the comparable estimate: %d -> %d", res.OriginalTokens, res.NewTokens)
	}
}

func TestMergedTelemetryPreservesWarningsAndSnapshotPaths(t *testing.T) {
	merged := mergePipelineResults(
		Result{Compacted: true, SnapshotPath: "snap-1.md", LedgerPath: "ledger.json", Warnings: []string{"warn-1"}, TiersApplied: []string{"snip"}},
		Result{Compacted: true, SnapshotPath: "snap-2.md", Warnings: []string{"warn-2"}, TiersApplied: []string{"prune"}},
	)
	if merged.SnapshotPath != "snap-2.md" || merged.LedgerPath != "ledger.json" {
		t.Fatalf("paths = %#v", merged)
	}
	if got, want := merged.Warnings, []string{"warn-1", "warn-2"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("warnings = %#v, want %#v", got, want)
	}
}

func TestAutoPreflightManualAndOverflowSharePipelineDecisionTable(t *testing.T) {
	tests := []struct {
		name        string
		request     PipelineRequest
		wantSummary bool
	}{
		{name: "auto", request: PipelineRequest{Trigger: "usage", Usage: &llm.Usage{PromptTokens: 180, TotalTokens: 180}, TargetWatermark: "summarize", AllowSummary: false}},
		{name: "preflight", request: PipelineRequest{Trigger: "preflight", EstimatedTokens: 180, TargetWatermark: "summarize", AllowSummary: false}},
		{name: "overflow", request: PipelineRequest{Trigger: "overflow", Usage: &llm.Usage{PromptTokens: 180, TotalTokens: 180}, TargetWatermark: "overflow", AllowSummary: false}},
		{name: "manual", request: PipelineRequest{Trigger: "manual", TargetWatermark: "summarize", AllowSummary: true, ForceSummary: true}, wantSummary: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sequence := []string{}
			svc := newOrderedPipelineService(t, &sequence, false)
			model := &orderedPipelineModel{sequence: &sequence}
			_, res, err := svc.CompactPipeline(context.Background(), model, orderedPipelineMessages(), tt.request)
			if err != nil {
				t.Fatalf("CompactPipeline: %v", err)
			}
			if tt.wantSummary {
				if model.calls != 1 || !containsTier(res.TiersApplied, "summarize") {
					t.Fatalf("manual pipeline result = %#v calls=%d", res, model.calls)
				}
				return
			}
			if model.calls != 0 || !reflect.DeepEqual(res.TiersApplied, []string{"snip", "prune"}) {
				t.Fatalf("%s pipeline result = %#v calls=%d", tt.name, res, model.calls)
			}
		})
	}
}

func newOrderedPipelineService(t *testing.T, sequence *[]string, microcompact bool) *Service {
	t.Helper()
	store := &memoryLedgerStore{ledger: NewLedger("sess-pipeline")}
	return NewService(&Config{
		Enabled:                    true,
		ContextWindow:              200,
		ThresholdRatio:             0.85,
		SessionID:                  "sess-pipeline",
		LedgerStore:                store,
		EnableUserCodeMicrocompact: microcompact,
		ToolArtifactWriter: ArtifactWriterFunc(func(_ context.Context, req ArtifactRequest) (ArtifactResult, error) {
			name := strings.TrimSpace(req.ToolName)
			if name == "" {
				name = "unknown"
			}
			if sequence != nil {
				*sequence = append(*sequence, "artifact:"+name)
			}
			return ArtifactResult{Path: ".goode/truncated/" + name + ".txt"}, nil
		}),
		ProtectedRecentMessages: 1,
	})
}

func orderedPipelineMessages() []llm.Message {
	return []llm.Message{
		llm.NewUserMessage(strings.Repeat("user constraint path=/repo/project ", 100)),
		llm.NewAssistantMessage("calling grep", []llm.ToolCall{{ID: "call-grep", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage("call-grep", "grep", llm.TextContent(strings.Repeat("hit\n", 300)), false),
		llm.NewAssistantMessage(strings.Repeat("analysis path=/repo/project error detail ", 120), nil),
		llm.NewUserMessage("latest protected"),
	}
}
