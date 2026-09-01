package agent

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type localCompactionCountingModel struct {
	calls atomic.Int32
}

func (m *localCompactionCountingModel) Provider() string { return "mock" }
func (m *localCompactionCountingModel) Model() string    { return "mock" }
func (m *localCompactionCountingModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls.Add(1)
	return &llm.Completion{Content: llm.TextContent(validCompactionSummary("unexpected summary"))}, nil
}

type overflowCompactionCountingModel struct {
	calls atomic.Int32
}

func (m *overflowCompactionCountingModel) Provider() string { return "mock" }
func (m *overflowCompactionCountingModel) Model() string    { return "mock" }
func (m *overflowCompactionCountingModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls.Add(1)
	return &llm.Completion{Content: llm.TextContent(validCompactionSummary("overflow summary"))}, nil
}

type providerBoundaryModel struct {
	called chan struct{}
}

func (m *providerBoundaryModel) Provider() string { return "mock" }
func (m *providerBoundaryModel) Model() string    { return "mock" }
func (m *providerBoundaryModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	select {
	case <-m.called:
	default:
		close(m.called)
	}
	return &llm.Completion{Content: llm.TextContent("done")}, nil
}

func TestCheckAndCompactUsesLocalSnipWithoutModelInvoke(t *testing.T) {
	model := &localCompactionCountingModel{}
	store := &agentLocalLedgerStore{ledger: compaction.NewLedger("sess-agent-local")}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  100,
			ThresholdRatio: 0.85,
			SessionID:      "sess-agent-local",
			LedgerStore:    store,
			ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
				return compaction.ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
			}),
			ProtectedRecentMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{
		llm.NewUserMessage("search"),
		llm.NewAssistantMessage("calling grep", []llm.ToolCall{{ID: "call-grep", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage("call-grep", "grep", llm.TextContent(strings.Repeat("hit\n", 300)), false),
		llm.NewUserMessage("latest"),
	})

	ag.checkAndCompact(context.Background(), &llm.Completion{Usage: &llm.Usage{PromptTokens: 70, TotalTokens: 70}}, nil)
	waitFor(t, time.Second, func() bool {
		return !ag.compactionInFlight.Load() && ag.hasPendingCompaction()
	}, "local snip pending compaction")
	if got := model.calls.Load(); got != 0 {
		t.Fatalf("compaction model calls = %d, want 0 for local snip", got)
	}

	ag.applyPendingCompaction(nil)
	msgs := ag.Messages()
	if len(msgs) < 3 {
		t.Fatalf("messages after compaction = %#v", msgs)
	}
	if got := msgs[2].Content.PlainText(); !strings.Contains(got, "[Tool result snipped:") {
		t.Fatalf("tool result was not snipped: %q", got)
	}
}

func TestOverflowStillCompactsSynchronouslyBeforeProviderCall(t *testing.T) {
	model := &overflowCompactionCountingModel{}
	store := &agentLocalLedgerStore{ledger: compaction.NewLedger("sess-agent-overflow")}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  100,
			ThresholdRatio: 0.85,
			SessionID:      "sess-agent-overflow",
			LedgerStore:    store,
			ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
				return compaction.ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
			}),
			ProtectedRecentMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{
		llm.NewUserMessage("search"),
		llm.NewAssistantMessage("calling grep", []llm.ToolCall{{ID: "call-grep", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage("call-grep", "grep", llm.TextContent(strings.Repeat("hit\n", 300)), false),
		llm.NewUserMessage("latest"),
	})

	ag.checkAndCompact(context.Background(), &llm.Completion{Usage: &llm.Usage{PromptTokens: 100, TotalTokens: 100}}, nil)
	if got := model.calls.Load(); got != 0 {
		t.Fatalf("compaction model calls = %d, want 0 when synchronous local reduction resolves overflow", got)
	}
	if ag.compactionInFlight.Load() || ag.hasPendingCompaction() {
		t.Fatal("overflow compaction should apply synchronously")
	}
	msgs := ag.Messages()
	if len(msgs) < 3 || !strings.Contains(msgs[2].Content.PlainText(), "[Tool result snipped:") {
		t.Fatalf("messages after overflow compaction = %#v", msgs)
	}
}

func TestCheckAndCompactCountsPostCompletionToolGrowth(t *testing.T) {
	model := &localCompactionCountingModel{}
	store := &agentLocalLedgerStore{ledger: compaction.NewLedger("sess-agent-growth")}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:             true,
			ContextWindow:       200,
			ReserveOutputTokens: 50,
			SessionID:           "sess-agent-growth",
			LedgerStore:         store,
			ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
				return compaction.ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
			}),
			ProtectedRecentMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{
		llm.NewUserMessage("search"),
		llm.NewAssistantMessage("calling grep", []llm.ToolCall{{ID: "call-grep", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage("call-grep", "grep", llm.TextContent(strings.Repeat("hit\n", 300)), false),
		llm.NewUserMessage("latest"),
	})

	// 90 is below Tier 1 for the 150-token usable prompt window (105), but
	// 20 tokens appended after the completion move the next request to 110.
	ag.checkAndCompact(context.Background(), &llm.Completion{Usage: llm.NewProviderUsage(85, 5, 90)}, nil, 20)
	waitFor(t, time.Second, func() bool {
		return !ag.compactionInFlight.Load() && ag.hasPendingCompaction()
	}, "post-completion growth compaction")
	if got := model.calls.Load(); got != 0 {
		t.Fatalf("compaction model calls = %d, want 0 for Tier 1 growth compaction", got)
	}
	ag.applyPendingCompaction(nil)
	if got := ag.Messages()[2].Content.PlainText(); !strings.Contains(got, "[Tool result snipped:") {
		t.Fatalf("tool growth did not trigger local snip: %q", got)
	}
}

func TestEffectiveCompactionUsageDoesNotDoubleCountCurrentHistoryEstimate(t *testing.T) {
	ag, err := New(Config{
		LLM: &localCompactionCountingModel{},
		Compaction: &compaction.Config{
			Enabled:       true,
			ContextWindow: 1000,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{
		llm.NewUserMessage("request"),
		llm.NewAssistantMessage("answer", nil),
		llm.NewToolMessage("call-1", "read", llm.TextContent(strings.Repeat("result ", 40)), false),
	})

	want := ag.compactor.EstimateMessages(ag.Messages())
	got := ag.effectiveCompactionUsage(nil, 50)
	if got == nil {
		t.Fatal("effectiveCompactionUsage returned nil")
	}
	if got.PromptTokens != want || got.TotalTokens != want {
		t.Fatalf("missing-usage estimate = prompt:%d total:%d, want current history %d without adding the delta twice", got.PromptTokens, got.TotalTokens, want)
	}
	pending := ag.effectiveCompactionUsageWithGrowth(nil, 50, 20)
	if pending.PromptTokens != want+20 || pending.TotalTokens != want+20 {
		t.Fatalf("pending history growth = prompt:%d total:%d, want %d", pending.PromptTokens, pending.TotalTokens, want+20)
	}
}

func TestNoOpAsyncCompactionDoesNotQueueSuccess(t *testing.T) {
	ag, err := New(Config{
		LLM: &localCompactionCountingModel{},
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  100,
			ThresholdRatio: 0.85,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("latest protected request")})

	ag.checkAndCompact(context.Background(), &llm.Completion{Usage: &llm.Usage{PromptTokens: 70, TotalTokens: 70}}, nil)
	waitFor(t, time.Second, func() bool { return !ag.compactionInFlight.Load() }, "no-op local compaction")
	if ag.hasPendingCompaction() {
		t.Fatal("no-op compaction must not queue a success event or history replacement")
	}
	if got := ag.compactionGeneration.Load(); got != 0 {
		t.Fatalf("no-op compaction generation = %d, want 0", got)
	}
}

func TestQueryWaitsForInFlightCompactionBeforeProviderBoundary(t *testing.T) {
	model := &providerBoundaryModel{called: make(chan struct{})}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:       true,
			ContextWindow: 1000,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.compactionInFlight.Store(true)
	events := ag.QueryStream(context.Background(), llm.TextContent("continue"))

	select {
	case <-model.called:
		t.Fatal("provider invoked before in-flight compaction reached the boundary")
	case <-time.After(30 * time.Millisecond):
	}
	ag.compactionInFlight.Store(false)
	for range events {
	}
	select {
	case <-model.called:
	case <-time.After(time.Second):
		t.Fatal("provider was not invoked after compaction boundary released")
	}
}

func TestTodoCompletionBelowMinimumThresholdDoesNotInvokeSummaryModel(t *testing.T) {
	model := &localCompactionCountingModel{}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  1_000_000,
			ThresholdRatio: 0.85,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("keep the current task state")})
	ag.NotifyTodoCompletion()

	ag.checkAndCompact(context.Background(), &llm.Completion{Usage: &llm.Usage{
		PromptTokens: 40_000,
		TotalTokens:  40_000,
	}}, nil)
	waitFor(t, time.Second, func() bool { return !ag.compactionInFlight.Load() }, "low-watermark todo check")

	if got := model.calls.Load(); got != 0 {
		t.Fatalf("compaction model calls = %d, want 0 below Tier 1", got)
	}
	if ag.hasPendingCompaction() {
		t.Fatal("low-watermark todo checkpoint must not queue a summary result")
	}
	if !ag.todoCompactionPending.Load() {
		t.Fatal("todo checkpoint should remain pending until a normal watermark is eligible")
	}
}

func TestTodoCheckpointAtEligibleWatermarkUsesNormalPipeline(t *testing.T) {
	model := &localCompactionCountingModel{}
	store := &agentLocalLedgerStore{ledger: compaction.NewLedger("sess-todo-checkpoint")}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  100,
			ThresholdRatio: 0.85,
			SessionID:      "sess-todo-checkpoint",
			LedgerStore:    store,
			ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
				return compaction.ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
			}),
			ProtectedRecentMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{
		llm.NewUserMessage("search"),
		llm.NewAssistantMessage("calling grep", []llm.ToolCall{{ID: "call-grep", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage("call-grep", "grep", llm.TextContent(strings.Repeat("hit\n", 300)), false),
		llm.NewUserMessage("latest"),
	})
	ag.NotifyTodoCompletion()

	ag.checkAndCompact(context.Background(), &llm.Completion{Usage: &llm.Usage{PromptTokens: 70, TotalTokens: 70}}, nil)
	waitFor(t, time.Second, func() bool {
		return !ag.compactionInFlight.Load() && ag.hasPendingCompaction()
	}, "todo checkpoint local compaction")
	if got := model.calls.Load(); got != 0 {
		t.Fatalf("compaction model calls = %d, want 0 for eligible local tier", got)
	}

	out := make(chan Event, 1)
	ag.applyPendingCompaction(wrapLegacyEventOutput(out))
	close(out)
	var got CompactionEvent
	for event := range out {
		if compacted, ok := event.(CompactionEvent); ok {
			got = compacted
		}
	}
	if got.Result.Trigger != "todo_checkpoint" || got.Result.Watermark != "snip" {
		t.Fatalf("compaction trigger/watermark = %q/%q, want todo_checkpoint/snip", got.Result.Trigger, got.Result.Watermark)
	}
	if got.Result.Usage == nil || got.Result.Usage.PromptTokens != 70 {
		t.Fatalf("todo checkpoint usage = %#v, want prompt_tokens=70", got.Result.Usage)
	}
}

func TestPlaceholderPressureUsesLocalDeterministicCleanup(t *testing.T) {
	model := &localCompactionCountingModel{}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  1_000_000,
			ThresholdRatio: 0.85,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory(destroyedPlaceholderHistory(defaultDestroyedToolCompactThreshold))

	ag.checkAndCompact(context.Background(), &llm.Completion{Usage: &llm.Usage{
		PromptTokens: 40_000,
		TotalTokens:  40_000,
	}}, nil)
	waitFor(t, time.Second, func() bool {
		return !ag.compactionInFlight.Load() && ag.hasPendingCompaction()
	}, "placeholder cleanup")
	if got := model.calls.Load(); got != 0 {
		t.Fatalf("compaction model calls = %d, want 0 for placeholder cleanup", got)
	}

	out := make(chan Event, 1)
	ag.applyPendingCompaction(wrapLegacyEventOutput(out))
	close(out)
	var compacted CompactionEvent
	for event := range out {
		if got, ok := event.(CompactionEvent); ok {
			compacted = got
		}
	}
	if compacted.Result.Trigger != "placeholder_pressure" || compacted.Result.Watermark != "placeholder_cleanup" {
		t.Fatalf("compaction trigger/watermark = %q/%q", compacted.Result.Trigger, compacted.Result.Watermark)
	}
	if len(compacted.Result.TiersApplied) != 1 || compacted.Result.TiersApplied[0] != "placeholder_cleanup" {
		t.Fatalf("placeholder tiers = %#v", compacted.Result.TiersApplied)
	}
	for i, msg := range ag.Messages() {
		if msg.Destroyed {
			t.Fatalf("destroyed placeholder survived cleanup at message %d: %#v", i, msg)
		}
	}
}

func TestPlaceholderPressureWithEstimatedUsageRemainsEligible(t *testing.T) {
	model := &localCompactionCountingModel{}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  1_000_000,
			ThresholdRatio: 0.85,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory(destroyedPlaceholderHistory(defaultDestroyedToolCompactThreshold))
	last := &llm.Completion{Usage: llm.NewProviderUsage(0, 2, 2)}

	if !ag.shouldAttemptCompaction(context.Background(), last) {
		t.Fatal("estimated prompt usage should keep placeholder cleanup eligible")
	}
	ag.checkAndCompact(context.Background(), last, nil)
	waitFor(t, time.Second, func() bool {
		return !ag.compactionInFlight.Load() && ag.hasPendingCompaction()
	}, "estimated placeholder cleanup")
	if got := model.calls.Load(); got != 0 {
		t.Fatalf("compaction model calls = %d, want 0 with estimated usage", got)
	}
}

func TestManualCompactionCanStillRequestSummaryExplicitly(t *testing.T) {
	model := &localCompactionCountingModel{}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:                true,
			ContextWindow:          1_000_000,
			ThresholdRatio:         0.85,
			KeepRecentUserMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("manual checkpoint")})

	res, err := ag.CompactNow(context.Background())
	if err != nil {
		t.Fatalf("CompactNow: %v", err)
	}
	if got := model.calls.Load(); got != 1 {
		t.Fatalf("compaction model calls = %d, want 1 for explicit manual summary", got)
	}
	if res.Trigger != "manual" || res.Watermark != "summarize" {
		t.Fatalf("manual result = %#v", res)
	}
}

func destroyedPlaceholderHistory(count int) []llm.Message {
	messages := []llm.Message{llm.NewUserMessage("inspect the fixture")}
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("destroyed-%d", i)
		messages = append(messages,
			llm.NewAssistantMessage("", []llm.ToolCall{{ID: id, Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}}),
			llm.Message{Role: llm.RoleTool, ToolCallID: id, ToolName: "read", Ephemeral: true, Destroyed: true, Content: llm.TextContent(ephemeralReleasedPlaceholder)},
		)
	}
	messages = append(messages, llm.NewUserMessage("latest protected request"))
	return messages
}

func TestCheckAndCompactOverflowWaitsForInFlightThenSummarizes(t *testing.T) {
	model := &overflowCompactionCountingModel{}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:                true,
			ContextWindow:          100,
			ThresholdRatio:         0.85,
			KeepRecentUserMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("hello")})
	ag.compactionInFlight.Store(true)
	go func() {
		time.Sleep(30 * time.Millisecond)
		ag.compactionInFlight.Store(false)
	}()

	ag.checkAndCompact(context.Background(), &llm.Completion{Usage: &llm.Usage{PromptTokens: 100, TotalTokens: 100}}, nil)
	if got := model.calls.Load(); got != 1 {
		t.Fatalf("compaction model calls = %d, want 1 after waiting for in-flight compaction", got)
	}
	if ag.compactionInFlight.Load() || ag.hasPendingCompaction() {
		t.Fatal("overflow compaction should apply synchronously after in-flight wait")
	}
}

func TestCompactLocalNowAppliesWithoutModelInvoke(t *testing.T) {
	model := &localCompactionCountingModel{}
	store := &agentLocalLedgerStore{ledger: compaction.NewLedger("sess-agent-local-now")}
	checkpointWrites := 0
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  100,
			ThresholdRatio: 0.85,
			SessionID:      "sess-agent-local-now",
			LedgerStore:    store,
			ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
				return compaction.ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
			}),
			CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
				checkpointWrites++
				return nil
			}),
			ProtectedRecentMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{
		llm.NewUserMessage("search"),
		llm.NewAssistantMessage("calling grep", []llm.ToolCall{{ID: "call-grep", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage("call-grep", "grep", llm.TextContent(strings.Repeat("hit\n", 300)), false),
		llm.NewUserMessage("latest"),
	})

	res, err := ag.CompactLocalNow(context.Background(), 90)
	if err != nil {
		t.Fatalf("CompactLocalNow: %v", err)
	}
	if got := model.calls.Load(); got != 0 {
		t.Fatalf("compaction model calls = %d, want 0", got)
	}
	if !res.Compacted || res.Watermark != "snip" {
		t.Fatalf("result = %#v, want local snip after re-estimation", res)
	}
	msgs := ag.Messages()
	if got := msgs[2].Content.PlainText(); !strings.Contains(got, "[Tool result snipped:") {
		t.Fatalf("tool result was not snipped: %q", got)
	}
	if checkpointWrites != 1 {
		t.Fatalf("checkpoint writes = %d, want 1", checkpointWrites)
	}

	res, err = ag.CompactLocalNow(context.Background(), 90)
	if err != nil {
		t.Fatalf("CompactLocalNow prune upgrade: %v", err)
	}
	if !res.Compacted || res.Watermark != "prune" {
		t.Fatalf("higher watermark did not upgrade snip to prune exactly once: %#v", res)
	}
	if checkpointWrites != 2 {
		t.Fatalf("checkpoint writes after prune upgrade = %d, want 2", checkpointWrites)
	}

	res, err = ag.CompactLocalNow(context.Background(), 90)
	if err != nil {
		t.Fatalf("CompactLocalNow already-pruned history: %v", err)
	}
	if res.Compacted {
		t.Fatalf("already-pruned history reported another compaction: %#v", res)
	}
	if checkpointWrites != 2 {
		t.Fatalf("no-op compaction wrote another checkpoint: %d", checkpointWrites)
	}
}

// TestShouldAttemptCompactionTriggersOnPlaceholderPressure verifies that a
// large accumulation of recycled (destroyed) tool results triggers compaction
// even when the token watermark has not been reached — the placeholder count is
// itself the signal that context is full of zero-information messages.
func TestShouldAttemptCompactionTriggersOnPlaceholderPressure(t *testing.T) {
	store := &agentLocalLedgerStore{ledger: compaction.NewLedger("sess-placeholder")}
	ag, err := New(Config{
		LLM: &localCompactionCountingModel{},
		Compaction: &compaction.Config{
			Enabled:                 true,
			ContextWindow:           1_000_000, // huge, so small usage never hits the watermark
			ThresholdRatio:          0.85,
			SessionID:               "sess-placeholder",
			LedgerStore:             store,
			ProtectedRecentMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// A compatible provider may report completion usage while leaving prompt
	// tokens at zero. Placeholder pressure must fall back to local history
	// estimation instead of disabling recovery.
	last := &llm.Completion{Usage: llm.NewProviderUsage(0, 2, 2)}

	// Below threshold: no compaction from placeholder pressure.
	msgs := []llm.Message{llm.NewUserMessage("hi")}
	for i := 0; i < defaultDestroyedToolCompactThreshold-1; i++ {
		msgs = append(msgs, llm.Message{Role: llm.RoleTool, ToolName: "read", Ephemeral: true, Destroyed: true, Content: llm.TextContent(ephemeralReleasedPlaceholder)})
	}
	ag.ReplaceHistory(msgs)
	if ag.shouldAttemptCompaction(context.Background(), last) {
		t.Fatalf("did not expect compaction below destroyed-placeholder threshold")
	}

	// At threshold: placeholder pressure triggers compaction.
	msgs = append(msgs, llm.Message{Role: llm.RoleTool, ToolName: "read", Ephemeral: true, Destroyed: true, Content: llm.TextContent(ephemeralReleasedPlaceholder)})
	ag.ReplaceHistory(msgs)
	if !ag.shouldAttemptCompaction(context.Background(), last) {
		t.Fatalf("expected compaction to trigger at destroyed-placeholder threshold")
	}
	if trigger, _ := ag.compactionTriggerAndWatermark(last); trigger != "placeholder_pressure" {
		t.Fatalf("expected placeholder_pressure trigger, got %q", trigger)
	}
}

type agentLocalLedgerStore struct {
	ledger *compaction.Ledger
}

func (s *agentLocalLedgerStore) Load(context.Context, string) (*compaction.Ledger, error) {
	if s.ledger == nil {
		return compaction.NewLedger(""), nil
	}
	return s.ledger.Clone(), nil
}

func (s *agentLocalLedgerStore) Save(_ context.Context, _ string, ledger *compaction.Ledger) error {
	s.ledger = ledger.Clone()
	return nil
}
