package agent

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// summaryCountingModel answers every call with a valid summary and counts
// the summary requests (those carrying the summary prompt).
type summaryCountingModel struct {
	mu    sync.Mutex
	calls int
}

func (m *summaryCountingModel) Provider() string { return "fixture" }
func (m *summaryCountingModel) Model() string    { return "summary-counting" }
func (m *summaryCountingModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	summary := false
	for _, msg := range req.Messages {
		if strings.Contains(msg.Content.PlainText(), "Write an operational checkpoint") {
			summary = true
		}
	}
	m.mu.Lock()
	if summary {
		m.calls++
	}
	m.mu.Unlock()
	return &llm.Completion{Content: llm.TextContent(validCompactionSummary("kept constraints"))}, nil
}

func (m *summaryCountingModel) Calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

type ineffectiveSummaryFixture struct {
	agent                *Agent
	model                *summaryCountingModel
	high, low, overflowU *llm.Usage
}

// newIneffectiveSummaryFixture builds an Agent whose newest user message,
// which every summary keeps verbatim, is alone above the summary threshold:
// a successful summary cannot bring the history back under it.
func newIneffectiveSummaryFixture(t *testing.T) ineffectiveSummaryFixture {
	t.Helper()
	kept := llm.NewUserMessage(strings.Repeat("constraint ", 3000))
	original := []llm.Message{llm.NewSystemMessage("base"), llm.NewUserMessage("earlier request"), llm.NewAssistantMessage("earlier answer", nil), kept}
	probe := compaction.NewService(&compaction.Config{Enabled: true})
	estimate := probe.EstimateMessages([]llm.Message{kept})
	model := &summaryCountingModel{}
	ag, err := New(Config{LLM: model, InitialMessages: original, Compaction: &compaction.Config{
		Enabled: true, ContextWindow: 4 * estimate, ReserveOutputTokens: 1,
		ThresholdRatio: 0.2, SnipThresholdRatio: 0.1, PruneThresholdRatio: 0.15,
	}})
	if err != nil {
		t.Fatal(err)
	}
	threshold, limit := ag.compactor.ThresholdTokens(), ag.compactor.OverflowLimit()
	if !(threshold < estimate && 2*estimate < limit) {
		t.Fatalf("fixture sizing: threshold=%d kept=%d limit=%d", threshold, estimate, limit)
	}
	usage := func(tokens int) *llm.Usage {
		return &llm.Usage{PromptTokens: tokens, TotalTokens: tokens}
	}
	return ineffectiveSummaryFixture{
		agent:     ag,
		model:     model,
		high:      usage(2 * estimate),
		low:       usage(threshold / 4),
		overflowU: usage(limit + 1),
	}
}

// decide runs one automatic compaction decision through the Agent entry the
// Query loop uses, waits for the asynchronous run on the idle signal, and
// publishes its result.
func (f ineffectiveSummaryFixture) decide(t *testing.T, usage *llm.Usage) {
	t.Helper()
	ctx := context.Background()
	if err := f.agent.checkAndCompactWithGrowth(ctx, "", &llm.Completion{Usage: usage}, nil, 0, 0); err != nil {
		t.Fatalf("automatic compaction: %v", err)
	}
	if err := f.agent.waitForCompactionIdle(ctx, nil); err != nil {
		t.Fatal(err)
	}
	f.agent.applyPendingCompaction(nil)
}

// A successful automatic summary that leaves the history above the summary
// threshold is not repeated for the same real user input; local tiers and the
// overflow boundary are unaffected.
func TestIneffectiveAutomaticSummaryIsNotRepeatedWithinUserInput(t *testing.T) {
	f := newIneffectiveSummaryFixture(t)
	f.decide(t, f.high)
	if got := f.model.Calls(); got != 1 {
		t.Fatalf("first automatic summary calls = %d, want 1", got)
	}
	for i := 0; i < 3; i++ {
		f.decide(t, f.high)
	}
	if got := f.model.Calls(); got != 1 {
		t.Fatalf("ineffective automatic summary repeated: calls = %d, want 1", got)
	}

	// Overflow is a hard boundary and still runs its own summary plan; whether
	// the irreducible history then fits is not what this test measures.
	_ = f.agent.checkAndCompactWithGrowth(context.Background(), "", &llm.Completion{Usage: f.overflowU}, nil, 0, 0)
	if got := f.model.Calls(); got != 2 {
		t.Fatalf("overflow summary calls = %d, want 2", got)
	}
}

// Suppression lasts only while its evidence holds: a new real user input or
// a drop below the summary threshold rearms the automatic summary, and a
// replacement compaction runtime starts unsuppressed.
func TestIneffectiveSummarySuppressionRearms(t *testing.T) {
	t.Run("new user input", func(t *testing.T) {
		f := newIneffectiveSummaryFixture(t)
		f.decide(t, f.high)
		f.decide(t, f.high)
		f.agent.userInputEpoch.Add(1)
		f.decide(t, f.high)
		if got := f.model.Calls(); got != 2 {
			t.Fatalf("calls = %d, want 2 (one per user input)", got)
		}
	})
	t.Run("low watermark", func(t *testing.T) {
		f := newIneffectiveSummaryFixture(t)
		f.decide(t, f.high)
		f.decide(t, f.low)
		f.decide(t, f.high)
		if got := f.model.Calls(); got != 2 {
			t.Fatalf("calls = %d, want 2 (rearmed below the threshold)", got)
		}
	})
	t.Run("runtime replacement", func(t *testing.T) {
		f := newIneffectiveSummaryFixture(t)
		f.decide(t, f.high)
		f.agent.resetCompactionOutcomeState()
		f.decide(t, f.high)
		if got := f.model.Calls(); got != 2 {
			t.Fatalf("calls = %d, want 2 (new runtime is not suppressed)", got)
		}
	})
	t.Run("manual compaction is not suppressed", func(t *testing.T) {
		f := newIneffectiveSummaryFixture(t)
		f.decide(t, f.high)
		if _, err := f.agent.CompactNow(context.Background()); err != nil {
			t.Fatalf("manual compaction: %v", err)
		}
		if got := f.model.Calls(); got != 2 {
			t.Fatalf("calls = %d, want 2 (manual runs as requested)", got)
		}
	})
}

// An effective automatic summary never arms suppression.
func TestEffectiveAutomaticSummaryKeepsSummaryTier(t *testing.T) {
	model := &summaryCountingModel{}
	var history []llm.Message
	history = append(history, llm.NewSystemMessage("base"))
	for i := 0; i < 6; i++ {
		history = append(history, llm.NewUserMessage(strings.Repeat("old ", 600)), llm.NewAssistantMessage(strings.Repeat("reply ", 600), nil))
	}
	history = append(history, llm.NewUserMessage("latest"))
	probe := compaction.NewService(&compaction.Config{Enabled: true})
	total := probe.EstimateMessages(history)
	ag, err := New(Config{LLM: model, InitialMessages: history, Compaction: &compaction.Config{
		Enabled: true, ContextWindow: 2 * total, ReserveOutputTokens: 1, ThresholdRatio: 0.4,
		SnipThresholdRatio: 0.3, PruneThresholdRatio: 0.35, KeepRecentUserMessages: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	f := ineffectiveSummaryFixture{agent: ag, model: model}
	high := &llm.Usage{PromptTokens: total, TotalTokens: total}
	f.decide(t, high)
	if got := model.Calls(); got != 1 {
		t.Fatalf("first summary calls = %d, want 1", got)
	}
	if ag.ineffectiveSummaryEpoch.Load() != 0 {
		t.Fatal("effective summary armed suppression")
	}
	f.decide(t, high)
	if got := model.Calls(); got != 2 {
		t.Fatalf("summary tier suppressed after an effective summary: calls = %d", got)
	}
}

// Each Query is a new real user input: the suppression armed by an earlier
// input does not carry into it. The Query's own post-completion decision may
// summarize once in the new input, and no more.
func TestIneffectiveSummarySuppressionEndsWithQuery(t *testing.T) {
	f := newIneffectiveSummaryFixture(t)
	f.decide(t, f.high)
	f.decide(t, f.high)
	if got := f.model.Calls(); got != 1 {
		t.Fatalf("summaries before Query = %d, want 1", got)
	}
	if _, err := f.agent.Query(context.Background(), "next request"); err != nil {
		t.Fatalf("Query: %v", err)
	}
	f.decide(t, f.high)
	f.decide(t, f.high)
	if got := f.model.Calls(); got != 2 {
		t.Fatalf("summaries across the new Query = %d, want 2 (one per user input)", got)
	}
}
