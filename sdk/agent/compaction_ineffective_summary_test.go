package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
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
// The summary is rejected as not smaller than this short history (#257), which
// counts as ineffective too.
func newIneffectiveSummaryFixture(t *testing.T) ineffectiveSummaryFixture {
	t.Helper()
	return newIneffectiveSummaryFixtureWith(t, "earlier answer")
}

// newIneffectiveSummaryFixtureWith uses earlier as the summarized assistant
// answer; a long one makes the summary accepted yet still ineffective.
func newIneffectiveSummaryFixtureWith(t *testing.T, earlier string) ineffectiveSummaryFixture {
	t.Helper()
	kept := llm.NewUserMessage(strings.Repeat("constraint ", 3000))
	original := []llm.Message{llm.NewSystemMessage("base"), llm.NewUserMessage("earlier request"), llm.NewAssistantMessage(earlier, nil), kept}
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
	for name, earlier := range map[string]string{
		"summary_not_smaller": "earlier answer",
		"summary_accepted":    strings.Repeat("earlier answer detail ", 400),
	} {
		t.Run(name, func(t *testing.T) { testIneffectiveAutomaticSummaryIsNotRepeated(t, earlier) })
	}
}

func testIneffectiveAutomaticSummaryIsNotRepeated(t *testing.T, earlier string) {
	f := newIneffectiveSummaryFixtureWith(t, earlier)
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

// G4: suppression identity. The suppression belongs to the compaction
// runtime that armed it and to the user input it was armed in. Installing a
// runtime through the public UpdateCompactionConfig rearms the automatic
// summary: a changed context window, and equally an identical
// configuration, since the SDK treats every replacement as a new identity and
// does not compare configurations. A replacement queued while the old runtime
// is still in use takes effect, and rearms, only when that use ends; until
// then the old runtime and its suppression stay coherent.
func TestIneffectiveSummarySuppressionFollowsConfiguredRuntimeIdentity(t *testing.T) {
	config := func(f ineffectiveSummaryFixture, windowFactor int) *compaction.Config {
		estimate := f.high.PromptTokens / 2
		return &compaction.Config{
			Enabled: true, ContextWindow: windowFactor * estimate, ReserveOutputTokens: 1,
			ThresholdRatio: 0.2, SnipThresholdRatio: 0.1, PruneThresholdRatio: 0.15,
		}
	}
	armed := func(t *testing.T) ineffectiveSummaryFixture {
		t.Helper()
		f := newIneffectiveSummaryFixture(t)
		f.decide(t, f.high)
		f.decide(t, f.high)
		if got := f.model.Calls(); got != 1 || f.agent.ineffectiveSummaryEpoch.Load() == 0 {
			t.Fatalf("fixture not suppressed: calls=%d", got)
		}
		return f
	}
	t.Run("context window change", func(t *testing.T) {
		f := armed(t)
		oldThreshold := f.agent.compactor.ThresholdTokens()
		f.agent.UpdateCompactionConfig(config(f, 5))
		if f.agent.compactor.ThresholdTokens() == oldThreshold {
			t.Fatal("replacement did not change the window")
		}
		if f.agent.compactor.DecisionTokens(f.high) < f.agent.compactor.ThresholdTokens() {
			t.Fatal("fixture sizing: the new window must still need a summary")
		}
		f.decide(t, f.high)
		if got := f.model.Calls(); got != 2 {
			t.Fatalf("calls = %d, want 2 (new window rearms the summary)", got)
		}
		// The new runtime arms its own suppression for the same material.
		f.decide(t, f.high)
		if got := f.model.Calls(); got != 2 {
			t.Fatalf("calls = %d, want 2 (new runtime suppresses its own ineffective summary)", got)
		}
	})
	t.Run("identical configuration", func(t *testing.T) {
		f := armed(t)
		f.agent.UpdateCompactionConfig(config(f, 4))
		f.decide(t, f.high)
		if got := f.model.Calls(); got != 2 {
			t.Fatalf("calls = %d, want 2 (any replacement is a new identity)", got)
		}
	})
	t.Run("replacement queued during use", func(t *testing.T) {
		f := armed(t)
		release, ok := f.agent.tryBeginCompactionRuntimeUse()
		if !ok {
			t.Fatal("runtime use not acquired")
		}
		f.agent.UpdateCompactionConfig(config(f, 5))
		f.decide(t, f.high)
		if got := f.model.Calls(); got != 1 {
			t.Fatalf("calls = %d, want 1 (the old runtime is still in use and suppressed)", got)
		}
		release()
		f.decide(t, f.high)
		if got := f.model.Calls(); got != 2 {
			t.Fatalf("calls = %d, want 2 (the queued runtime rearmed on installation)", got)
		}
	})
}

// fixedOverheadModel reports a provider prompt of prompt(step) tokens for each
// agent request, asks for one probe call per step for six steps, and counts
// summary requests.
type fixedOverheadModel struct {
	mu               sync.Mutex
	prompt           func(step int) int
	steps, summaries int
}

func (*fixedOverheadModel) Provider() string { return "fixture" }
func (*fixedOverheadModel) Model() string    { return "fixed-overhead" }

func (m *fixedOverheadModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, msg := range req.Messages {
		if strings.Contains(msg.Content.PlainText(), "Write an operational checkpoint") {
			m.summaries++
			return &llm.Completion{Content: llm.TextContent(validCompactionSummary("probe calls so far")), StopReason: "stop"}, nil
		}
	}
	m.steps++
	prompt := m.prompt(m.steps)
	usage := &llm.Usage{PromptTokens: prompt, CompletionTokens: 50, TotalTokens: prompt + 50}
	if m.steps%7 == 0 {
		return &llm.Completion{Content: llm.TextContent("finished"), StopReason: "stop", Usage: usage}, nil
	}
	call := llm.ToolCall{ID: fmt.Sprintf("probe-%d", m.steps), Type: "function", Function: llm.FunctionCall{Name: "probe", Arguments: `{}`}}
	return &llm.Completion{ToolCalls: []llm.ToolCall{call}, StopReason: "tool_calls", Usage: usage}, nil
}

func (m *fixedOverheadModel) Summaries() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.summaries
}

func newFixedOverheadAgent(t *testing.T, model *fixedOverheadModel) *Agent {
	t.Helper()
	cc := compaction.DefaultConfig()
	cc.ContextWindow = 12000
	probe := tools.Tool{
		Name:        "probe",
		Description: "probe",
		Schema:      map[string]any{"type": "object", "properties": map[string]any{}},
		Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
			return llm.TextContent(strings.Repeat("probe output line\n", 300)), nil
		},
	}
	ag, err := New(Config{
		LLM: model,
		InitialMessages: []llm.Message{
			llm.NewSystemMessage("system"),
			llm.NewUserMessage("earlier request"),
			llm.NewAssistantMessage(strings.Repeat("earlier notes ", 600), nil),
		},
		Tools:                  []tools.Tool{probe},
		Compaction:             &cc,
		MaxIterations:          20,
		InvokeRetryMaxAttempts: 1,
		Warningf:               func(string, ...any) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	return ag
}

func runFixedOverheadQuery(ag *Agent) (compactions int, errs []ErrorEvent) {
	for ev := range ag.QueryStream(context.Background(), llm.TextContent("call probe until done")) {
		switch e := ev.(type) {
		case CompactionEvent:
			compactions++
		case ErrorEvent:
			errs = append(errs, e)
		}
	}
	return compactions, errs
}

// When the provider still reports a prompt over the overflow limit for the
// first request after an overflow compaction, the part compaction cannot
// shrink does not fit: the turn ends with a typed error instead of paying for
// a summary on every step (#258). A new user input may try once more, and an
// overflow that compaction resolves keeps running.
func TestOverflowCompactionStopsWhenFixedPromptExceedsWindow(t *testing.T) {
	model := &fixedOverheadModel{prompt: func(int) int { return 31000 }}
	ag := newFixedOverheadAgent(t, model)
	compactions, errs := runFixedOverheadQuery(ag)
	if compactions != 1 || model.Summaries() != 1 {
		t.Fatalf("compactions=%d summaries=%d, want 1 and 1", compactions, model.Summaries())
	}
	if len(errs) != 1 || errs[0].Kind != "fixed_prompt_exceeds_window" ||
		!strings.Contains(errs[0].Message, "31000 prompt tokens") || !strings.Contains(errs[0].Message, fmt.Sprintf("overflow limit %d", ag.compactor.OverflowLimit())) {
		t.Fatalf("errors = %#v, want one fixed_prompt_exceeds_window error with the provider prompt and limit", errs)
	}

	// New user input: one more overflow summary attempt, then the same stop.
	_, errs = runFixedOverheadQuery(ag)
	if model.Summaries() != 2 || len(errs) != 1 || errs[0].Kind != "fixed_prompt_exceeds_window" {
		t.Fatalf("second input: summaries=%d errors=%#v, want 2 and one fixed_prompt_exceeds_window", model.Summaries(), errs)
	}

	// History-driven overflow: the compacted request fits, so the run continues.
	fits := &fixedOverheadModel{prompt: func(step int) int {
		if step == 1 {
			return 31000
		}
		return 2000
	}}
	compactions, errs = runFixedOverheadQuery(newFixedOverheadAgent(t, fits))
	if compactions != 1 || len(errs) != 0 || fits.steps != 7 {
		t.Fatalf("history overflow: compactions=%d errors=%#v steps=%d, want 1, none, 7", compactions, errs, fits.steps)
	}
}

// A summary rejected as not smaller than a small history counts as
// ineffective even though the history estimate is below the threshold: the
// provider usage (system prompt, tools) keeps every decision above it, and
// each repeat would pay for the same rejected summary (#257).
func TestNotSmallerSummaryIsNotRepeatedWithinUserInput(t *testing.T) {
	cc := compaction.DefaultConfig()
	cc.ContextWindow = 12000
	model := &summaryCountingModel{}
	ag, err := New(Config{LLM: model, Compaction: &cc, Warningf: func(string, ...any) {}, InitialMessages: []llm.Message{
		llm.NewUserMessage("rename helper foo to bar in util.go"),
		llm.NewAssistantMessage("Done: foo is now bar in util.go.", nil),
		llm.NewUserMessage("thanks, anything else?"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	f := ineffectiveSummaryFixture{agent: ag}
	above := ag.compactor.ThresholdTokens() + 100
	for i := 0; i < 4; i++ {
		f.decide(t, &llm.Usage{PromptTokens: above, TotalTokens: above})
	}
	if got := model.Calls(); got != 1 {
		t.Fatalf("summary calls = %d, want 1", got)
	}
}

// Local tiers that compacted before a not-smaller summary still leave the
// summary tier ineffective.
func TestNotSmallerSummaryAfterLocalTiersArmsSuppression(t *testing.T) {
	f := newIneffectiveSummaryFixture(t)
	f.agent.noteSummaryEffect(f.agent.userInputEpoch.Load(), compaction.Result{Compacted: true, Watermark: "prune", TiersApplied: []string{"prune"}, NewTokens: 1, SummaryNotSmaller: true})
	if !f.agent.automaticSummarySuppressed(f.agent.userInputEpoch.Load(), f.high) {
		t.Fatal("not-smaller summary after prune did not suppress the automatic summary tier")
	}
}

// imageToolModel calls a screenshot tool ten times and reports a provider
// prompt of 2000 tokens plus 1500 per image in the request.
type imageToolModel struct {
	mu               sync.Mutex
	steps, summaries int
}

func (*imageToolModel) Provider() string { return "fixture" }
func (*imageToolModel) Model() string    { return "image-tool" }

func (m *imageToolModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	images := 0
	for _, msg := range req.Messages {
		if strings.Contains(msg.Content.PlainText(), "Write an operational checkpoint") {
			m.summaries++
			return &llm.Completion{Content: llm.TextContent(validCompactionSummary("screenshots inspected so far")), StopReason: "stop"}, nil
		}
		for _, block := range msg.Content.Blocks {
			if block.Type == "image_url" {
				images++
			}
		}
	}
	m.steps++
	prompt := 2000 + 1500*images
	usage := &llm.Usage{PromptTokens: prompt, CompletionTokens: 50, TotalTokens: prompt + 50}
	if m.steps > 10 {
		return &llm.Completion{Content: llm.TextContent("finished"), StopReason: "stop", Usage: usage}, nil
	}
	call := llm.ToolCall{ID: fmt.Sprintf("shot-%d", m.steps), Type: "function", Function: llm.FunctionCall{Name: "screenshot", Arguments: `{}`}}
	return &llm.Completion{ToolCalls: []llm.ToolCall{call}, StopReason: "tool_calls", Usage: usage}, nil
}

// Images are invisible to the local estimate. An overflow they cause is
// resolved by the summary (which drops them), so the run continues instead of
// rejecting the summary and stopping with fixed_prompt_exceeds_window.
func TestImageOverflowCompactsAndContinues(t *testing.T) {
	cc := compaction.DefaultConfig()
	cc.ContextWindow = 12000
	shot := tools.Tool{
		Name:        "screenshot",
		Description: "screenshot",
		Schema:      map[string]any{"type": "object", "properties": map[string]any{}},
		Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
			return llm.Content{Blocks: []llm.ContentBlock{
				{Type: "text", Text: "screenshot"},
				{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64,AAAA"}},
			}}, nil
		},
	}
	model := &imageToolModel{}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{shot}, Compaction: &cc, MaxIterations: 30, InvokeRetryMaxAttempts: 1, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	compactions := 0
	var errs []ErrorEvent
	final := false
	for ev := range ag.QueryStream(context.Background(), llm.TextContent("take ten screenshots")) {
		switch e := ev.(type) {
		case CompactionEvent:
			compactions++
		case ErrorEvent:
			errs = append(errs, e)
		case FinalResponseEvent:
			final = true
		}
	}
	if len(errs) != 0 || !final || compactions == 0 || model.summaries == 0 {
		t.Fatalf("errors=%#v final=%v compactions=%d summaries=%d, want a compacted run that finishes", errs, final, compactions, model.summaries)
	}
}
