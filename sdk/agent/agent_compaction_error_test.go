package agent

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type compactionErrorModel struct{}

func (m compactionErrorModel) Provider() string { return "stub" }
func (m compactionErrorModel) Model() string    { return "stub" }

func (m compactionErrorModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("boom")
}

type flakyCompactionModel struct {
	mu      sync.Mutex
	calls   int
	failFor int
}

func (m *flakyCompactionModel) Provider() string { return "stub" }
func (m *flakyCompactionModel) Model() string    { return "stub" }
func (m *flakyCompactionModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	m.calls++
	call := m.calls
	failFor := m.failFor
	m.mu.Unlock()
	if call <= failFor {
		return nil, errors.New("temporary compaction failure")
	}
	return &llm.Completion{Content: llm.TextContent(validCompactionSummary("retry succeeded"))}, nil
}

func (m *flakyCompactionModel) Calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

type countingCompactionModel struct {
	mu    sync.Mutex
	calls int
}

type providerContextOverflowModel struct {
	mu           sync.Mutex
	requests     []llm.InvokeRequest
	summaryCalls int
}

func (m *providerContextOverflowModel) Provider() string { return "fixture" }
func (m *providerContextOverflowModel) Model() string    { return "context-overflow" }
func (m *providerContextOverflowModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	owned, err := llm.CloneInvokeRequest(req)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	m.requests = append(m.requests, owned)
	for _, message := range owned.Messages {
		if message.Role == llm.RoleSystem && strings.Contains(message.Content.PlainText(), "operational checkpoint") {
			m.summaryCalls++
		}
	}
	m.mu.Unlock()
	return nil, &llm.ProviderError{Provider: "fixture", StatusCode: 400, Message: "maximum context length exceeded"}
}

func (m *providerContextOverflowModel) Snapshot() ([]llm.InvokeRequest, int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llm.InvokeRequest(nil), m.requests...), m.summaryCalls
}

func (m *countingCompactionModel) Provider() string { return "stub" }
func (m *countingCompactionModel) Model() string    { return "stub" }
func (m *countingCompactionModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls++
	return &llm.Completion{Content: llm.TextContent(validCompactionSummary("ok"))}, nil
}

func (m *countingCompactionModel) Calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

type cancelAwareCompactionModel struct {
	entered chan struct{}
	release chan struct{}

	once sync.Once
}

type overflowFailureBoundaryModel struct {
	mu           sync.Mutex
	normalCalls  int
	summaryCalls int
}

func (m *overflowFailureBoundaryModel) Provider() string { return "stub" }
func (m *overflowFailureBoundaryModel) Model() string    { return "stub" }
func (m *overflowFailureBoundaryModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, message := range req.Messages {
		if message.Role == llm.RoleSystem && strings.Contains(message.Content.PlainText(), "operational checkpoint") {
			m.summaryCalls++
			return nil, errors.New("injected overflow compaction failure")
		}
	}
	m.normalCalls++
	if m.normalCalls > 1 {
		return nil, errors.New("provider crossed overflow boundary")
	}
	return &llm.Completion{
		Content:    llm.TextContent("partial response"),
		StopReason: "max_tokens",
		Usage:      llm.NewProviderUsage(90, 10, 100),
	}, nil
}

func (m *overflowFailureBoundaryModel) Counts() (normal, summary int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.normalCalls, m.summaryCalls
}

func (m *cancelAwareCompactionModel) Provider() string { return "stub" }
func (m *cancelAwareCompactionModel) Model() string    { return "stub" }
func (m *cancelAwareCompactionModel) Invoke(ctx context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.once.Do(func() { close(m.entered) })
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-m.release:
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &llm.Completion{Content: llm.TextContent(validCompactionSummary("completed after turn cancellation"))}, nil
}

type lockedBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (w *lockedBuffer) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.b.Write(p)
}

func (w *lockedBuffer) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.b.String()
}

func waitFor(t *testing.T, timeout time.Duration, cond func() bool, what string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestCheckAndCompactLogsError(t *testing.T) {
	var buf lockedBuffer
	origOut := log.Writer()
	origFlags := log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(origOut)
		log.SetFlags(origFlags)
	})

	ag, err := New(Config{LLM: compactionErrorModel{}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	comp := &llm.Completion{
		Usage: &llm.Usage{
			TotalTokens:      200_000,
			CompletionTokens: 0,
		},
	}

	ag.checkAndCompact(context.Background(), comp, nil)
	waitFor(t, time.Second, func() bool {
		return strings.Contains(buf.String(), "compaction failed")
	}, "compaction failure log")
	waitFor(t, time.Second, func() bool {
		return !ag.compactionInFlight.Load()
	}, "async compaction completion")

	if got := buf.String(); !strings.Contains(got, "compaction failed") {
		t.Fatalf("expected compaction error to be logged, got %q", got)
	}
}

func TestCheckAndCompactUsesConfiguredWarningSink(t *testing.T) {
	var warnings lockedBuffer
	ag, err := New(Config{
		LLM: compactionErrorModel{},
		Warningf: func(format string, args ...any) {
			_, _ = warnings.Write([]byte(fmt.Sprintf(format, args...) + "\n"))
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	comp := &llm.Completion{
		Usage: &llm.Usage{
			TotalTokens:      200_000,
			CompletionTokens: 0,
		},
	}

	ag.checkAndCompact(context.Background(), comp, nil)
	waitFor(t, time.Second, func() bool {
		return strings.Contains(warnings.String(), "compaction failed")
	}, "compaction failure warning")
	waitFor(t, time.Second, func() bool {
		return !ag.compactionInFlight.Load()
	}, "async compaction completion")

	if got := warnings.String(); !strings.Contains(got, "compaction failed") {
		t.Fatalf("expected compaction warning in configured sink, got %q", got)
	}
}

func TestCheckAndCompactCancelsAsyncCompactionWithTurn(t *testing.T) {
	model := &cancelAwareCompactionModel{
		entered: make(chan struct{}),
		release: make(chan struct{}),
	}
	var warnings lockedBuffer
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:                true,
			ContextWindow:          100,
			ThresholdRatio:         0.5,
			SummaryPrompt:          "summarize",
			KeepRecentUserMessages: 1,
			CompactionRetryBackoff: 5 * time.Millisecond,
		},
		Warningf: func(format string, args ...any) {
			_, _ = warnings.Write([]byte(fmt.Sprintf(format, args...) + "\n"))
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("hello")})

	ctx, cancel := context.WithCancel(context.Background())
	comp := &llm.Completion{Usage: &llm.Usage{TotalTokens: 100, PromptTokens: 80}}
	ag.checkAndCompact(ctx, comp, nil)
	select {
	case <-model.entered:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for compaction to start")
	}
	cancel()
	waitFor(t, time.Second, func() bool {
		return !ag.compactionInFlight.Load()
	}, "async compaction cancellation")

	if got := warnings.String(); got != "" {
		t.Fatalf("expected no warning for canceled async compaction, got %q", got)
	}
	if ag.todoCompactionPending.Load() {
		t.Fatal("canceled turn should not schedule a compaction retry")
	}
}

func TestCheckAndCompactRetriesOnceByDefault(t *testing.T) {
	model := &flakyCompactionModel{failFor: 1}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:                true,
			ContextWindow:          100,
			ThresholdRatio:         0.5,
			SummaryPrompt:          "summarize",
			KeepRecentUserMessages: 1,
			CompactionRetryBackoff: 5 * time.Millisecond,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("hello")})

	comp := &llm.Completion{Usage: &llm.Usage{TotalTokens: 100, PromptTokens: 80}}
	ag.checkAndCompact(context.Background(), comp, nil)
	waitFor(t, time.Second, func() bool {
		return model.Calls() == 2 && ag.hasPendingCompaction()
	}, "async compaction retry completion")
	ag.applyPendingCompaction(nil)

	if got := model.Calls(); got != 2 {
		t.Fatalf("expected compaction invoke to retry once, got %d calls", got)
	}
	messages := ag.Messages()
	if len(messages) == 0 || messages[0].Name != "compaction_summary" {
		t.Fatalf("expected compacted summary message, got %#v", messages)
	}
}

func TestOverflowCompactionFailureStopsBeforeNextProviderCall(t *testing.T) {
	model := &overflowFailureBoundaryModel{}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:                true,
			ContextWindow:          100,
			ThresholdRatio:         0.85,
			CompactionRetryBackoff: time.Millisecond,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var sawError bool
	for event := range ag.QueryStream(context.Background(), llm.TextContent("continue safely")) {
		if _, ok := event.(ErrorEvent); ok {
			sawError = true
		}
	}
	normal, summary := model.Counts()
	if !sawError {
		t.Fatal("overflow compaction failure did not surface an error event")
	}
	if normal != 1 {
		t.Fatalf("normal provider calls = %d, want 1; overflow failure must stop before the next provider request", normal)
	}
	if summary != 2 {
		t.Fatalf("summary attempts = %d, want default two attempts", summary)
	}
}

func TestCheckAndCompactSkipsInvokeWhenBelowThreshold(t *testing.T) {
	model := &countingCompactionModel{}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("hello")})

	comp := &llm.Completion{Usage: &llm.Usage{TotalTokens: 10, PromptTokens: 10}}
	ag.checkAndCompact(context.Background(), comp, nil)

	if got := model.Calls(); got != 0 {
		t.Fatalf("expected no compaction invoke below threshold, got %d", got)
	}
}

func TestCheckAndCompactSkipsInvokeWhenContextCanceled(t *testing.T) {
	model := &countingCompactionModel{}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:                true,
			ContextWindow:          100,
			ThresholdRatio:         0.5,
			SummaryPrompt:          "summarize",
			KeepRecentUserMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("hello")})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	comp := &llm.Completion{Usage: &llm.Usage{TotalTokens: 120, PromptTokens: 99}}
	ag.checkAndCompact(ctx, comp, nil)

	if got := model.Calls(); got != 0 {
		t.Fatalf("expected canceled context to skip compaction invoke, got %d", got)
	}
	if ag.compactionInFlight.Load() {
		t.Fatal("expected compactionInFlight=false when context is canceled")
	}
}

func TestCompactionLegacyDecisionParity(t *testing.T) {
	ag, err := New(Config{
		LLM: &countingCompactionModel{},
		Compaction: &compaction.Config{
			Enabled:             true,
			ContextWindow:       100,
			ReserveOutputTokens: 0,
			SnipThresholdRatio:  0.70,
			PruneThresholdRatio: 0.80,
			ThresholdRatio:      0.85,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		tokens    int
		trigger   string
		watermark string
		attempt   bool
	}{
		{tokens: 69, trigger: "usage", watermark: "", attempt: false},
		{tokens: 70, trigger: "usage", watermark: "snip", attempt: true},
		{tokens: 80, trigger: "usage", watermark: "prune", attempt: true},
		{tokens: 85, trigger: "usage", watermark: "summarize", attempt: true},
		{tokens: 100, trigger: "overflow", watermark: "overflow", attempt: true},
	} {
		usage := llm.WithPromptEstimate(nil, test.tokens)
		trigger, watermark := ag.compactionTriggerAndWatermarkForUsage(usage)
		if trigger != test.trigger || watermark != test.watermark {
			t.Errorf("tokens=%d decision=%q/%q want %q/%q", test.tokens, trigger, watermark, test.trigger, test.watermark)
		}
		if got := ag.shouldAttemptCompactionUsage(context.Background(), usage); got != test.attempt {
			t.Errorf("tokens=%d attempt=%v want %v", test.tokens, got, test.attempt)
		}
	}

	eligible := llm.WithPromptEstimate(nil, 70)
	ag.todoCompactionPending.Store(true)
	ag.compactionRetryPending.Store(true)
	if trigger, _ := ag.compactionTriggerAndWatermarkForUsage(eligible); trigger != "todo_checkpoint" {
		t.Fatalf("todo+retry trigger=%q want todo_checkpoint", trigger)
	}
	ag.todoCompactionPending.Store(false)
	if trigger, _ := ag.compactionTriggerAndWatermarkForUsage(eligible); trigger != "retry_checkpoint" {
		t.Fatalf("retry trigger=%q want retry_checkpoint", trigger)
	}
	ag.compactionRetryPending.Store(false)
	if trigger, _ := ag.compactionTriggerAndWatermarkForUsage(eligible); trigger != "usage" {
		t.Fatalf("plain trigger=%q want usage", trigger)
	}

	destroyed := []llm.Message{llm.NewUserMessage("request")}
	for i := 0; i < defaultDestroyedToolCompactThreshold; i++ {
		destroyed = append(destroyed, llm.Message{Role: llm.RoleTool, Destroyed: true, Content: llm.TextContent(ephemeralReleasedPlaceholder)})
	}
	ag.ReplaceHistory(destroyed)
	low := llm.WithPromptEstimate(nil, 1)
	if trigger, watermark := ag.compactionTriggerAndWatermarkForUsage(low); trigger != "placeholder_pressure" || watermark != "placeholder_cleanup" {
		t.Fatalf("placeholder decision=%q/%q", trigger, watermark)
	}

	ag.compactionInFlight.Store(true)
	if ag.shouldAttemptCompactionUsage(context.Background(), eligible) {
		t.Fatal("in-flight compaction did not suppress eligible trigger")
	}
	ag.compactionInFlight.Store(false)
	ag.pendingCompactionMu.Lock()
	ag.pendingCompaction = &pendingCompaction{}
	ag.pendingCompactionMu.Unlock()
	if ag.shouldAttemptCompactionUsage(context.Background(), eligible) {
		t.Fatal("pending compaction did not suppress eligible trigger")
	}
	ag.pendingCompactionMu.Lock()
	ag.pendingCompaction = nil
	ag.pendingCompactionMu.Unlock()
	ag.compactionCooldownUntil.Store(time.Now().Add(time.Hour).UnixNano())
	if ag.shouldAttemptCompactionUsage(context.Background(), eligible) {
		t.Fatal("cooldown did not suppress eligible non-overflow trigger")
	}
	if !ag.shouldAttemptCompactionUsage(context.Background(), llm.WithPromptEstimate(nil, 100)) {
		t.Fatal("overflow did not bypass cooldown")
	}
}

func TestProviderRejectedContextOverflowIsTerminal(t *testing.T) {
	model := &providerContextOverflowModel{}
	ag, err := New(Config{
		LLM:                    model,
		InvokeRetryMaxAttempts: 3,
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  100,
			ThresholdRatio: 0.85,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	errorsSeen := 0
	compactions := 0
	finals := 0
	for event := range ag.QueryStream(context.Background(), llm.TextContent("bounded request")) {
		switch event := event.(type) {
		case ErrorEvent:
			errorsSeen++
			if event.Kind != "invalid_request" || event.StatusCode != 400 {
				t.Fatalf("terminal error=%#v want invalid_request/400", event)
			}
		case CompactionEvent:
			compactions++
		case FinalResponseEvent:
			finals++
		}
	}

	requests, summaryCalls := model.Snapshot()
	if len(requests) != 1 || summaryCalls != 0 {
		t.Fatalf("provider admissions=%d summary_calls=%d want 1/0", len(requests), summaryCalls)
	}
	if errorsSeen != 1 || compactions != 0 || finals != 0 {
		t.Fatalf("errors=%d compactions=%d finals=%d want 1/0/0", errorsSeen, compactions, finals)
	}
	if ag.hasPendingCompaction() || ag.compactionInFlight.Load() || ag.compactionRetryPending.Load() || ag.todoCompactionPending.Load() || ag.compactionGeneration.Load() != 0 {
		t.Fatalf("provider rejection changed compaction state: pending=%v in_flight=%v retry=%v todo=%v generation=%d",
			ag.hasPendingCompaction(), ag.compactionInFlight.Load(), ag.compactionRetryPending.Load(), ag.todoCompactionPending.Load(), ag.compactionGeneration.Load())
	}
	request := requests[0]
	if len(request.Messages) != 1 || request.Messages[0].Role != llm.RoleUser || request.Messages[0].Content.PlainText() != "bounded request" {
		t.Fatalf("first provider payload changed: %#v", request.Messages)
	}
}

func TestCheckAndCompactAsyncSurvivesCallerCancelAfterStart(t *testing.T) {
	model := &cancelAwareCompactionModel{
		entered: make(chan struct{}),
		release: make(chan struct{}),
	}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:                true,
			ContextWindow:          100,
			ThresholdRatio:         0.5,
			SummaryPrompt:          "summarize",
			KeepRecentUserMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("hello")})

	ctx, cancel := context.WithCancel(context.Background())
	comp := &llm.Completion{Usage: &llm.Usage{TotalTokens: 120, PromptTokens: 99}}
	ag.checkAndCompact(ctx, comp, nil)

	select {
	case <-model.entered:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for compaction invoke to start")
	}
	cancel()
	close(model.release)

	waitFor(t, time.Second, func() bool {
		return !ag.compactionInFlight.Load()
	}, "async compaction completion")
	if !ag.hasPendingCompaction() {
		t.Fatal("expected pending compaction to survive caller cancellation after start")
	}
}

func TestNewCachesDisabledCompactorState(t *testing.T) {
	model := &countingCompactionModel{}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled: false,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if ag.hasCompactor {
		t.Fatalf("expected hasCompactor=false when compaction is disabled")
	}
	if ag.compactor != nil {
		t.Fatalf("expected compactor service to be nil when compaction is disabled")
	}

	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("hello")})
	comp := &llm.Completion{Usage: &llm.Usage{TotalTokens: 1000, PromptTokens: 1000}}
	ag.checkAndCompact(context.Background(), comp, nil)
	if got := model.Calls(); got != 0 {
		t.Fatalf("expected disabled compaction to skip model invoke, got %d", got)
	}
}

// emergencyTrimTestAgent builds an agent whose emergency-trim budget equals the
// configured context window, so trim decisions are easy to reason about in
// tokens.
func emergencyTrimTestAgent(t *testing.T, window int) *Agent {
	t.Helper()
	ag, err := New(Config{
		LLM: &countingCompactionModel{},
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  window,
			ThresholdRatio: 1.0,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return ag
}

// REG: reporting success for a history that still overflows made
// applyEmergencyTrim publish Compacted:true and compactOverflow return nil, and
// the turn then sent a request that was still over the window instead of
// surfacing the overflow error. The guarantee is therefore "reported success
// implies the result fits the budget".
//
// This fixture's newest block is itself oversized. The trim answers that by
// giving the block up and keeping the older content that does fit - refusing the
// whole trim here would abort the turn even though a legal in-budget history
// exists (R4-CC-001). Either way it must never claim success for an over-budget
// result; the genuinely irreducible case is covered by
// TestEmergencyTrimStillRefusesGenuinelyIrreducibleHistory.
func TestEmergencyTrimNeverReportsSuccessForResultOverBudget(t *testing.T) {
	ag := emergencyTrimTestAgent(t, 1000)
	messages := []llm.Message{
		llm.NewSystemMessage("sys"),
		llm.NewUserMessage("real request"),
		llm.NewAssistantMessage(strings.Repeat("a ", 500), nil),
		llm.NewAssistantMessage(strings.Repeat("b ", 500), nil),
		llm.NewAssistantMessage(strings.Repeat("x ", 3000), nil),
	}
	budget := ag.compactor.ThresholdTokens()
	if budget <= 0 {
		t.Fatalf("expected a positive trim budget, got %d", budget)
	}
	if before := ag.compactor.EstimateMessages(messages); before <= budget {
		t.Fatalf("test history must start over budget: estimate=%d budget=%d", before, budget)
	}
	trimmed, ok := ag.emergencyTrimHistory(messages)
	if !ok {
		t.Fatalf("emergency trim refused although dropping the oversized newest block leaves a legal %d-token history for a %d budget",
			ag.compactor.EstimateMessages(messages[:len(messages)-1]), budget)
	}
	if estimate := ag.compactor.EstimateMessages(trimmed); estimate > budget {
		t.Fatalf("emergency trim reported success while the retained history still needs %d tokens of a %d budget",
			estimate, budget)
	}
	if len(trimmed) >= len(messages) {
		t.Fatalf("an accepted trim must shrink history, got %d of %d messages", len(trimmed), len(messages))
	}
}

// REG: a trim that does fit the budget must still be accepted, so the refusal
// above cannot be satisfied by refusing everything.
func TestEmergencyTrimStillAcceptsResultWithinBudget(t *testing.T) {
	ag := emergencyTrimTestAgent(t, 1000)
	messages := []llm.Message{
		llm.NewSystemMessage("sys"),
		llm.NewUserMessage("real request"),
		llm.NewAssistantMessage(strings.Repeat("a ", 800), nil),
		llm.NewAssistantMessage(strings.Repeat("b ", 800), nil),
		llm.NewAssistantMessage(strings.Repeat("c ", 800), nil),
	}
	budget := ag.compactor.ThresholdTokens()
	if before := ag.compactor.EstimateMessages(messages); before <= budget {
		t.Fatalf("test history must start over budget: estimate=%d budget=%d", before, budget)
	}
	trimmed, ok := ag.emergencyTrimHistory(messages)
	if !ok {
		t.Fatal("emergency trim refused a history whose newest block fits the budget")
	}
	if estimate := ag.compactor.EstimateMessages(trimmed); estimate > budget {
		t.Fatalf("accepted trim is over budget: estimate=%d budget=%d", estimate, budget)
	}
	if len(trimmed) >= len(messages) {
		t.Fatalf("accepted trim did not shrink history: %d -> %d", len(messages), len(trimmed))
	}
}

// emergencyTrimInflightHistory builds a history that ends with an assistant
// tool_use whose arguments are still being accumulated by an in-flight tool-call
// continuation: it has no tool_result yet. withReminder controls whether the
// framework's continuation reminder has already been appended — compaction at
// the overflow boundary can run either before or after that append.
func emergencyTrimInflightHistory(withReminder bool) []llm.Message {
	messages := []llm.Message{
		llm.NewSystemMessage("sys"),
		llm.NewUserMessage("do the work"),
	}
	for i := 0; i < 8; i++ {
		id := fmt.Sprintf("call-%d", i)
		messages = append(messages,
			llm.NewAssistantMessage(strings.Repeat("step ", 40), []llm.ToolCall{{
				ID:       id,
				Type:     "function",
				Function: llm.FunctionCall{Name: "read", Arguments: `{"path":"f.go"}`},
			}}),
			llm.NewToolMessage(id, "read", llm.TextContent(strings.Repeat("result ", 60)), false),
		)
	}
	messages = append(messages, llm.NewAssistantMessage("partial", []llm.ToolCall{{
		ID:       "call-inflight",
		Type:     "function",
		Function: llm.FunctionCall{Name: "write", Arguments: `{"path":"x.go","content":"package ma`},
	}}))
	if withReminder {
		messages = append(messages, messageorigin.NewInternalUserMessage(
			messageorigin.KindToolCallContinuation, messageorigin.ResponseTruncatedContinuationText))
	}
	return messages
}

// REG: the trim's repair pass used to clear ToolCalls on the trailing in-flight
// tool_use, destroying the partial arguments the continuation has to merge the
// next chunk into and leaving the continuation unable to ever complete.
func TestEmergencyTrimPreservesInFlightToolCallContinuation(t *testing.T) {
	for _, withReminder := range []bool{false, true} {
		messages := emergencyTrimInflightHistory(withReminder)
		ag := emergencyTrimTestAgent(t, 1200)
		trimmed, ok := ag.emergencyTrimHistory(messages)
		if !ok {
			t.Fatalf("withReminder=%v: emergency trim refused a reducible history", withReminder)
		}
		args := ""
		found := false
		for _, m := range trimmed {
			for _, call := range m.ToolCalls {
				if call.ID == "call-inflight" {
					found = true
					args = call.Function.Arguments
				}
			}
		}
		if !found {
			t.Fatalf("withReminder=%v: trim stripped the in-flight tool_use; the continuation can never complete", withReminder)
		}
		if args != `{"path":"x.go","content":"package ma` {
			t.Fatalf("withReminder=%v: in-flight partial arguments were altered: %q", withReminder, args)
		}
	}
}

// REG: preserving the in-flight block must not turn into preserving genuinely
// unpaired tool_use blocks — a trailing assistant tool_use that is followed by a
// non-continuation message is still repaired.
func TestEmergencyTrimStillRepairsUnpairedToolCallsBeforeTail(t *testing.T) {
	messages := emergencyTrimInflightHistory(false)
	// Drop one tool_result so an older block becomes genuinely unpaired.
	broken := append([]llm.Message(nil), messages[:5]...)
	broken = append(broken, messages[6:]...)
	ag := emergencyTrimTestAgent(t, 1200)
	trimmed, ok := ag.emergencyTrimHistory(broken)
	if !ok {
		t.Fatal("emergency trim refused a reducible history")
	}
	for i, m := range trimmed {
		if m.Role != llm.RoleAssistant || len(m.ToolCalls) == 0 {
			continue
		}
		if m.ToolCalls[0].ID == "call-inflight" {
			continue
		}
		results := 0
		for j := i + 1; j < len(trimmed) && trimmed[j].Role == llm.RoleTool; j++ {
			results++
		}
		if results != len(m.ToolCalls) {
			t.Fatalf("trimmed[%d] kept %d tool_use blocks with %d results; pairing was not repaired",
				i, len(m.ToolCalls), results)
		}
	}
}
