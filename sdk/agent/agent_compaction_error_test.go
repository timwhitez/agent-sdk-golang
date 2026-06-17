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
	return &llm.Completion{Content: llm.TextContent("<summary>retry succeeded</summary>")}, nil
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

func (m *countingCompactionModel) Provider() string { return "stub" }
func (m *countingCompactionModel) Model() string    { return "stub" }
func (m *countingCompactionModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls++
	return &llm.Completion{Content: llm.TextContent("<summary>ok</summary>")}, nil
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
	return &llm.Completion{Content: llm.TextContent("<summary>completed after turn cancellation</summary>")}, nil
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
	comp := &llm.Completion{Usage: &llm.Usage{TotalTokens: 120, PromptTokens: 100}}
	ag.checkAndCompact(ctx, comp, nil)

	if got := model.Calls(); got != 0 {
		t.Fatalf("expected canceled context to skip compaction invoke, got %d", got)
	}
	if ag.compactionInFlight.Load() {
		t.Fatal("expected compactionInFlight=false when context is canceled")
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
	comp := &llm.Completion{Usage: &llm.Usage{TotalTokens: 120, PromptTokens: 100}}
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
