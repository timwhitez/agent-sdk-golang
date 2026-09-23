package agent

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// indeterminateAppend is a writer failure whose checkpoint may already be
// durable, as a host's append with an indeterminate state reports it.
type indeterminateAppend struct{}

func (indeterminateAppend) Error() string                  { return "event append state is indeterminate" }
func (indeterminateAppend) CheckpointOutcomeUnknown() bool { return true }

type outcomeWriter struct {
	mu     sync.Mutex
	writes int
	fail   func(n int) error
}

func (w *outcomeWriter) SaveCompactionCheckpoint(context.Context, compaction.CompactionCheckpoint) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.writes++
	if w.fail != nil {
		return w.fail(w.writes)
	}
	return nil
}

func (w *outcomeWriter) count() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.writes
}

func outcomeAgent(t *testing.T, writer *outcomeWriter, store *checkpointLedgerStore, cfg func(*compaction.Config)) *Agent {
	t.Helper()
	c := &compaction.Config{Enabled: true, SessionID: "outcome", LedgerStore: store, CheckpointWriter: writer}
	if cfg != nil {
		cfg(c)
	}
	ag, err := New(Config{LLM: &countingCompactionModel{}, InitialMessages: []llm.Message{llm.NewUserMessage("source"), llm.NewAssistantMessage("answer", nil)}, Compaction: c, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	return ag
}

func commitCandidate(t *testing.T, ag *Agent) ([]llm.Message, compaction.Result, error) {
	t.Helper()
	source := ag.Messages()
	candidate, res, err := ag.compactor.Compact(context.Background(), ag.llm, source)
	if err != nil {
		t.Fatal(err)
	}
	out, err := ag.CommitCompactionHistory(context.Background(), source, candidate, res)
	return source, out, err
}

// #86: a checkpoint write whose outcome is unknown is never rolled back or
// retried: history stays unpublished, the ledger is kept, and every later
// checkpoint write is refused before I/O until the compaction runtime is
// replaced after the host reconciled its store.
func TestUnknownCheckpointOutcomeIsNeitherRolledBackNorRetried(t *testing.T) {
	writer := &outcomeWriter{fail: func(n int) error {
		if n == 1 {
			return fmt.Errorf("session=s stage=append_compaction_checkpoint: %w", indeterminateAppend{})
		}
		return nil
	}}
	store := &checkpointLedgerStore{}
	ag := outcomeAgent(t, writer, store, nil)
	source := ag.Messages()

	_, out, err := commitCandidate(t, ag)
	if err == nil || !compaction.CheckpointOutcomeIsUnknown(err) || out.Compacted || !reflect.DeepEqual(source, ag.Messages()) {
		t.Fatalf("unknown outcome: compacted=%v err=%v", out.Compacted, err)
	}
	if ledger, _ := store.snapshot(); ledger == nil || ledger.Summary == nil {
		t.Fatal("ledger was rolled back although the checkpoint may be durable")
	}
	if !strings.Contains(strings.Join(out.Warnings, "\n"), "outcome unknown") {
		t.Fatalf("warnings=%v", out.Warnings)
	}

	// Refused before the writer: no second, possibly duplicate, checkpoint.
	_, out, err = commitCandidate(t, ag)
	if !errors.Is(err, compaction.ErrCheckpointStoreQuarantined) || out.Compacted || writer.count() != 1 {
		t.Fatalf("quarantined commit: err=%v writes=%d", err, writer.count())
	}
	if ag.turnActive.Load() || ag.compactionInFlight.Load() {
		t.Fatal("refusal did not release the operation")
	}

	// A replacement runtime (the host reconciled its store) writes again.
	ag.resetCompactionOutcomeState()
	if _, out, err = commitCandidate(t, ag); err != nil || !out.Compacted || writer.count() != 2 {
		t.Fatalf("after replacement: compacted=%v err=%v writes=%d", out.Compacted, err, writer.count())
	}
}

// A plain writer failure keeps its retryable behavior: the ledger is rolled
// back and the next write is attempted.
func TestPlainCheckpointFailureStaysRetryable(t *testing.T) {
	writer := &outcomeWriter{fail: func(n int) error {
		if n == 1 {
			return errors.New("permission denied")
		}
		return nil
	}}
	store := &checkpointLedgerStore{}
	ag := outcomeAgent(t, writer, store, nil)
	if _, _, err := commitCandidate(t, ag); err == nil || compaction.CheckpointOutcomeIsUnknown(err) {
		t.Fatalf("plain failure err=%v", err)
	}
	if ledger, _ := store.snapshot(); ledger != nil && ledger.Summary != nil {
		t.Fatal("plain failure kept the pending ledger")
	}
	if _, out, err := commitCandidate(t, ag); err != nil || !out.Compacted || writer.count() != 2 {
		t.Fatalf("retry: compacted=%v err=%v writes=%d", out.Compacted, err, writer.count())
	}
}

// The automatic apply path does not requeue or schedule a retry after an
// unknown outcome, and later automatic compaction writes nothing.
func TestAutomaticApplyDoesNotRetryUnknownCheckpointOutcome(t *testing.T) {
	writer := &outcomeWriter{fail: func(int) error { return indeterminateAppend{} }}
	history := []llm.Message{llm.NewSystemMessage("base")}
	for i := 0; i < 6; i++ {
		history = append(history, llm.NewUserMessage(strings.Repeat("old ", 600)), llm.NewAssistantMessage(strings.Repeat("reply ", 600), nil))
	}
	history = append(history, llm.NewUserMessage("latest"))
	total := compaction.NewService(&compaction.Config{Enabled: true}).EstimateMessages(history)
	model := &summaryCountingModel{}
	ag, err := New(Config{LLM: model, InitialMessages: history, Warningf: func(string, ...any) {}, Compaction: &compaction.Config{
		Enabled: true, SessionID: "outcome", LedgerStore: &checkpointLedgerStore{}, CheckpointWriter: writer,
		ContextWindow: 2 * total, ReserveOutputTokens: 1, ThresholdRatio: 0.4, SnipThresholdRatio: 0.3, PruneThresholdRatio: 0.35, KeepRecentUserMessages: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	f := ineffectiveSummaryFixture{agent: ag}
	high := &llm.Usage{PromptTokens: total, TotalTokens: total}
	f.decide(t, high) // compacts, then the apply's checkpoint write has an unknown outcome
	if writer.count() != 1 || ag.hasPendingCompaction() || ag.compactionRetryPending.Load() || !reflect.DeepEqual(history, ag.Messages()) {
		t.Fatalf("writes=%d pending=%v retry=%v", writer.count(), ag.hasPendingCompaction(), ag.compactionRetryPending.Load())
	}
	summaries := model.Calls()
	f.decide(t, high)
	f.decide(t, high)
	if writer.count() != 1 || model.Calls() != summaries {
		t.Fatalf("automatic compaction ran again after an unknown outcome: writes=%d summaries=%d→%d", writer.count(), summaries, model.Calls())
	}
}
