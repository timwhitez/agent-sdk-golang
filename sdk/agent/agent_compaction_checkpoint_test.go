package agent

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type checkpointLedgerStore struct {
	mu         sync.Mutex
	ledger     *compaction.Ledger
	saves      int
	failSaveAt map[int]error
}

func (s *checkpointLedgerStore) Load(_ context.Context, sessionID string) (*compaction.Ledger, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ledger == nil {
		return compaction.NewLedger(sessionID), nil
	}
	return s.ledger.Clone(), nil
}

func (s *checkpointLedgerStore) Save(_ context.Context, _ string, ledger *compaction.Ledger) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.saves++
	if err := s.failSaveAt[s.saves]; err != nil {
		return err
	}
	s.ledger = ledger.Clone()
	return nil
}

func (s *checkpointLedgerStore) snapshot() (*compaction.Ledger, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ledger == nil {
		return nil, s.saves
	}
	return s.ledger.Clone(), s.saves
}

func TestCompactionCheckpointWriterFailurePreservesHistory(t *testing.T) {
	original := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewUserMessage("keep this original history"),
		llm.NewAssistantMessage("original assistant state", nil),
	}
	writes := 0
	ledgerStore := &checkpointLedgerStore{}
	ag, err := New(Config{
		LLM: &countingCompactionModel{},
		Compaction: &compaction.Config{
			Enabled:     true,
			SessionID:   "checkpoint-writer-failure",
			LedgerStore: ledgerStore,
			CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
				writes++
				return errors.New("events.jsonl append denied")
			}),
		},
		InitialMessages: original,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := ag.CompactNow(context.Background())
	if err == nil || !strings.Contains(err.Error(), "checkpoint persistence") {
		t.Fatalf("CompactNow error = %v, want checkpoint persistence failure", err)
	}
	if writes != 1 {
		t.Fatalf("checkpoint writes = %d, want 1", writes)
	}
	if res.Compacted || !containsAgentWarning(res.Warnings, "original in-memory history was preserved") {
		t.Fatalf("failed checkpoint result = %#v", res)
	}
	if got := ag.Messages(); !reflect.DeepEqual(got, original) {
		t.Fatalf("checkpoint failure mutated history:\n got=%#v\nwant=%#v", got, original)
	}
	if ledger, _ := ledgerStore.snapshot(); ledger != nil && ledger.Summary != nil {
		t.Fatalf("checkpoint failure left a summary ledger that does not match current history: %#v", ledger.Summary)
	}
}

func TestSummaryLedgerDefersUntilCheckpointCommit(t *testing.T) {
	original := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewUserMessage("compact this history"),
		llm.NewAssistantMessage("old assistant state", nil),
	}
	ledgerStore := &checkpointLedgerStore{}
	writes := 0
	ag, err := New(Config{
		LLM: &countingCompactionModel{},
		Compaction: &compaction.Config{
			Enabled:     true,
			SessionID:   "deferred-summary-ledger",
			LedgerStore: ledgerStore,
			CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
				writes++
				return nil
			}),
		},
		InitialMessages: original,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	newMessages, res, err := ag.compactor.Compact(context.Background(), ag.llm, original)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if ledger, saves := ledgerStore.snapshot(); saves != 0 || (ledger != nil && ledger.Summary != nil) {
		t.Fatalf("summary ledger persisted before runtime checkpoint: saves=%d ledger=%#v", saves, ledger)
	}

	committed, err := ag.CommitCompactionCheckpoint(context.Background(), newMessages, res)
	if err != nil {
		t.Fatalf("CommitCompactionCheckpoint: %v", err)
	}
	if !committed.Compacted || writes != 1 {
		t.Fatalf("checkpoint commit result=%#v writes=%d", committed, writes)
	}
	ledger, saves := ledgerStore.snapshot()
	if saves != 1 || ledger == nil || ledger.Summary == nil {
		t.Fatalf("summary ledger not committed with checkpoint: saves=%d ledger=%#v", saves, ledger)
	}
}

func TestCompactionLedgerCommitFailureSkipsRuntimeCheckpoint(t *testing.T) {
	original := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewUserMessage("compact this history"),
		llm.NewAssistantMessage("old assistant state", nil),
	}
	ledgerStore := &checkpointLedgerStore{failSaveAt: map[int]error{1: errors.New("ledger disk full")}}
	writes := 0
	ag, err := New(Config{
		LLM: &countingCompactionModel{},
		Compaction: &compaction.Config{
			Enabled:     true,
			SessionID:   "ledger-commit-failure",
			LedgerStore: ledgerStore,
			CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
				writes++
				return nil
			}),
		},
		InitialMessages: original,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	newMessages, res, err := ag.compactor.Compact(context.Background(), ag.llm, original)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	committed, err := ag.CommitCompactionCheckpoint(context.Background(), newMessages, res)
	if err == nil || !strings.Contains(err.Error(), "compaction ledger persistence failed") {
		t.Fatalf("commit error = %v", err)
	}
	if committed.Compacted || writes != 0 {
		t.Fatalf("ledger failure crossed checkpoint boundary: result=%#v writes=%d", committed, writes)
	}
	if len(committed.Warnings) == 0 || !strings.Contains(committed.Warnings[len(committed.Warnings)-1], "stage=save_compaction_ledger") {
		t.Fatalf("missing actionable ledger failure warning: %#v", committed.Warnings)
	}
	if ledger, saves := ledgerStore.snapshot(); saves != 1 || (ledger != nil && ledger.Summary != nil) {
		t.Fatalf("failed ledger commit mutated durable state: saves=%d ledger=%#v", saves, ledger)
	}
}

func TestCompactionLedgerRollbackFailureIsVisible(t *testing.T) {
	original := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewUserMessage("compact this history"),
		llm.NewAssistantMessage("old assistant state", nil),
	}
	ledgerStore := &checkpointLedgerStore{failSaveAt: map[int]error{2: errors.New("rollback disk full")}}
	writes := 0
	ag, err := New(Config{
		LLM: &countingCompactionModel{},
		Compaction: &compaction.Config{
			Enabled:     true,
			SessionID:   "ledger-rollback-failure",
			LedgerStore: ledgerStore,
			CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
				writes++
				return errors.New("events append denied")
			}),
		},
		InitialMessages: original,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	newMessages, res, err := ag.compactor.Compact(context.Background(), ag.llm, original)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	committed, err := ag.CommitCompactionCheckpoint(context.Background(), newMessages, res)
	if err == nil || !strings.Contains(err.Error(), "ledger rollback failed") {
		t.Fatalf("commit error = %v", err)
	}
	if committed.Compacted || writes != 1 {
		t.Fatalf("rollback failure result=%#v writes=%d", committed, writes)
	}
	if len(committed.Warnings) < 2 || !strings.Contains(committed.Warnings[len(committed.Warnings)-1], "[ERROR] Compaction ledger rollback failed") {
		t.Fatalf("rollback failure was not surfaced separately: %#v", committed.Warnings)
	}
	ledger, saves := ledgerStore.snapshot()
	if saves != 2 || ledger == nil || ledger.Summary == nil {
		t.Fatalf("expected failed rollback to leave the attempted ledger visible: saves=%d ledger=%#v", saves, ledger)
	}
}

func TestOverflowLocalFallbackCommitsDeferredLedgerWithCheckpoint(t *testing.T) {
	ledgerStore := &checkpointLedgerStore{}
	writes := 0
	ag, err := New(Config{
		LLM: compactionErrorModel{},
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  100,
			ThresholdRatio: 0.85,
			SessionID:      "overflow-local-fallback",
			LedgerStore:    ledgerStore,
			CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
				writes++
				return nil
			}),
			ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
				return compaction.ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
			}),
			ProtectedRecentMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	messages := []llm.Message{
		llm.NewUserMessage(strings.Repeat("unreduced repository constraint ", 200)),
		llm.NewAssistantMessage("calling grep", []llm.ToolCall{{ID: "call-grep", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage("call-grep", "grep", llm.TextContent(strings.Repeat("large tool result\n", 300)), false),
		llm.NewUserMessage("latest protected request"),
	}

	newMessages, res, err := ag.compactOverflowWithRetry(context.Background(), messages, &llm.Usage{PromptTokens: 100, TotalTokens: 100})
	if err != nil {
		t.Fatalf("compactOverflowWithRetry: %v", err)
	}
	if !res.Compacted || len(newMessages) == 0 {
		t.Fatalf("expected local fallback after summary failure: result=%#v messages=%#v", res, newMessages)
	}
	if ledger, saves := ledgerStore.snapshot(); saves != 0 || (ledger != nil && len(ledger.Replacements) != 0) {
		t.Fatalf("fallback ledger persisted before checkpoint: saves=%d ledger=%#v", saves, ledger)
	}

	committed, err := ag.CommitCompactionCheckpoint(context.Background(), newMessages, res)
	if err != nil {
		t.Fatalf("CommitCompactionCheckpoint: %v", err)
	}
	if !committed.Compacted || writes != 1 {
		t.Fatalf("fallback checkpoint result=%#v writes=%d", committed, writes)
	}
	ledger, saves := ledgerStore.snapshot()
	if saves != 1 || ledger == nil || len(ledger.Replacements) == 0 {
		t.Fatalf("fallback ledger not committed with checkpoint: saves=%d ledger=%#v", saves, ledger)
	}
}

func TestSuccessfulFullRebuildRefreshesStaleDeferredLedger(t *testing.T) {
	ledgerStore := &checkpointLedgerStore{}
	ag, err := New(Config{
		LLM: &countingCompactionModel{},
		Compaction: &compaction.Config{
			Enabled:     true,
			SessionID:   "refresh-stale-deferred-ledger",
			LedgerStore: ledgerStore,
			CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
				return nil
			}),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	original := []llm.Message{
		llm.NewUserMessage("covered request"),
		llm.NewAssistantMessage("covered answer", nil),
		llm.NewUserMessage("latest retained request"),
	}

	firstMessages, firstRes, err := ag.compactor.Compact(context.Background(), ag.llm, original)
	if err != nil {
		t.Fatalf("first Compact: %v", err)
	}
	if _, err := ag.CommitCompactionCheckpoint(context.Background(), firstMessages, firstRes); err != nil {
		t.Fatalf("first checkpoint commit: %v", err)
	}
	ledgerStore.mu.Lock()
	if ledgerStore.ledger == nil || ledgerStore.ledger.Summary == nil {
		ledgerStore.mu.Unlock()
		t.Fatal("initial checkpoint did not persist summary ledger")
	}
	ledgerStore.ledger.Summary.SummaryHash = compaction.ContentHash("tampered summary")
	ledgerStore.mu.Unlock()

	secondInput := append(append([]llm.Message(nil), firstMessages...), llm.NewAssistantMessage("delta after stale ledger", nil))
	secondMessages, secondRes, err := ag.compactor.Compact(context.Background(), ag.llm, secondInput)
	if err != nil {
		t.Fatalf("second Compact: %v", err)
	}
	if !warningsContain(secondRes.Warnings, "integrity mismatch") || !warningsContain(secondRes.Warnings, "full rebuild") {
		t.Fatalf("stale ledger did not trigger visible full rebuild: %#v", secondRes.Warnings)
	}
	if _, err := ag.CommitCompactionCheckpoint(context.Background(), secondMessages, secondRes); err != nil {
		t.Fatalf("rebuild checkpoint commit: %v", err)
	}

	thirdInput := append(append([]llm.Message(nil), secondMessages...), llm.NewAssistantMessage("delta after repaired ledger", nil))
	_, thirdRes, err := ag.compactor.Compact(context.Background(), ag.llm, thirdInput)
	if err != nil {
		t.Fatalf("third Compact: %v", err)
	}
	if warningsContain(thirdRes.Warnings, "integrity mismatch") {
		t.Fatalf("successful rebuild did not refresh stale ledger: %#v", thirdRes.Warnings)
	}
}

func warningsContain(warnings []string, needle string) bool {
	for _, warning := range warnings {
		if strings.Contains(strings.ToLower(warning), strings.ToLower(needle)) {
			return true
		}
	}
	return false
}

func TestCompactionCheckpointWriterRunsBeforeHistoryReplacement(t *testing.T) {
	original := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewUserMessage("compact this history"),
		llm.NewAssistantMessage("old assistant state", nil),
	}
	var persisted compaction.CompactionCheckpoint
	ag, err := New(Config{
		LLM: &countingCompactionModel{},
		Compaction: &compaction.Config{
			Enabled: true,
			CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(_ context.Context, checkpoint compaction.CompactionCheckpoint) error {
				persisted = checkpoint
				return nil
			}),
		},
		InitialMessages: original,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := ag.CompactNow(context.Background())
	if err != nil {
		t.Fatalf("CompactNow: %v", err)
	}
	if !res.Compacted || res.CheckpointID == "" || persisted.CheckpointID != res.CheckpointID {
		t.Fatalf("checkpoint result=%#v persisted=%#v", res, persisted)
	}
	if err := persisted.Validate(); err != nil {
		t.Fatalf("persisted checkpoint invalid: %v", err)
	}
	if got := ag.Messages(); !reflect.DeepEqual(got, persisted.Messages) {
		t.Fatalf("persisted seed differs from applied history:\nseed=%#v\nhistory=%#v", persisted.Messages, got)
	}
}

func TestAutomaticCompactionCheckpointFailurePreservesHistory(t *testing.T) {
	original := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewUserMessage("active request"),
		llm.NewAssistantMessage("active assistant state", nil),
	}
	writes := 0
	ag, err := New(Config{
		LLM: &countingCompactionModel{},
		Compaction: &compaction.Config{
			Enabled: true,
			CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
				writes++
				return errors.New("automatic checkpoint append failed")
			}),
		},
		InitialMessages: original,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.pendingCompaction = &pendingCompaction{
		messages: []llm.Message{
			llm.NewSystemMessage("system"),
			llm.Message{Role: llm.RoleUser, Name: compaction.CompactionSummaryMessageName, Content: llm.TextContent("summary")},
		},
		snapshotLen: len(original),
		result: compaction.Result{
			Compacted:      true,
			Trigger:        "usage",
			Watermark:      "summarize",
			TiersApplied:   []string{"summarize"},
			OriginalTokens: 900,
			NewTokens:      180,
		},
	}
	out := make(chan Event, 1)
	ag.applyPendingCompaction(out)
	if writes != 1 {
		t.Fatalf("automatic checkpoint writes = %d, want 1", writes)
	}
	if got := ag.Messages(); !reflect.DeepEqual(got, original) {
		t.Fatalf("automatic checkpoint failure mutated history:\n got=%#v\nwant=%#v", got, original)
	}
	if !ag.hasPendingCompaction() || !ag.compactionRetryPending.Load() {
		t.Fatal("automatic checkpoint failure was not retained as retryable pending work")
	}
	select {
	case event := <-out:
		t.Fatalf("automatic checkpoint failure emitted success event: %#v", event)
	default:
	}
}

func containsAgentWarning(warnings []string, needle string) bool {
	for _, warning := range warnings {
		if strings.Contains(warning, needle) {
			return true
		}
	}
	return false
}
