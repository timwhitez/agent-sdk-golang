package agent

import (
	"context"
	"errors"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestD3DiscardedAutomaticSummaryDoesNotWriteLedger(t *testing.T) {
	var high *llm.Usage
	model := newGatedSummaryModel(func() (*llm.Completion, error) {
		return &llm.Completion{Content: llm.TextContent("first answer"), Usage: high}, nil
	})
	store := &checkpointLedgerStore{}
	ag, usage := interleaveAgent(t, model, nil, func(c *Config) {
		c.Compaction.LedgerStore = store
		c.Compaction.SessionID = "review-stale"
	})
	high = usage
	if _, errs, finals := drainQuery(ag, context.Background(), "first request", nil); errs != 0 || finals != 1 {
		t.Fatalf("first turn errors=%d finals=%d", errs, finals)
	}
	<-model.summaryStarted
	replacement := ag.Messages()
	replacement[len(replacement)-1] = llm.NewAssistantMessage("revised answer", nil)
	if err := ag.ReplaceHistoryChecked(replacement); err != nil {
		t.Fatal(err)
	}
	close(model.gate)
	compactions, errs, finals := drainQuery(ag, context.Background(), "second request", nil)
	if compactions != 0 || errs != 0 || finals != 1 {
		t.Fatalf("compactions=%d errors=%d finals=%d", compactions, errs, finals)
	}
	ledger, saves := store.snapshot()
	if saves != 0 {
		if ledger == nil || ledger.Summary == nil {
			t.Fatal("no saved summary metadata")
		}
		t.Fatalf("discarded summary emitted %d compactions but wrote ledger %d time(s); binding against live history: %v", compactions, saves, compaction.ValidateLedgerSummaryBinding(ag.Messages(), ledger.Summary))
	}
}

func TestLedgerOnlyManualCompactionCommitsAtPublication(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "save failure"}[fail], func(t *testing.T) {
			store := &checkpointLedgerStore{}
			if fail {
				store.failSaveAt = map[int]error{1: errors.New("ledger unavailable")}
			}
			ag, err := New(Config{LLM: &countingCompactionModel{}, InitialMessages: longHistory(), Compaction: &compaction.Config{Enabled: true, SessionID: "ledger-only", LedgerStore: store}})
			if err != nil {
				t.Fatal(err)
			}
			source := ag.Messages()
			res, err := ag.CompactNow(context.Background())
			ledger, saves := store.snapshot()
			if saves != 1 || res.CheckpointID != "" {
				t.Fatalf("ledger-only commit: saves=%d result=%+v", saves, res)
			}
			if fail {
				if err == nil || res.Compacted || !messageJSONEqual(source, ag.Messages()) || ledger != nil {
					t.Fatalf("failed ledger write changed state: result=%+v err=%v ledger=%+v", res, err, ledger)
				}
				return
			}
			if err != nil || !res.Compacted || ledger == nil || ledger.Summary == nil {
				t.Fatalf("valid ledger-only compaction failed: result=%+v err=%v ledger=%+v", res, err, ledger)
			}
			if err := compaction.ValidateLedgerSummaryBinding(ag.Messages(), ledger.Summary); err != nil {
				t.Fatal(err)
			}
		})
	}
}

type publicationHookLedgerStore struct {
	checkpointLedgerStore
	onSave func()
}

func (s *publicationHookLedgerStore) Save(ctx context.Context, session string, ledger *compaction.Ledger) error {
	if hook := s.onSave; hook != nil {
		s.onSave = nil
		hook()
	}
	return s.checkpointLedgerStore.Save(ctx, session, ledger)
}

func TestLedgerOnlyPublicationConflictRollsBack(t *testing.T) {
	store := &publicationHookLedgerStore{}
	ag, err := New(Config{LLM: &countingCompactionModel{}, InitialMessages: longHistory(), Compaction: &compaction.Config{Enabled: true, SessionID: "ledger-only-conflict", LedgerStore: store}, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	source := ag.Messages()
	candidate, res, err := ag.compactWithRetry(context.Background(), source, compaction.PipelineRequest{ForceSummary: true, AllowSummary: true})
	if err != nil || !res.Compacted {
		t.Fatalf("prepare candidate: %+v %v", res, err)
	}
	if _, saves := store.snapshot(); saves != 0 {
		t.Fatal("candidate persisted before publication")
	}
	ag.pendingCompaction = &pendingCompaction{messages: candidate, source: source, snapshotLen: len(source), result: res}
	replacement := []llm.Message{llm.NewUserMessage("new branch")}
	store.onSave = func() {
		if err := ag.ReplaceHistoryChecked(replacement); err != nil {
			t.Fatal(err)
		}
	}
	if ag.applyPendingCompaction(nil) {
		t.Fatal("published stale candidate")
	}
	ledger, saves := store.snapshot()
	if saves != 2 || ledger == nil || ledger.Summary != nil || !messageJSONEqual(ag.Messages(), replacement) {
		t.Fatalf("publication conflict did not roll back ledger: saves=%d ledger=%+v history=%+v", saves, ledger, ag.Messages())
	}
}
