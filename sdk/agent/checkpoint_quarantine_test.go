package agent

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// Q02: a public configuration update (threshold change, disable/enable, a
// new writer pointer) never releases a store quarantine: the next manual
// commit is refused before the writer, and automatic compaction pays for no
// summary.
func TestQ02ConfigUpdatesKeepCheckpointQuarantine(t *testing.T) {
	writer := &outcomeWriter{fail: func(n int) error {
		if n == 1 {
			return indeterminateAppend{}
		}
		return nil
	}}
	store := &checkpointLedgerStore{}
	ag := outcomeAgent(t, writer, store, nil)
	if _, _, err := commitCandidate(t, ag); !compaction.CheckpointOutcomeIsUnknown(err) {
		t.Fatalf("first commit err=%v", err)
	}
	// Ordered: disable happens before re-enable.
	for _, step := range []struct {
		name string
		cfg  *compaction.Config
	}{
		{"threshold", &compaction.Config{Enabled: true, SessionID: "outcome", LedgerStore: store, CheckpointWriter: writer, ThresholdRatio: 0.5}},
		{"disabled", &compaction.Config{Enabled: false}},
		{"re-enabled", &compaction.Config{Enabled: true, SessionID: "outcome", LedgerStore: store, CheckpointWriter: writer}},
		{"new writer", &compaction.Config{Enabled: true, SessionID: "outcome", LedgerStore: store, CheckpointWriter: &outcomeWriter{}}},
	} {
		name, cfg := step.name, step.cfg
		ag.UpdateCompactionConfig(cfg)
		if !cfg.Enabled {
			continue
		}
		if _, out, err := commitCandidate(t, ag); !errors.Is(err, compaction.ErrCheckpointStoreQuarantined) || out.Compacted {
			t.Fatalf("after %s update: err=%v compacted=%v", name, err, out.Compacted)
		}
	}
	if writer.count() != 1 {
		t.Fatalf("writes=%d after updates", writer.count())
	}
	if ledger, _ := store.snapshot(); ledger == nil || ledger.Summary == nil {
		t.Fatal("ledger rolled back")
	}
}

// Q03: only the host's explicit reconciliation releases the quarantine; the
// failure already reported is not rewritten.
func TestQ03ExplicitReconciliationReleasesQuarantine(t *testing.T) {
	writer := &outcomeWriter{fail: func(n int) error {
		if n == 1 {
			return indeterminateAppend{}
		}
		return nil
	}}
	ag := outcomeAgent(t, writer, &checkpointLedgerStore{}, nil)
	_, first, err := commitCandidate(t, ag)
	if !compaction.CheckpointOutcomeIsUnknown(err) || first.Compacted {
		t.Fatalf("first commit err=%v", err)
	}
	warnings := strings.Join(first.Warnings, "\n")
	ag.CheckpointStoreReconciled()
	if _, out, err := commitCandidate(t, ag); err != nil || !out.Compacted || writer.count() != 2 {
		t.Fatalf("after reconciliation: err=%v compacted=%v writes=%d", err, out.Compacted, writer.count())
	}
	if first.Compacted || strings.Join(first.Warnings, "\n") != warnings || !strings.Contains(warnings, "outcome unknown") {
		t.Fatal("the earlier failure record changed")
	}
}

// falseMarker declares "not unknown" for itself only.
type falseMarker struct{ inner error }

func (e falseMarker) Error() string                  { return "false marker" }
func (e falseMarker) Unwrap() error                  { return e.inner }
func (e falseMarker) CheckpointOutcomeUnknown() bool { return false }

// Q04: any explicit positive marker in the error tree means unknown; a false
// marker speaks only for itself, whatever the wrapping or join order.
func TestQ04UnknownOutcomeIsConservativeOverTheErrorTree(t *testing.T) {
	positive := indeterminateAppend{}
	plain := errors.New("plain")
	for name, tc := range map[string]struct {
		err  error
		want bool
	}{
		"direct true":             {positive, true},
		"wrapped true":            {fmt.Errorf("a: %w", positive), true},
		"double wrapped true":     {fmt.Errorf("b: %w", fmt.Errorf("a: %w", positive)), true},
		"false wrapping true":     {falseMarker{inner: positive}, true},
		"join false then true":    {errors.Join(falseMarker{}, positive), true},
		"join true then false":    {errors.Join(positive, falseMarker{}), true},
		"wrapped join":            {fmt.Errorf("c: %w", errors.Join(plain, falseMarker{inner: fmt.Errorf("d: %w", positive)})), true},
		"canceled joined with it": {errors.Join(context.Canceled, positive), true},
		"false only":              {falseMarker{}, false},
		"false wrapping plain":    {falseMarker{inner: plain}, false},
		"plain":                   {plain, false},
		"nil":                     {nil, false},
	} {
		if got := compaction.CheckpointOutcomeIsUnknown(tc.err); got != tc.want {
			t.Errorf("%s: got %v want %v", name, got, tc.want)
		}
	}
}

// Q05: cancellation reported together with a positive unknown does not turn
// the write into "not written": the ledger is kept and the store quarantined.
func TestQ05CancelWithUnknownStaysUnknown(t *testing.T) {
	writer := &outcomeWriter{fail: func(int) error { return errors.Join(context.Canceled, indeterminateAppend{}) }}
	store := &checkpointLedgerStore{}
	ag := outcomeAgent(t, writer, store, nil)
	_, out, err := commitCandidate(t, ag)
	if !compaction.CheckpointOutcomeIsUnknown(err) || !errors.Is(err, context.Canceled) || out.Compacted {
		t.Fatalf("err=%v compacted=%v", err, out.Compacted)
	}
	if ledger, _ := store.snapshot(); ledger == nil || ledger.Summary == nil {
		t.Fatal("ledger rolled back on cancel+unknown")
	}
	if _, _, err := commitCandidate(t, ag); !errors.Is(err, compaction.ErrCheckpointStoreQuarantined) || writer.count() != 1 {
		t.Fatalf("second commit err=%v writes=%d", err, writer.count())
	}
	_ = llm.Message{}
}

// Q02 (automatic): after an unknown outcome and a public threshold update of
// the same store, automatic compaction neither writes nor pays for a summary.
func TestQ02AutomaticCompactionStaysQuarantinedAfterUpdate(t *testing.T) {
	writer := &outcomeWriter{fail: func(int) error { return indeterminateAppend{} }}
	store := &checkpointLedgerStore{}
	history := []llm.Message{llm.NewSystemMessage("base")}
	for i := 0; i < 6; i++ {
		history = append(history, llm.NewUserMessage(strings.Repeat("old ", 600)), llm.NewAssistantMessage(strings.Repeat("reply ", 600), nil))
	}
	history = append(history, llm.NewUserMessage("latest"))
	total := compaction.NewService(&compaction.Config{Enabled: true}).EstimateMessages(history)
	config := func(threshold float64) *compaction.Config {
		return &compaction.Config{Enabled: true, SessionID: "outcome", LedgerStore: store, CheckpointWriter: writer,
			ContextWindow: 2 * total, ReserveOutputTokens: 1, ThresholdRatio: threshold, SnipThresholdRatio: 0.3, PruneThresholdRatio: 0.35, KeepRecentUserMessages: 1}
	}
	model := &summaryCountingModel{}
	ag, err := New(Config{LLM: model, InitialMessages: history, Warningf: func(string, ...any) {}, Compaction: config(0.4)})
	if err != nil {
		t.Fatal(err)
	}
	f := ineffectiveSummaryFixture{agent: ag}
	high := &llm.Usage{PromptTokens: total, TotalTokens: total}
	f.decide(t, high)
	if writer.count() != 1 {
		t.Fatalf("writes=%d", writer.count())
	}
	summaries := model.Calls()
	ag.UpdateCompactionConfig(config(0.39))
	f.decide(t, high)
	f.decide(t, high)
	if writer.count() != 1 || model.Calls() != summaries || !reflect.DeepEqual(history, ag.Messages()) {
		t.Fatalf("after update: writes=%d summaries=%d→%d", writer.count(), summaries, model.Calls())
	}
}
