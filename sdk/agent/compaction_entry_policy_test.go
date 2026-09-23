package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func entryPolicyHistory() []llm.Message {
	return []llm.Message{
		llm.NewSystemMessage("base"),
		llm.NewUserMessage(strings.Repeat("older context ", 40)),
		llm.NewAssistantMessage(strings.Repeat("older answer ", 40), nil),
		llm.NewUserMessage("latest"),
	}
}

func newEntryPolicyAgent(t *testing.T, model llm.ChatModel) *Agent {
	t.Helper()
	ag, err := New(Config{LLM: model, InitialMessages: entryPolicyHistory(), Warningf: func(string, ...any) {}, Compaction: &compaction.Config{
		Enabled: true, ContextWindow: 100, ThresholdRatio: 0.85, ProtectedRecentMessages: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	return ag
}

// The asynchronous executor uses the summary admission sampled with the
// decision; a failure streak that changes after the decision cannot flip it.
func TestCompactionExecutorUsesSampledSummaryAdmission(t *testing.T) {
	for _, test := range []struct {
		name          string
		sampled       bool
		streakAfter   uint64
		wantSummaries bool
	}{
		{name: "sampled allow survives later suppression", sampled: true, streakAfter: compactionSummaryDisableStreak, wantSummaries: true},
		{name: "sampled deny survives later recovery", sampled: false, streakAfter: 0, wantSummaries: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := &countingCompactionModel{}
			ag := newEntryPolicyAgent(t, model)
			usage := llm.WithPromptEstimate(nil, 90)
			decision := compactionDecision{run: true, trigger: "usage", targetWatermark: "summarize", allowSummary: test.sampled}
			// State the executor would have re-read before this change.
			ag.compactionFailureStreak.Store(test.streakAfter)
			if !test.sampled {
				ag.compactionFailureStreak.Store(0)
			}
			snapshot := ag.Messages()
			ag.compactionInFlight.Store(true)
			ag.runCompactionAsync(context.Background(), snapshot, len(snapshot), usage, usage, decision)
			if got := model.Calls() > 0; got != test.wantSummaries {
				t.Fatalf("summary calls=%d, want summaries=%v", model.Calls(), test.wantSummaries)
			}
		})
	}
}

// The decision is the single place each entry's policy is expressed.
func TestCompactionEntryPolicies(t *testing.T) {
	manualRequest := compaction.PipelineRequest{Trigger: "manual", TargetWatermark: "summarize", AllowSummary: true, ForceSummary: true, AdditionalTokens: 7}
	manual := manualCompactionDecision(manualRequest)
	if !manual.run || !manual.rejectsOverlap() || manual.recordsOutcome() || manual.pipelineRequest(nil) != manualRequest {
		t.Fatalf("manual decision=%+v", manual)
	}
	if todo, retry := manual.clearsPending(); !todo || !retry {
		t.Fatalf("manual clears todo=%v retry=%v", todo, retry)
	}
	for _, test := range []struct {
		trigger     string
		todo, retry bool
	}{{"usage", false, false}, {"todo_checkpoint", true, false}, {"retry_checkpoint", false, true}} {
		automatic := compactionDecision{run: true, trigger: test.trigger, targetWatermark: "snip", allowSummary: true}
		if automatic.rejectsOverlap() || !automatic.recordsOutcome() {
			t.Fatalf("automatic decision=%+v", automatic)
		}
		if todo, retry := automatic.clearsPending(); todo != test.todo || retry != test.retry {
			t.Fatalf("%s clears todo=%v retry=%v", test.trigger, todo, retry)
		}
		usage := llm.WithPromptEstimate(nil, 70)
		if got := automatic.pipelineRequest(usage); got.Trigger != test.trigger || got.Usage != usage || !got.AllowSummary || got.ForceSummary {
			t.Fatalf("automatic request=%+v", got)
		}
	}
}

type failingCompactionModel struct {
	mu    sync.Mutex
	calls int
}

func (m *failingCompactionModel) Provider() string { return "stub" }
func (m *failingCompactionModel) Model() string    { return "stub" }
func (m *failingCompactionModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	m.calls++
	m.mu.Unlock()
	return nil, errors.New("summary unavailable")
}

// Manual/preflight runs never inherit or feed the automatic suppression:
// they run while automatic compaction is cooling down, leave the failure
// streak and cooldown untouched on success or failure, clear pending work on
// success, and reject overlap with ErrAgentBusy.
func TestManualCompactionEntryPolicy(t *testing.T) {
	cooldown := time.Now().Add(time.Hour).UnixNano()

	model := &countingCompactionModel{}
	ag := newEntryPolicyAgent(t, model)
	ag.compactionFailureStreak.Store(compactionSummaryDisableStreak)
	ag.compactionCooldownUntil.Store(cooldown)
	ag.todoCompactionPending.Store(true)
	ag.compactionRetryPending.Store(true)
	if _, err := ag.CompactNow(context.Background()); err != nil {
		t.Fatal(err)
	}
	if model.Calls() == 0 {
		t.Fatal("manual force-summary inherited automatic summary suppression")
	}
	if ag.compactionFailureStreak.Load() != compactionSummaryDisableStreak || ag.compactionCooldownUntil.Load() != cooldown {
		t.Fatal("manual success reset the automatic failure streak/cooldown")
	}
	if ag.todoCompactionPending.Load() || ag.compactionRetryPending.Load() {
		t.Fatal("manual success left pending work")
	}

	failing := &failingCompactionModel{}
	ag = newEntryPolicyAgent(t, failing)
	ag.compactionCooldownUntil.Store(cooldown)
	if _, err := ag.CompactNow(context.Background()); err == nil {
		t.Fatal("expected manual summary failure")
	}
	if ag.compactionFailureStreak.Load() != 0 || ag.compactionCooldownUntil.Load() != cooldown {
		t.Fatal("manual failure fed the automatic failure streak/cooldown")
	}

	ag = newEntryPolicyAgent(t, &countingCompactionModel{})
	ag.compactionInFlight.Store(true)
	if _, err := ag.CompactPipelineNow(context.Background(), compaction.PipelineRequest{Trigger: "preflight", TargetWatermark: "snip"}); !errors.Is(err, ErrAgentBusy) {
		t.Fatalf("overlap err=%v, want ErrAgentBusy", err)
	}
}
