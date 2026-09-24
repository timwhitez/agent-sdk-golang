package agent

import (
	"context"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// interventionEnvelopes returns every envelope carrying intervention labels.
func interventionEnvelopes(envs []EventEnvelope) []EventEnvelope {
	var out []EventEnvelope
	for _, env := range envs {
		if env.Intervention != "" || env.InterventionResult != "" || env.InterventionStage != "" {
			out = append(out, env)
		}
	}
	return out
}

// #87/#88: a stream-idle recovery reports its applied lifecycle once, on a
// warning emitted after the recovery reminder was appended to history.
func TestStreamIdleRecoveryReportsAppliedIntervention(t *testing.T) {
	origTimeout, origRecoveries := agentStreamIdleTimeout, agentStreamIdleMaxRecoveries
	agentStreamIdleTimeout, agentStreamIdleMaxRecoveries = 20*time.Millisecond, 2
	t.Cleanup(func() { agentStreamIdleTimeout, agentStreamIdleMaxRecoveries = origTimeout, origRecoveries })

	ag, err := New(Config{LLM: &streamIdleRecoveryModel{}, StreamIdleMaxRecoveries: -1})
	if err != nil {
		t.Fatal(err)
	}
	// The observation hook runs synchronously where the driver applies the
	// intervention, immediately before it emits the labeled warning.
	historyHadReminder, observed := false, 0
	ag.interventionObserved = func(record interventionRecord) {
		if record.kind != InterventionStreamIdleRecovery || record.stage != interventionApplied {
			return
		}
		observed++
		for _, m := range ag.Messages() {
			historyHadReminder = historyHadReminder || m.Content.PlainText() == streamIdleRecoveryText
		}
	}
	var envs []EventEnvelope
	for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("hello")) {
		envs = append(envs, env)
	}
	if observed != 1 {
		t.Fatalf("applied observations=%d", observed)
	}
	labeled := interventionEnvelopes(envs)
	if len(labeled) != 1 {
		t.Fatalf("labeled envelopes=%d: %+v", len(labeled), labeled)
	}
	got := labeled[0]
	if w, ok := got.Event.(WarnEvent); !ok || w.Kind != "stream_idle_recovery" ||
		got.Intervention != InterventionStreamIdleRecovery || got.InterventionStage != InterventionStageApplied ||
		got.InterventionResult != InterventionResultRecoveryReminderAppended || got.InterventionStrike != 1 {
		t.Fatalf("envelope=%+v", got)
	}
	if !historyHadReminder {
		t.Fatal("applied label emitted before the reminder entered history")
	}
}

// An idle stream with recovery disabled reports no intervention.
func TestStreamIdleWithoutRecoveryReportsNothing(t *testing.T) {
	origTimeout := agentStreamIdleTimeout
	agentStreamIdleTimeout = 20 * time.Millisecond
	t.Cleanup(func() { agentStreamIdleTimeout = origTimeout })
	ag, err := New(Config{LLM: &streamIdleForeverModel{}, StreamIdleMaxRecoveries: 0})
	if err != nil {
		t.Fatal(err)
	}
	var envs []EventEnvelope
	for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("hello")) {
		envs = append(envs, env)
	}
	if labeled := interventionEnvelopes(envs); len(labeled) != 0 {
		t.Fatalf("labels without recovery: %+v", labeled)
	}
}

// A typed overflow recovery reports its applied lifecycle on the existing
// recovery warning, after compaction changed history; a terminal overflow
// reports none.
func TestContextOverflowRecoveryReportsAppliedIntervention(t *testing.T) {
	collect := func(script ...func() (*llm.Completion, error)) []EventEnvelope {
		ag := overflowAgent(t, &overflowScriptModel{script: script}, nil)
		before := len(ag.Messages())
		ag.interventionObserved = func(record interventionRecord) {
			// Applied means the compacted history is already in place.
			if record.kind == InterventionContextOverflowRecovery && len(ag.Messages()) >= before {
				t.Errorf("overflow recovery observed before history was compacted")
			}
		}
		var envs []EventEnvelope
		for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("current request")) {
			envs = append(envs, env)
		}
		return envs
	}
	overflow := func() (*llm.Completion, error) { return nil, typedOverflow() }

	labeled := interventionEnvelopes(collect(overflow))
	if len(labeled) != 1 {
		t.Fatalf("labeled envelopes=%d: %+v", len(labeled), labeled)
	}
	got := labeled[0]
	if w, ok := got.Event.(WarnEvent); !ok || w.Kind != "context_overflow_recovery" ||
		got.Intervention != InterventionContextOverflowRecovery || got.InterventionStage != InterventionStageApplied ||
		got.InterventionResult != InterventionResultHistoryCompacted || got.InterventionStrike != 1 {
		t.Fatalf("envelope=%+v", got)
	}

	// Second rejection in the same epoch: one applied recovery, then terminal.
	if labeled := interventionEnvelopes(collect(overflow, overflow)); len(labeled) != 1 {
		t.Fatalf("terminal overflow added labels: %+v", labeled)
	}
	// Untyped 400: no recovery, no label.
	untyped := func() (*llm.Completion, error) {
		return nil, &llm.ProviderError{Provider: "fixture", StatusCode: 400, Message: "input too long"}
	}
	if labeled := interventionEnvelopes(collect(untyped)); len(labeled) != 0 {
		t.Fatalf("untyped error labeled: %+v", labeled)
	}
}
