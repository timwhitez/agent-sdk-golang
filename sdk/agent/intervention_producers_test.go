package agent

import (
	"context"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// The RequireDone safety fallback reports its applied intervention on the
// warning and on the partial final response; the strike is the number of
// reminders appended to history before it.
func TestRequireDoneSafetyFallbackReportsAppliedIntervention(t *testing.T) {
	echo := tools.Func[struct {
		Message string `json:"message"`
	}]("echo", "echo", func(context.Context, struct {
		Message string `json:"message"`
	}, *tools.Container) (any, error) {
		return "ok", nil
	})
	ag, err := New(Config{LLM: &requireDoneSafetyValveModel{}, Tools: []tools.Tool{echo}, MaxIterations: 20, RequireDoneTool: true, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	var warn, final *EventEnvelope
	reported := 0
	for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("do something")) {
		env := env
		if env.Intervention != "" {
			reported++
		}
		switch e := env.Event.(type) {
		case WarnEvent:
			if e.Kind == "require_done_safety" {
				warn = &env
			}
		case FinalResponseEvent:
			final = &env
		}
	}
	for _, env := range []*EventEnvelope{warn, final} {
		if env == nil || env.Intervention != InterventionRequireDone || env.InterventionStage != InterventionStageApplied || env.InterventionResult != InterventionResultSafetyFallback || env.InterventionStrike != uint64(defaultRequireDoneMaxReminders) || env.FrameID == "" {
			t.Fatalf("envelope=%+v", env)
		}
	}
	if reported != 2 {
		t.Fatalf("intervention reported on %d events, want only the fallback warning and final", reported)
	}
}

// A normal RequireDone completion reports no intervention.
func TestRequireDoneWithoutFallbackReportsNothing(t *testing.T) {
	calls := 0
	ag := newInterventionLifecycleAgent(t, &calls)
	ag.repeatSigThreshold = 0
	for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("hi")) {
		if env.Intervention == InterventionRequireDone {
			t.Fatalf("unexpected require-done intervention: %+v", env)
		}
	}
}

// Evidence-progress suppression reports tool_suppressed on the suppressed
// result (published after its commit) and reminder_queued on the recovery
// reminder and warning, with the same strike; other results carry nothing.
func TestEvidenceProgressSuppressionReportsAppliedIntervention(t *testing.T) {
	readTool := tools.Func[evidenceReadArgs]("read", "read", func(_ context.Context, args evidenceReadArgs, _ *tools.Container) (any, error) {
		if args.Offset >= 101 {
			return "101: new block", nil
		}
		return "1: same block", nil
	})
	readAlias := readTool
	readAlias.Name = "read_file"
	doneTool := tools.Func[evidenceDoneArgs]("done", "done", func(_ context.Context, args evidenceDoneArgs, _ *tools.Container) (any, error) {
		return nil, &tools.TaskCompleteError{Message: args.Message}
	})
	ag, err := New(Config{LLM: &evidenceFixtureModel{}, Tools: []tools.Tool{readTool, readAlias, doneTool}, MaxIterations: -1, RequireDoneTool: true, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	suppressed, queued, plainResults := 0, 0, 0
	for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("inspect")) {
		switch e := env.Event.(type) {
		case ToolResultEvent:
			if e.Metadata != nil && e.Metadata["no_progress_suppressed"] == true {
				if env.Intervention != InterventionEvidenceProgress || env.InterventionResult != InterventionResultToolSuppressed || env.InterventionStrike != 1 || env.ToolCallOrdinal == 0 {
					t.Fatalf("suppressed result envelope=%+v", env)
				}
				suppressed++
			} else if env.Intervention != "" {
				t.Fatalf("executed result reports intervention: %+v", env)
			} else {
				plainResults++
			}
		case HiddenUserMessageEvent, WarnEvent:
			if env.Intervention == InterventionEvidenceProgress {
				if env.InterventionResult != InterventionResultReminderQueued || env.InterventionStrike != 1 {
					t.Fatalf("reminder envelope=%+v", env)
				}
				queued++
			}
		}
	}
	if suppressed != 1 || queued != 2 || plainResults == 0 {
		t.Fatalf("suppressed=%d queued=%d plain=%d", suppressed, queued, plainResults)
	}
}
