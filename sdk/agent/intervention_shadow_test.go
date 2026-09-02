package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestShadowRepeatedSignatureIntervention(t *testing.T) {
	for _, test := range []struct {
		name        string
		observation repeatedSignatureObservation
		want        interventionDecision
	}{
		{
			name:        "below threshold",
			observation: repeatedSignatureObservation{count: 2, threshold: 3, reminderConfigured: true, nextStrike: 1, strikeLimit: 2},
			want:        interventionDecision{detection: interventionDetection{kind: interventionKindRepeatedSignature}, action: interventionActionProceed},
		},
		{
			name:        "threshold with reminder",
			observation: repeatedSignatureObservation{count: 3, threshold: 3, reminderConfigured: true, nextStrike: 1, strikeLimit: 2},
			want:        interventionDecision{detection: interventionDetection{kind: interventionKindRepeatedSignature, active: true}, action: interventionActionSuppressTool, queueReminder: true},
		},
		{
			name:        "threshold without reminder",
			observation: repeatedSignatureObservation{count: 3, threshold: 3, nextStrike: 1, strikeLimit: 2},
			want:        interventionDecision{detection: interventionDetection{kind: interventionKindRepeatedSignature, active: true}, action: interventionActionSuppressTool},
		},
		{
			name:        "exhausted normal repeat proceeds",
			observation: repeatedSignatureObservation{count: 3, threshold: 3, exhausted: true, reminderConfigured: true, nextStrike: 2, strikeLimit: 2},
			want:        interventionDecision{detection: interventionDetection{kind: interventionKindRepeatedSignature, active: true}, action: interventionActionProceed},
		},
		{
			name:        "exhausted recycled repeat is suppressed",
			observation: repeatedSignatureObservation{count: 3, threshold: 3, exhausted: true, lastResultRecycled: true, nextStrike: 2, strikeLimit: 2},
			want:        interventionDecision{detection: interventionDetection{kind: interventionKindRepeatedSignature, active: true}, action: interventionActionSuppressTool, queueReminder: true},
		},
		{
			name:        "strike boundary downgrades",
			observation: repeatedSignatureObservation{count: 3, threshold: 3, reminderConfigured: true, nextStrike: 2, strikeLimit: 2},
			want:        interventionDecision{detection: interventionDetection{kind: interventionKindRepeatedSignature, active: true}, action: interventionActionSuppressTool, queueReminder: true, downgradeGuard: true},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := shadowRepeatedSignatureIntervention(test.observation); got != test.want {
				t.Fatalf("decision=%#v want %#v", got, test.want)
			}
		})
	}
}

func TestObserveRepeatedSignatureInterventionComparesEveryFieldSafely(t *testing.T) {
	legacy := interventionDecision{
		detection:      interventionDetection{kind: interventionKindRepeatedSignature, active: true},
		action:         interventionActionSuppressTool,
		queueReminder:  true,
		downgradeGuard: true,
	}
	for _, test := range []struct {
		name         string
		shadow       interventionDecision
		wantWarnings int
	}{
		{name: "equal", shadow: legacy},
		{name: "kind", shadow: interventionDecision{detection: interventionDetection{active: true}, action: interventionActionSuppressTool, queueReminder: true, downgradeGuard: true}, wantWarnings: 1},
		{name: "active", shadow: interventionDecision{detection: interventionDetection{kind: interventionKindRepeatedSignature}, action: interventionActionSuppressTool, queueReminder: true, downgradeGuard: true}, wantWarnings: 1},
		{name: "action", shadow: interventionDecision{detection: interventionDetection{kind: interventionKindRepeatedSignature, active: true}, queueReminder: true, downgradeGuard: true}, wantWarnings: 1},
		{name: "reminder", shadow: interventionDecision{detection: interventionDetection{kind: interventionKindRepeatedSignature, active: true}, action: interventionActionSuppressTool, downgradeGuard: true}, wantWarnings: 1},
		{name: "downgrade", shadow: interventionDecision{detection: interventionDetection{kind: interventionKindRepeatedSignature, active: true}, action: interventionActionSuppressTool, queueReminder: true}, wantWarnings: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			observations, warnings := 0, 0
			agent := &Agent{
				repeatInterventionShadowObserved: func(interventionDecision, interventionDecision) { observations++ },
				warningf:                         func(string, ...any) { warnings++ },
			}
			agent.observeRepeatedSignatureIntervention(legacy, test.shadow)
			if observations != 1 || warnings != test.wantWarnings {
				t.Fatalf("observations=%d warnings=%d want 1/%d", observations, warnings, test.wantWarnings)
			}
		})
	}

	var warning string
	agent := &Agent{warningf: func(format string, args ...any) { warning = fmt.Sprintf(format, args...) }}
	agent.observeRepeatedSignatureIntervention(
		interventionDecision{detection: interventionDetection{kind: interventionKind(255)}, action: interventionAction(255)},
		legacy,
	)
	if !strings.Contains(warning, "legacy_kind=unknown") || !strings.Contains(warning, "legacy_action=unknown") {
		t.Fatalf("unsafe mismatch warning: %q", warning)
	}
}

func TestRepeatedSignatureInterventionRuntimeObservationCounts(t *testing.T) {
	for _, test := range []struct {
		name      string
		model     *repeatedInterventionRecordingModel
		threshold int
		want      int
	}{
		{name: "enabled", model: &repeatedInterventionRecordingModel{}, threshold: 3, want: 4},
		{name: "terminal provider failure", model: &repeatedInterventionRecordingModel{failAfterIntervention: true}, threshold: 3, want: 3},
		{name: "disabled", model: &repeatedInterventionRecordingModel{}, want: 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			agent, _ := newRepeatedInterventionCharacterizationAgent(t, test.model, test.threshold)
			observations := 0
			agent.repeatInterventionShadowObserved = func(legacy, shadow interventionDecision) {
				observations++
				if legacy != shadow {
					t.Errorf("legacy=%#v shadow=%#v", legacy, shadow)
				}
			}
			collectEvents(agent.QueryStream(context.Background(), llm.TextContent("loop")))
			if observations != test.want {
				t.Fatalf("observations=%d want %d", observations, test.want)
			}
		})
	}
}

func TestRepeatedSignatureInterventionSkipsEvidenceTools(t *testing.T) {
	model := &cancelBoundaryScriptModel{toolCalls: []llm.ToolCall{cancelBoundaryCall("read-1", "read")}}
	read := tools.Func[struct{}]("read", "read", func(context.Context, struct{}, *tools.Container) (any, error) { return "ok", nil })
	agent, err := New(Config{LLM: model, Tools: []tools.Tool{read}, RepeatToolSignatureThreshold: 2})
	if err != nil {
		t.Fatal(err)
	}
	agent.repeatInterventionShadowObserved = func(interventionDecision, interventionDecision) {
		t.Fatal("evidence tool entered repeated-signature intervention shadow")
	}
	collectEvents(agent.QueryStream(context.Background(), llm.TextContent("read")))
}

var benchmarkInterventionDecision interventionDecision

func BenchmarkShadowRepeatedSignatureIntervention(b *testing.B) {
	observation := repeatedSignatureObservation{count: 3, threshold: 3, reminderConfigured: true, nextStrike: 2, strikeLimit: 2}
	legacy := shadowRepeatedSignatureIntervention(observation)
	agent := &Agent{warningf: func(string, ...any) {}}
	b.ReportAllocs()
	var decision interventionDecision
	for i := 0; i < b.N; i++ {
		decision = shadowRepeatedSignatureIntervention(observation)
		agent.observeRepeatedSignatureIntervention(legacy, decision)
	}
	benchmarkInterventionDecision = decision
}
