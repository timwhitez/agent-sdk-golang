package agent

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type repeatedInterventionBoundaryModel struct {
	calls int
}

func (m *repeatedInterventionBoundaryModel) Provider() string { return "fixture" }
func (m *repeatedInterventionBoundaryModel) Model() string    { return "repeat-boundary" }
func (m *repeatedInterventionBoundaryModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls <= 4 {
		return &llm.Completion{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{{
			ID:       fmt.Sprintf("echo-%d", m.calls),
			Type:     "function",
			Function: llm.FunctionCall{Name: "echo", Arguments: `{"text":"repeat"}`},
		}}}, nil
	}
	return &llm.Completion{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{{
		ID:       "done-5",
		Type:     "function",
		Function: llm.FunctionCall{Name: "done", Arguments: `{"message":"finished"}`},
	}}}, nil
}

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

func TestRepeatedSignatureInterventionRuntimeBoundaries(t *testing.T) {
	for _, test := range []struct {
		name               string
		recycled           bool
		wantFourthAction   interventionAction
		wantFourthReminder bool
		wantEchoCalls      int
	}{
		{name: "exhausted normal proceeds", wantFourthAction: interventionActionProceed, wantEchoCalls: 3},
		{name: "exhausted recycled suppresses", recycled: true, wantFourthAction: interventionActionSuppressTool, wantFourthReminder: true, wantEchoCalls: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := &repeatedInterventionBoundaryModel{}
			echoCalls := 0
			echo := tools.Func[struct {
				Text string `json:"text"`
			}]("echo", "echo", func(context.Context, struct {
				Text string `json:"text"`
			}, *tools.Container) (any, error) {
				echoCalls++
				return "ok", nil
			})
			done := tools.Func[struct {
				Message string `json:"message"`
			}]("done", "done", func(_ context.Context, args struct {
				Message string `json:"message"`
			}, _ *tools.Container) (any, error) {
				return nil, tools.TaskComplete(args.Message)
			})
			agent, err := New(Config{
				LLM:                          model,
				Tools:                        []tools.Tool{echo, done},
				MaxIterations:                10,
				RepeatToolSignatureThreshold: 2,
				RepeatToolSignatureWindow:    4,
				LoopGuardStrikeThreshold:     1,
				LoopGuardUserMessage:         "stop repeating",
				Warningf:                     failOnToolBlockShadowWarning(t),
			})
			if err != nil {
				t.Fatal(err)
			}
			lookups := 0
			agent.repeatResultRecycled = func(string) bool {
				lookups++
				return test.recycled
			}
			var decisions []interventionDecision
			agent.repeatInterventionShadowObserved = func(legacy, shadow interventionDecision) {
				if legacy != shadow {
					t.Errorf("legacy=%#v shadow=%#v", legacy, shadow)
				}
				decisions = append(decisions, legacy)
			}
			collectEvents(agent.QueryStream(context.Background(), llm.TextContent("loop")))
			if len(decisions) != 5 || lookups != 1 || echoCalls != test.wantEchoCalls {
				t.Fatalf("decisions=%d lookups=%d echo_calls=%d want 5/1/%d", len(decisions), lookups, echoCalls, test.wantEchoCalls)
			}
			if !decisions[1].downgradeGuard || decisions[1].action != interventionActionSuppressTool {
				t.Fatalf("strike-boundary decision=%#v", decisions[1])
			}
			if decisions[3].action != test.wantFourthAction || decisions[3].queueReminder != test.wantFourthReminder {
				t.Fatalf("exhausted decision=%#v want action=%v reminder=%t", decisions[3], test.wantFourthAction, test.wantFourthReminder)
			}
		})
	}
}

func TestRepeatedSignatureInterventionShadowMismatchDoesNotChangeLegacyApplication(t *testing.T) {
	model := &repeatedInterventionRecordingModel{}
	agent, echoCalls := newRepeatedInterventionCharacterizationAgent(t, model, 3)
	agent.repeatInterventionShadowEvaluator = func(observation repeatedSignatureObservation) interventionDecision {
		decision := shadowRepeatedSignatureIntervention(observation)
		if decision.action == interventionActionSuppressTool {
			decision.action = interventionActionProceed
		}
		return decision
	}
	warnings := 0
	previousWarningf := agent.warningf
	agent.warningf = func(format string, args ...any) {
		if strings.Contains(format, "repeated-signature intervention shadow mismatch") {
			warnings++
			return
		}
		previousWarningf(format, args...)
	}
	events := collectEvents(agent.QueryStream(context.Background(), llm.TextContent("loop")))

	if *echoCalls != 2 || warnings != 1 || len(model.requests) != 4 {
		t.Fatalf("echo_calls=%d warnings=%d requests=%d want 2/1/4", *echoCalls, warnings, len(model.requests))
	}
	want := repeatedInterventionExpectedRequests(true)
	for i, request := range model.requests {
		if got := interventionRequestTranscript(request); !slices.Equal(got, want[i]) {
			t.Fatalf("request[%d] transcript=%#v want %#v", i, got, want[i])
		}
	}
	wantOrder := []string{"hidden", "warn", "step_start", "tool_call", "tool_result", "accounting", "step_complete"}
	if got := repeatedInterventionEventOrder(events, "call-3"); !slices.Equal(got, wantOrder) {
		t.Fatalf("event order=%v want %v", got, wantOrder)
	}
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
