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

func TestRepeatedSignatureInterventionDecision(t *testing.T) {
	for _, test := range []struct {
		name        string
		observation repeatedSignatureObservation
		want        interventionDecision
	}{
		{
			name:        "below threshold",
			observation: repeatedSignatureObservation{count: 2, threshold: 3, reminderConfigured: true, nextStrike: 1, strikeLimit: 2},
			want:        interventionDecision{action: interventionActionProceed},
		},
		{
			name:        "threshold with reminder",
			observation: repeatedSignatureObservation{count: 3, threshold: 3, reminderConfigured: true, nextStrike: 1, strikeLimit: 2},
			want:        interventionDecision{action: interventionActionSuppressTool, queueReminder: true},
		},
		{
			name:        "threshold without reminder",
			observation: repeatedSignatureObservation{count: 3, threshold: 3, nextStrike: 1, strikeLimit: 2},
			want:        interventionDecision{action: interventionActionSuppressTool},
		},
		{
			name:        "exhausted normal repeat proceeds",
			observation: repeatedSignatureObservation{count: 3, threshold: 3, exhausted: true, reminderConfigured: true, nextStrike: 2, strikeLimit: 2},
			want:        interventionDecision{action: interventionActionProceed},
		},
		{
			name:        "exhausted recycled repeat is suppressed",
			observation: repeatedSignatureObservation{count: 3, threshold: 3, exhausted: true, lastResultRecycled: true, nextStrike: 2, strikeLimit: 2},
			want:        interventionDecision{action: interventionActionSuppressTool, queueReminder: true},
		},
		{
			name:        "strike boundary downgrades",
			observation: repeatedSignatureObservation{count: 3, threshold: 3, reminderConfigured: true, nextStrike: 2, strikeLimit: 2},
			want:        interventionDecision{action: interventionActionSuppressTool, queueReminder: true, downgradeGuard: true},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := decideRepeatedSignatureIntervention(test.observation); got != test.want {
				t.Fatalf("decision=%#v want %#v", got, test.want)
			}
		})
	}
}

func TestRepeatedSignatureInterventionRuntimeBoundaries(t *testing.T) {
	for _, test := range []struct {
		name           string
		recycled       bool
		wantSuppressed int
		wantReminders  int
		wantEchoCalls  int
	}{
		{name: "exhausted normal proceeds", wantSuppressed: 1, wantReminders: 1, wantEchoCalls: 3},
		{name: "exhausted recycled suppresses", recycled: true, wantSuppressed: 2, wantReminders: 2, wantEchoCalls: 2},
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
			events := collectEvents(agent.QueryStream(context.Background(), llm.TextContent("loop")))
			suppressed, reminders, downgrades := 0, 0, 0
			for _, event := range events {
				switch event := event.(type) {
				case ToolResultEvent:
					if event.Metadata["loop_guard_suppressed"] == true {
						suppressed++
					}
				case HiddenUserMessageEvent:
					reminders++
				case WarnEvent:
					if event.Kind == "loop_guard" && strings.Contains(event.Message, "budget spent") {
						downgrades++
					}
				}
			}
			if lookups != 1 || echoCalls != test.wantEchoCalls || suppressed != test.wantSuppressed || reminders != test.wantReminders || downgrades != 1 {
				t.Fatalf("lookups/calls/suppressed/reminders/downgrades=%d/%d/%d/%d/%d want 1/%d/%d/%d/1", lookups, echoCalls, suppressed, reminders, downgrades, test.wantEchoCalls, test.wantSuppressed, test.wantReminders)
			}
			assertContiguousToolResults(t, agent.Messages())
		})
	}
}

func TestRepeatedSignatureInterventionWithoutReminder(t *testing.T) {
	model := &repeatedInterventionRecordingModel{}
	ag, calls := newRepeatedInterventionCharacterizationAgent(t, model, 3)
	ag.loopGuardUserMsg = "  "
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("loop")))
	if *calls != 2 || len(model.requests) != 4 {
		t.Fatalf("calls=%d requests=%d", *calls, len(model.requests))
	}
	for _, event := range events {
		if _, ok := event.(HiddenUserMessageEvent); ok {
			t.Fatal("unexpected reminder")
		}
	}
	want := repeatedInterventionExpectedRequests(true)
	want[3] = want[3][:len(want[3])-1]
	for i, request := range model.requests {
		if got := interventionRequestTranscript(request); !slices.Equal(got, want[i]) {
			t.Fatalf("request[%d]=%v want %v", i, got, want[i])
		}
	}
	assertContiguousToolResults(t, ag.Messages())
}

func TestRepeatedSignatureInterventionSkipsEvidenceTools(t *testing.T) {
	model := &cancelBoundaryScriptModel{toolCalls: []llm.ToolCall{cancelBoundaryCall("read-1", "read")}}
	calls := 0
	read := tools.Func[struct{}]("read", "read", func(context.Context, struct{}, *tools.Container) (any, error) { calls++; return "ok", nil })
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{read}, RepeatToolSignatureThreshold: 1})
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("read")))
	if calls != 1 {
		t.Fatalf("read calls=%d want 1", calls)
	}
	for _, event := range events {
		if e, ok := event.(ToolResultEvent); ok && e.Metadata["loop_guard_suppressed"] == true {
			t.Fatal("evidence tool was suppressed by repeat guard")
		}
	}
	assertContiguousToolResults(t, ag.Messages())
}
