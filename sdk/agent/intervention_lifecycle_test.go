package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func newInterventionLifecycleAgent(t *testing.T, echoCalls *int) *Agent {
	t.Helper()
	echo := tools.Func[struct {
		Text string `json:"text"`
	}]("echo", "echo", func(context.Context, struct {
		Text string `json:"text"`
	}, *tools.Container) (any, error) {
		*echoCalls++
		return "ok", nil
	})
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "done", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})
	ag, err := New(Config{
		LLM:                          &repeatedInterventionBoundaryModel{},
		Tools:                        []tools.Tool{echo, done},
		MaxIterations:                10,
		RepeatToolSignatureThreshold: 2,
		RepeatToolSignatureWindow:    4,
		LoopGuardStrikeThreshold:     1,
		LoopGuardUserMessage:         "stop repeating",
		Warningf:                     func(string, ...any) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Recycled placeholders keep the downgraded guard suppressing, so the
	// fixture applies the intervention twice in one Query.
	ag.repeatResultRecycled = func(string) bool { return true }
	return ag
}

func suppressedHistoryResults(messages []llm.Message) int {
	count := 0
	for _, message := range messages {
		if message.Role == llm.RoleTool && strings.Contains(message.Content.PlainText(), "skipped by loop guard") {
			count++
		}
	}
	return count
}

// Detection proposes; only a committed suppressed result makes the
// intervention applied. Applied envelopes report fixed labels and a strike
// consumed at commit, tool identity stays on tool events, and an internal
// reminder does not reset the Query's strike count.
func TestInterventionAppliedOnlyAfterAcceptedMutation(t *testing.T) {
	echoCalls := 0
	ag := newInterventionLifecycleAgent(t, &echoCalls)
	var records []interventionRecord
	var historyAtApply []int
	ag.interventionObserved = func(record interventionRecord) {
		records = append(records, record)
		if record.stage == interventionApplied {
			historyAtApply = append(historyAtApply, suppressedHistoryResults(ag.Messages()))
		}
	}
	var envelopes []EventEnvelope
	for envelope := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("loop")) {
		envelopes = append(envelopes, envelope)
	}

	proposed, applied := 0, 0
	for _, record := range records {
		switch record.stage {
		case interventionProposed:
			proposed++
		case interventionApplied:
			applied++
			if record.strike != applied || !record.reminderQueued {
				t.Fatalf("applied record=%+v", record)
			}
		}
	}
	if proposed != 2 || applied != 2 {
		t.Fatalf("proposed=%d applied=%d records=%+v", proposed, applied, records)
	}
	// The suppressed result is already in history when the application is
	// reported: strike k sees k committed suppressed results.
	if len(historyAtApply) != 2 || historyAtApply[0] != 1 || historyAtApply[1] != 2 {
		t.Fatalf("suppressed history results at apply=%v", historyAtApply)
	}

	results := map[string][]uint64{}
	for _, envelope := range envelopes {
		if envelope.Intervention == "" {
			if envelope.InterventionStage != "" || envelope.InterventionResult != "" || envelope.InterventionStrike != 0 {
				t.Fatalf("partial intervention labels: %+v", envelope)
			}
			if result, ok := envelope.Event.(ToolResultEvent); ok && result.Metadata["loop_guard_suppressed"] == true {
				t.Fatal("suppressed tool result missing its applied intervention")
			}
			continue
		}
		if envelope.Intervention != InterventionRepeatedToolSignature || envelope.InterventionStage != InterventionStageApplied || envelope.InterventionStrike == 0 {
			t.Fatalf("intervention envelope=%+v", envelope)
		}
		if _, accounting := envelope.Event.(AccountingEvent); !accounting {
			results[envelope.InterventionResult] = append(results[envelope.InterventionResult], envelope.InterventionStrike)
		}
		switch event := envelope.Event.(type) {
		case ToolResultEvent:
			if envelope.InterventionResult != InterventionResultToolSuppressed || event.Metadata["loop_guard_suppressed"] != true || envelope.ToolBlockID == "" || envelope.ToolCallOrdinal == 0 {
				t.Fatalf("suppressed tool result envelope=%+v", envelope)
			}
		case AccountingEvent:
			// Accounting for the suppressed result shares its correlation.
			if envelope.InterventionResult != InterventionResultToolSuppressed || envelope.ToolBlockID == "" {
				t.Fatalf("suppressed accounting envelope=%+v", envelope)
			}
			continue
		case WarnEvent, HiddenUserMessageEvent:
			if envelope.ToolBlockID != "" || envelope.FrameID != "" {
				t.Fatalf("non-tool intervention event inherited tool identity: %+v", envelope)
			}
		default:
			t.Fatalf("unexpected intervention event %T", envelope.Event)
		}
	}
	// Two applications: each has its warning, reminder and suppressed result
	// with strikes 1 and 2; the downgrade happens once, at strike 1.
	for result, want := range map[string][]uint64{
		InterventionResultToolSuppressed:  {1, 1, 2, 2}, // warning + tool result per strike
		InterventionResultReminderQueued:  {1, 2},
		InterventionResultGuardDowngraded: {1},
	} {
		got := results[result]
		if len(got) != len(want) {
			t.Fatalf("%s strikes=%v want %v", result, got, want)
		}
		seen := map[uint64]int{}
		for _, strike := range got {
			seen[strike]++
		}
		for _, strike := range want {
			seen[strike]--
		}
		for strike, n := range seen {
			if n != 0 {
				t.Fatalf("%s strike %d count mismatch: %v want %v", result, strike, got, want)
			}
		}
	}
	if echoCalls != 2 {
		t.Fatalf("echo calls=%d", echoCalls)
	}
	assertContiguousToolResults(t, ag.Messages())
}

// A proposed intervention whose history commit is rejected is never reported
// as applied: no applied envelope, no strike consumed, no reminder queued.
func TestInterventionRejectedCommitIsNotApplied(t *testing.T) {
	echoCalls := 0
	ag := newInterventionLifecycleAgent(t, &echoCalls)
	var records []interventionRecord
	ag.interventionObserved = func(record interventionRecord) { records = append(records, record) }
	blocks := 0
	ag.toolBlockTestHook = func(block *toolBlockState) {
		blocks++
		if blocks == 2 {
			// The second block carries the first repeated call; corrupting its
			// phase makes the suppressed result's commit fail validation.
			block.calls[0].phase = toolCallRunning
		}
	}
	var envelopes []EventEnvelope
	for envelope := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("loop")) {
		envelopes = append(envelopes, envelope)
	}
	for _, record := range records {
		if record.stage == interventionApplied {
			t.Fatalf("rejected commit reported applied: %+v", records)
		}
	}
	if len(records) == 0 {
		t.Fatal("fixture never proposed an intervention")
	}
	sawBlockError := false
	for _, envelope := range envelopes {
		if envelope.Intervention != "" {
			t.Fatalf("applied envelope after rejected commit: %+v", envelope)
		}
		switch event := envelope.Event.(type) {
		case HiddenUserMessageEvent:
			t.Fatal("reminder queued without an applied intervention")
		case ErrorEvent:
			sawBlockError = sawBlockError || event.Kind == "invalid_tool_call_block"
		}
	}
	if !sawBlockError {
		t.Fatal("fixture did not reject the commit")
	}
}

// The new envelope fields are optional: absent intervention labels are
// omitted from JSON, so older strict consumers see no new keys.
func TestInterventionEnvelopeFieldsAreOmittedWhenUnreported(t *testing.T) {
	data, err := json.Marshal(EventEnvelope{SchemaVersion: EventEnvelopeSchemaVersion, Kind: EventKindWarning, Event: WarnEvent{}})
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"Intervention", "InterventionStage", "InterventionResult", "InterventionStrike"} {
		if strings.Contains(string(data), `"`+key+`"`) {
			t.Fatalf("unreported %s serialized: %s", key, data)
		}
	}
}
