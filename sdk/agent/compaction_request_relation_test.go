package agent

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// relationScriptModel returns scripted completions/errors and records the
// history text size of each request.
type relationScriptModel struct {
	mu       sync.Mutex
	steps    []func() (*llm.Completion, error)
	requests []int
}

func (m *relationScriptModel) Provider() string { return "fixture" }
func (m *relationScriptModel) Model() string    { return "relation" }
func (m *relationScriptModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	size := 0
	for _, message := range req.Messages {
		size += len(message.Content.PlainText())
	}
	m.requests = append(m.requests, size)
	step := m.steps[0]
	if len(m.steps) > 1 {
		m.steps = m.steps[1:]
	}
	return step()
}

func relationToolCall(tokens int) func() (*llm.Completion, error) {
	return func() (*llm.Completion, error) {
		return &llm.Completion{StopReason: "tool_calls", Usage: &llm.Usage{PromptTokens: tokens, CompletionTokens: 1, TotalTokens: tokens + 1},
			ToolCalls: []llm.ToolCall{{ID: "echo-1", Type: "function", Function: llm.FunctionCall{Name: "echo", Arguments: `{}`}}}}, nil
	}
}

func relationFinal() (*llm.Completion, error) {
	return &llm.Completion{StopReason: "stop", Content: llm.TextContent("finished"), Usage: &llm.Usage{PromptTokens: 10, CompletionTokens: 1, TotalTokens: 11}}, nil
}

func relationTransient() (*llm.Completion, error) {
	return nil, &llm.ProviderError{Provider: "fixture", StatusCode: 503, Message: "overloaded"}
}

func newRelationAgent(t *testing.T, model *relationScriptModel, bigHistory bool) *Agent {
	t.Helper()
	echo := tools.Func[struct{}]("echo", "echo", func(context.Context, struct{}, *tools.Container) (any, error) { return "ok", nil })
	history := []llm.Message{llm.NewUserMessage("earlier")}
	if bigHistory {
		history = append(history,
			llm.NewAssistantMessage("searching", []llm.ToolCall{{ID: "search", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
			llm.NewToolMessage("search", "grep", llm.TextContent(strings.Repeat("hit\n", 400)), false),
		)
	}
	ag, err := New(Config{
		LLM: model, Tools: []tools.Tool{echo}, InitialMessages: history, InvokeRetryMaxAttempts: 2, Warningf: func(string, ...any) {},
		Compaction: &compaction.Config{
			Enabled: true, ContextWindow: 100, SnipThresholdRatio: 0.70, PruneThresholdRatio: 0.80, ThresholdRatio: 0.85,
			SessionID: "relation", LedgerStore: &agentLocalLedgerStore{ledger: compaction.NewLedger("relation")},
			ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
				return compaction.ArtifactResult{Path: "fixture/tool.txt"}, nil
			}),
			ProtectedRecentMessages: 1,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return ag
}

type relationObservation struct {
	frameID, relation, source string
	attempt                   uint64
	compacted                 bool
}

func observeRelations(ag *Agent, prompt string) []relationObservation {
	var observed []relationObservation
	for envelope := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent(prompt)) {
		switch event := envelope.Event.(type) {
		case UsageEvent:
			observed = append(observed, relationObservation{frameID: envelope.FrameID, attempt: envelope.InvokeAttempt, relation: envelope.RequestHistoryRelation, source: envelope.RequestHistorySourceFrameID})
		case CompactionEvent:
			observed = append(observed, relationObservation{compacted: event.Result.Compacted, relation: envelope.RequestHistoryRelation})
		}
	}
	return observed
}

// An automatic compaction triggered by Frame 1's usage and published before
// Frame 2 is recorded on Frame 2 only, naming Frame 1; the triggering Frame
// and later Frames carry no relation.
func TestCompactionAppliedRelationNamesTriggeringFrame(t *testing.T) {
	model := &relationScriptModel{steps: []func() (*llm.Completion, error){relationToolCall(75), relationToolCall(10), relationFinal}}
	ag := newRelationAgent(t, model, true)
	observed := observeRelations(ag, "go")
	var usage []relationObservation
	compacted := false
	for _, o := range observed {
		if o.frameID == "" {
			compacted = compacted || o.compacted
			if o.relation != "" {
				t.Fatalf("compaction event carried a request relation: %+v", o)
			}
			continue
		}
		usage = append(usage, o)
	}
	if !compacted || len(usage) != 3 {
		t.Fatalf("compacted=%v usage=%+v", compacted, usage)
	}
	first, second, third := usage[0], usage[1], usage[2]
	if third.relation != "" || third.source != "" {
		t.Fatalf("relation persisted past the first Frame after publication: %+v", third)
	}
	if first.relation != "" || first.source != "" {
		t.Fatalf("triggering Frame carried a relation: %+v", first)
	}
	if second.relation != RequestHistoryCompactionApplied || second.source != first.frameID || second.frameID == first.frameID {
		t.Fatalf("next Frame relation=%+v, want source %s", second, first.frameID)
	}
	if len(model.requests) != 3 || model.requests[1] >= model.requests[0] {
		t.Fatalf("second request history was not compacted: text sizes %v", model.requests)
	}
}

// A framework retry of the related Frame reuses its relation.
func TestCompactionAppliedRelationSurvivesFrameRetry(t *testing.T) {
	model := &relationScriptModel{steps: []func() (*llm.Completion, error){relationToolCall(75), relationTransient, relationFinal}}
	ag := newRelationAgent(t, model, true)
	var usage []relationObservation
	for _, o := range observeRelations(ag, "go") {
		if o.frameID != "" {
			usage = append(usage, o)
		}
	}
	if len(usage) != 2 {
		t.Fatalf("usage=%+v", usage)
	}
	retried := usage[1]
	if retried.attempt != 2 || retried.relation != RequestHistoryCompactionApplied || retried.source != usage[0].frameID {
		t.Fatalf("retried Frame=%+v", retried)
	}
}

// No published compaction means no relation, and a relation recorded in one
// Query is never attributed to the next Query's Frames.
func TestCompactionAppliedRelationRequiresPublicationInSameQuery(t *testing.T) {
	model := &relationScriptModel{steps: []func() (*llm.Completion, error){relationToolCall(75), relationFinal}}
	ag := newRelationAgent(t, model, false)
	for _, o := range observeRelations(ag, "go") {
		if o.relation != "" || o.compacted {
			t.Fatalf("no-op compaction produced a relation: %+v", o)
		}
	}

	// A source recorded before this Query started is discarded.
	model = &relationScriptModel{steps: []func() (*llm.Completion, error){relationFinal}}
	ag = newRelationAgent(t, model, false)
	ag.mu.Lock()
	ag.appliedCompactionSource = "query_previous/frame/3"
	ag.mu.Unlock()
	for _, o := range observeRelations(ag, "again") {
		if o.relation != "" || o.source != "" {
			t.Fatalf("cross-Query relation leaked: %+v", o)
		}
	}

	// A compaction triggered by Query 1's final Frame is never attributed to a
	// Query 2 Frame.
	model = &relationScriptModel{steps: []func() (*llm.Completion, error){func() (*llm.Completion, error) {
		return &llm.Completion{StopReason: "stop", Content: llm.TextContent("done"), Usage: &llm.Usage{PromptTokens: 75, CompletionTokens: 1, TotalTokens: 76}}, nil
	}}}
	ag = newRelationAgent(t, model, true)
	ag.ReplaceHistory(append(ag.Messages(), llm.NewUserMessage("latest")))
	_ = observeRelations(ag, "first")
	for _, o := range observeRelations(ag, "second") {
		if o.relation != "" || o.source != "" {
			t.Fatalf("Query 2 attributed a Query 1 compaction: %+v", o)
		}
	}
}

// Manual/preflight compaction has no Frame source and clears a pending one.
func TestManualCompactionClearsAppliedRelation(t *testing.T) {
	ag := newRelationAgent(t, &relationScriptModel{steps: []func() (*llm.Completion, error){relationFinal}}, true)
	// Keep the large tool result outside the protected recent window.
	ag.ReplaceHistory(append(ag.Messages(), llm.NewUserMessage("latest")))
	ag.mu.Lock()
	ag.appliedCompactionSource = "query_x/frame/1"
	ag.mu.Unlock()
	res, err := ag.CompactPipelineNow(context.Background(), compaction.PipelineRequest{Trigger: "preflight", TargetWatermark: "snip", Usage: llm.WithPromptEstimate(nil, 75)})
	if err != nil || !res.Compacted {
		t.Fatalf("manual preflight compacted=%v err=%v", res.Compacted, err)
	}
	ag.mu.Lock()
	defer ag.mu.Unlock()
	if ag.appliedCompactionSource != "" {
		t.Fatalf("manual compaction kept source %q", ag.appliedCompactionSource)
	}
}

// The shared correlation projection keeps every producer's labels: the
// RequireDone control relation, the compaction history relation and an
// applied intervention can coexist on one envelope without overwriting.
func TestEventCorrelationComposesAllRelations(t *testing.T) {
	var envelope EventEnvelope
	correlation := eventCorrelation{frameID: "q/frame/2", attempt: 1, controlSource: "q/frame/1", historySource: "q/frame/1", recoverySource: "q/frame/1", steeringSource: "q/frame/1"}.withIntervention(InterventionResultToolSuppressed, 3)
	applyEventCorrelation(&envelope, []eventCorrelation{correlation})
	if envelope.RequestRecoveryRelation != RequestRecoveryStreamIdle || envelope.RequestRecoverySourceFrameID != "q/frame/1" ||
		envelope.RequestSteeringRelation != RequestSteeringAccepted || envelope.RequestSteeringSourceFrameID != "q/frame/1" {
		t.Fatalf("envelope=%+v", envelope)
	}
	if envelope.FrameID != "q/frame/2" || envelope.RequestControlRelation != RequestControlRequireDoneDisableThinking ||
		envelope.RequestHistoryRelation != RequestHistoryCompactionApplied || envelope.RequestHistorySourceFrameID != "q/frame/1" ||
		envelope.Intervention != InterventionRepeatedToolSignature || envelope.InterventionStrike != 3 {
		t.Fatalf("envelope=%+v", envelope)
	}
	// Without a Frame, the history relation is not attached (it describes a
	// Frame's request), while intervention labels still are.
	var bare EventEnvelope
	applyEventCorrelation(&bare, []eventCorrelation{eventCorrelation{historySource: "q/frame/1", steeringSource: "q/frame/1"}.withIntervention(InterventionResultReminderQueued, 1)})
	if bare.RequestHistoryRelation != "" || bare.RequestSteeringRelation != "" || bare.Intervention == "" {
		t.Fatalf("bare envelope=%+v", bare)
	}
}
