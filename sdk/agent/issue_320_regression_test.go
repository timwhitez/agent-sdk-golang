package agent

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/accounting"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"strings"
	"testing"
)

func TestD3AccountingLabelsOnlyEstimatorThatProducedUsage(t *testing.T) {
	usage := llm.NewProviderUsage(0, 2, 2)
	model := newGatedSummaryModel(func() (*llm.Completion, error) {
		return &llm.Completion{Content: llm.TextContent("answer"), Usage: usage}, nil
	})
	calls := 0
	est := accounting.Estimator{Name: "byte-counter", Version: "1", PolicyHash: "sha256:" + strings.Repeat("b", 64), EstimateTokens: func(s string) int { calls++; return len(s) }}
	ag, err := New(Config{LLM: model, AccountingEstimator: est, Compaction: &compaction.Config{Enabled: false}})
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("x", 128)
	for _, event := range collectEvents(ag.QueryStream(context.Background(), llm.TextContent(text))) {
		if ae, ok := event.(AccountingEvent); ok && ae.Payload.EventKind == accounting.EventKindProviderUsage {
			p := ae.Payload
			if p.Estimator != nil || calls != 0 || p.Usage.EffectivePromptTokens == nil || p.Usage.PromptTokensSource != llm.PromptTokensSourceEstimate {
				t.Fatalf("unexpected prompt provenance: usage=%+v estimator=%+v calls=%d (user text alone is %d bytes)", p.Usage, p.Estimator, calls, len(text))
			}
			return
		}
	}
	t.Fatal("no usage accounting event found")
}

func TestCompactionAccountingDoesNotAttributeToolEstimator(t *testing.T) {
	ag, err := New(Config{LLM: estimatedUsageAccountingModel{}, AccountingEstimator: accounting.Estimator{
		Name: "tool-only", Version: "1", PolicyHash: "sha256:" + strings.Repeat("b", 64), EstimateTokens: func(s string) int { return len(s) },
	}})
	if err != nil {
		t.Fatal(err)
	}
	out := newEventOutput(2, false, "", nil)
	ag.emitCompactionWithAccounting(out, CompactionEvent{Result: compaction.Result{Compacted: true, TokenCountSource: compaction.TokenCountSourceEstimate, OriginalTokens: 100, NewTokens: 20}})
	<-out.legacy
	event, ok := (<-out.legacy).(AccountingEvent)
	if !ok || event.Payload.Estimator != nil || event.Payload.Compaction.OriginalTokens == nil || *event.Payload.Compaction.OriginalTokens != 100 {
		t.Fatalf("compaction attribution=%+v", event)
	}
}
