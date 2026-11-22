package compaction

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestD3NoShrinkKeepsAdditionalTokens(t *testing.T) {
	svc := newOrderedPipelineService(t, nil, false)
	svc.Config.KeepRecentUserMessages = 3
	const additional = 400
	messages := orderedPipelineMessages()
	model := &orderedPipelineModel{}
	out, res, err := svc.CompactPipeline(context.Background(), model, messages, PipelineRequest{
		Usage: &llm.Usage{PromptTokens: 180, TotalTokens: 180}, TargetWatermark: "summarize", AllowSummary: true, AdditionalTokens: additional,
	})
	if err != nil || model.calls != 1 || !res.SummaryNotSmaller || !res.Compacted {
		t.Fatalf("summary setup: %+v, %v", res, err)
	}
	if want := svc.EstimateMessages(out) + additional; res.NewTokens != want {
		t.Fatalf("returned new_tokens=%d; output history=%d plus additional=%d requires %d; original_tokens=%d", res.NewTokens, svc.EstimateMessages(out), additional, want, res.OriginalTokens)
	}
}
