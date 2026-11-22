package agent

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestD3SecondEstimatePreservesRawProviderValues(t *testing.T) {
	// A custom model follows the SDK contract and supplies its own estimate before returning.
	before := llm.WithPromptEstimate(llm.NewProviderUsage(0, 2, 2), 40)
	model := newGatedSummaryModel(func() (*llm.Completion, error) {
		return &llm.Completion{Content: llm.TextContent("answer"), Usage: before}, nil
	})
	ag, err := New(Config{LLM: model, Compaction: &compaction.Config{Enabled: false}})
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range collectEvents(ag.QueryStream(context.Background(), llm.TextContent("fixture"))) {
		if usage, ok := event.(UsageEvent); ok {
			u := usage.Usage
			if u.ProviderPromptTokens == nil || u.ProviderTotalTokens == nil {
				t.Fatal("raw provider evidence lost")
			}
			if *u.ProviderPromptTokens != 0 || *u.ProviderTotalTokens != 2 {
				t.Fatalf("provider reported 0 prompt / 2 total, usage event now says raw provider counts %d / %d (earlier local estimate: 40 / 42)", *u.ProviderPromptTokens, *u.ProviderTotalTokens)
			}
			return
		}
	}
	t.Fatal("no usage event found")
}
