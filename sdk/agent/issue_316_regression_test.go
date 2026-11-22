package agent

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestD3ReturnedResultOwnsCheckpointMetadata(t *testing.T) {
	source := []llm.Message{llm.NewUserMessage("source")}
	candidate := []llm.Message{llm.NewUserMessage("candidate")}
	var saved compaction.CompactionCheckpoint
	ag, err := New(Config{LLM: &countingCompactionModel{}, InitialMessages: source, Compaction: &compaction.Config{
		Enabled:          true,
		CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(_ context.Context, cp compaction.CompactionCheckpoint) error { saved = cp; return nil }),
	}})
	if err != nil {
		t.Fatal(err)
	}
	input := compaction.Result{Compacted: true, TiersApplied: []string{"snip"}, Warnings: []string{"stored warning"}}
	input.Usage = llm.NewProviderUsage(10, 2, 12)
	cached := 3
	input.Usage.PromptCachedTokens = &cached
	res, err := ag.CommitCompactionHistory(context.Background(), source, candidate, input)
	if err != nil || !res.Compacted {
		t.Fatalf("commit: %+v, %v", res, err)
	}
	if err := saved.Validate(); err != nil {
		t.Fatalf("before caller edit: %v", err)
	}
	res.Warnings[0] = "warning edited for display"
	res.TiersApplied[0] = "edited tier"
	res.Usage.PromptTokens = 999
	*res.Usage.PromptCachedTokens = 999
	if err := saved.Validate(); err != nil {
		t.Fatalf("public result edit also changed writer snapshot to %q; validation=%v", saved.Result.Warnings[0], err)
	}
}
