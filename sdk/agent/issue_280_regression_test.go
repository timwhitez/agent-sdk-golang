package agent

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestD3PublishedCheckpointCanReplay(t *testing.T) {
	source := []llm.Message{llm.NewUserMessage("source")}
	var saved compaction.CompactionCheckpoint
	writes := 0
	ag, err := New(Config{LLM: &countingCompactionModel{}, InitialMessages: source, Compaction: &compaction.Config{
		Enabled:          true,
		CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(_ context.Context, cp compaction.CompactionCheckpoint) error { writes++; saved = cp; return nil }),
	}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := ag.CommitCompactionHistory(context.Background(), ag.Messages(), nil, compaction.Result{Compacted: true})
	if err == nil || res.Compacted || writes != 0 || len(ag.Messages()) != 1 || ag.Messages()[0].PlainText() != "source" {
		t.Fatalf("invalid candidate changed durable/live state: result=%+v err=%v writes=%d history=%+v validation=%v", res, err, writes, ag.Messages(), saved.Validate())
	}
}
