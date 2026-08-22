package compaction

import (
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestCompactionCheckpointIsDeterministicAndValidated(t *testing.T) {
	messages := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewUserMessage("continue"),
	}
	res := Result{
		Compacted:      true,
		Trigger:        "usage",
		Watermark:      "summarize",
		OriginalTokens: 900,
		NewTokens:      180,
		TiersApplied:   []string{"snip", "summarize"},
		Warnings:       []string{"verify retained evidence"},
		Summary:        "bounded summary",
	}
	first, err := NewCompactionCheckpoint(messages, res)
	if err != nil {
		t.Fatalf("NewCompactionCheckpoint first: %v", err)
	}
	second, err := NewCompactionCheckpoint(messages, res)
	if err != nil {
		t.Fatalf("NewCompactionCheckpoint second: %v", err)
	}
	if first.CheckpointID == "" || first.CheckpointID != second.CheckpointID || first.Result.CheckpointID != first.CheckpointID || first.Result.CheckpointMessages != len(messages) {
		t.Fatalf("checkpoint ids are not stable: first=%#v second=%#v", first, second)
	}
	if err := first.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	tampered := first
	tampered.Messages = append([]llm.Message(nil), first.Messages...)
	tampered.Messages[1] = llm.NewUserMessage("tampered")
	if err := tampered.Validate(); err == nil {
		t.Fatal("tampered checkpoint unexpectedly validated")
	}
}
