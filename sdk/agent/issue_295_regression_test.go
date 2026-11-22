package agent

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
	"time"
)

func TestD1CanceledQueryDuringPriorSummary(t *testing.T) {
	var high *llm.Usage
	model := newGatedSummaryModel(func() (*llm.Completion, error) {
		return &llm.Completion{Content: llm.TextContent("first answer"), Usage: high}, nil
	})
	a, u := interleaveAgent(t, model, nil, nil)
	high = u
	if _, err := a.Query(t.Context(), "first question"); err != nil {
		t.Fatal(err)
	}
	select {
	case <-model.summaryStarted:
	case <-time.After(3 * time.Second):
		t.Fatal("summary never started")
	}
	defer func() { close(model.gate); _ = a.waitForCompactionIdle(t.Context(), nil) }()
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	answer, err := a.Query(ctx, "second question")
	if err == nil {
		t.Fatalf("second Query returned answer=%q err=%v after context expired while waiting for the first summary; want cancellation error", answer, err)
	}
}
