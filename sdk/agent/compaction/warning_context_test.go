package compaction

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type compactWarningModel struct{}

func (compactWarningModel) Provider() string { return "fixture" }
func (compactWarningModel) Model() string    { return "fixture" }
func (compactWarningModel) Invoke(ctx context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	if sink := llm.WarningSink(ctx, nil); sink != nil {
		sink("compaction fixture")
	}
	return nil, errors.New("fixture stop after warning")
}

func TestCompactionWarningContextUsesServiceScope(t *testing.T) {
	var outer atomic.Int32
	ctx := llm.WithWarningSink(context.Background(), func(string, ...any) { outer.Add(1) })
	var counts [8]atomic.Int32
	var workers sync.WaitGroup
	for i := range counts {
		workers.Add(1)
		go func(index int) {
			defer workers.Done()
			service := NewService(&Config{Warningf: func(string, ...any) { counts[index].Add(1) }})
			if _, _, err := service.compactSummary(ctx, compactWarningModel{}, []llm.Message{llm.NewUserMessage("fixture")}); err == nil {
				t.Error("expected fixture stop")
			}
		}(i)
	}
	workers.Wait()
	if outer.Load() != 0 {
		t.Fatal("compaction reused caller's unrelated sink")
	}
	for i := range counts {
		if counts[i].Load() != 1 {
			t.Fatalf("service%d received %d diagnostics", i, counts[i].Load())
		}
	}
}
