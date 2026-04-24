package agent

import (
	"context"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type noopCompactionUpdateModel struct{}

func (noopCompactionUpdateModel) Provider() string { return "stub" }
func (noopCompactionUpdateModel) Model() string    { return "stub" }
func (noopCompactionUpdateModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("ok")}, nil
}

func TestAgentUpdateCompactionConfigReplacesService(t *testing.T) {
	ag, err := New(Config{
		LLM: noopCompactionUpdateModel{},
		Compaction: &compaction.Config{
			Enabled:       true,
			ContextWindow: 1000,
		},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	if !ag.hasCompactor || ag.compactor == nil {
		t.Fatal("expected initial compactor")
	}
	if ag.compactor.ContextWindow != 1000 {
		t.Fatalf("initial context window = %d, want 1000", ag.compactor.ContextWindow)
	}

	ag.UpdateCompactionConfig(&compaction.Config{Enabled: true, ContextWindow: 2000})
	if !ag.hasCompactor || ag.compactor == nil {
		t.Fatal("expected updated compactor")
	}
	if ag.compactor.ContextWindow != 2000 {
		t.Fatalf("updated context window = %d, want 2000", ag.compactor.ContextWindow)
	}

	ag.UpdateCompactionConfig(&compaction.Config{Enabled: false})
	if ag.hasCompactor {
		t.Fatal("expected compactor to be disabled")
	}
	if ag.compactor != nil {
		t.Fatal("expected compactor service to be cleared")
	}
}
