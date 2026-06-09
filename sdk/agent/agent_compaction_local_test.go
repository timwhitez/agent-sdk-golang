package agent

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type localCompactionCountingModel struct {
	calls atomic.Int32
}

func (m *localCompactionCountingModel) Provider() string { return "mock" }
func (m *localCompactionCountingModel) Model() string    { return "mock" }
func (m *localCompactionCountingModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls.Add(1)
	return &llm.Completion{Content: llm.TextContent("<summary>unexpected summary</summary>")}, nil
}

func TestCheckAndCompactUsesLocalSnipWithoutModelInvoke(t *testing.T) {
	model := &localCompactionCountingModel{}
	store := &agentLocalLedgerStore{ledger: compaction.NewLedger("sess-agent-local")}
	ag, err := New(Config{
		LLM: model,
		Compaction: &compaction.Config{
			Enabled:        true,
			ContextWindow:  100,
			ThresholdRatio: 0.85,
			SessionID:      "sess-agent-local",
			LedgerStore:    store,
			ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
				return compaction.ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
			}),
			ProtectedRecentMessages: 1,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ag.ReplaceHistory([]llm.Message{
		llm.NewUserMessage("search"),
		llm.NewAssistantMessage("calling grep", []llm.ToolCall{{ID: "call-grep", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage("call-grep", "grep", llm.TextContent(strings.Repeat("hit\n", 300)), false),
		llm.NewUserMessage("latest"),
	})

	ag.checkAndCompact(context.Background(), &llm.Completion{Usage: &llm.Usage{PromptTokens: 70, TotalTokens: 70}}, nil)
	waitFor(t, time.Second, func() bool {
		return !ag.compactionInFlight.Load() && ag.hasPendingCompaction()
	}, "local snip pending compaction")
	if got := model.calls.Load(); got != 0 {
		t.Fatalf("compaction model calls = %d, want 0 for local snip", got)
	}

	ag.applyPendingCompaction(nil)
	msgs := ag.Messages()
	if len(msgs) < 3 {
		t.Fatalf("messages after compaction = %#v", msgs)
	}
	if got := msgs[2].Content.PlainText(); !strings.Contains(got, "[Tool result snipped:") {
		t.Fatalf("tool result was not snipped: %q", got)
	}
}

type agentLocalLedgerStore struct {
	ledger *compaction.Ledger
}

func (s *agentLocalLedgerStore) Load(context.Context, string) (*compaction.Ledger, error) {
	if s.ledger == nil {
		return compaction.NewLedger(""), nil
	}
	return s.ledger.Clone(), nil
}

func (s *agentLocalLedgerStore) Save(_ context.Context, _ string, ledger *compaction.Ledger) error {
	s.ledger = ledger.Clone()
	return nil
}
