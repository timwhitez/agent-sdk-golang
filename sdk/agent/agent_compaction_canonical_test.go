package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type agentCanonicalLedgerStore struct {
	mu     sync.Mutex
	ledger *compaction.Ledger
}

func (s *agentCanonicalLedgerStore) Load(_ context.Context, sessionID string) (*compaction.Ledger, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ledger == nil {
		return compaction.NewLedger(sessionID), nil
	}
	return s.ledger.Clone(), nil
}

func (s *agentCanonicalLedgerStore) Save(_ context.Context, sessionID string, ledger *compaction.Ledger) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ledger == nil {
		return errors.New("nil ledger")
	}
	if err := ledger.Validate(sessionID); err != nil {
		return err
	}
	s.ledger = ledger.Clone()
	return nil
}

func (s *agentCanonicalLedgerStore) snapshot() *compaction.Ledger {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ledger == nil {
		return nil
	}
	return s.ledger.Clone()
}

func TestAgentCompactionUsesCanonicalBindingAcrossConfigUpdate(t *testing.T) {
	const sessionID = "session-boundary"
	sink := &artifactBoundarySink{}
	capability := artifactBoundaryCapability("Call artifact_read with object_ref, or artifact_range with object_ref, start, and end.")
	firstLedger := &agentCanonicalLedgerStore{}
	legacyWrites := 0
	newCompactionConfig := func(ledger *agentCanonicalLedgerStore) *compaction.Config {
		return &compaction.Config{
			Enabled:                 true,
			ContextWindow:           4000,
			SessionID:               sessionID,
			LedgerStore:             ledger,
			ProtectedRecentMessages: 1,
			ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
				legacyWrites++
				return compaction.ArtifactResult{}, errors.New("legacy writer must not be used")
			}),
		}
	}
	firstMessages := canonicalAgentCompactionMessages("call-first", strings.Repeat("first canonical source\n", 400))
	ag, err := New(Config{
		LLM:                        noopCompactionUpdateModel{},
		InitialMessages:            firstMessages,
		Compaction:                 newCompactionConfig(firstLedger),
		ArtifactOwner:              artifactBoundaryOwner(),
		ArtifactSink:               sink,
		ArtifactResolver:           sink,
		ArtifactResolverCapability: capability,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	res, err := ag.CompactLocalNow(context.Background(), 3000)
	if err != nil {
		t.Fatalf("first CompactLocalNow: %v", err)
	}
	if !res.Compacted || legacyWrites != 0 {
		t.Fatalf("first canonical result=%#v legacy_writes=%d", res, legacyWrites)
	}
	first := firstLedger.snapshot()
	if first == nil || len(first.Replacements) != 1 || first.Replacements[0].CanonicalArtifact == nil || first.Replacements[0].FullArtifact != "" {
		t.Fatalf("first canonical ledger = %#v", first)
	}

	secondLedger := &agentCanonicalLedgerStore{}
	ag.UpdateCompactionConfig(newCompactionConfig(secondLedger))
	if ag.compactor == nil || ag.compactor.Config.ArtifactResolver != sink || ag.compactor.Config.ArtifactSink != sink || ag.compactor.Config.ArtifactOwnerProvider == nil {
		t.Fatalf("updated compaction binding was not retained: %#v", ag.compactor)
	}
	ag.ReplaceHistory(canonicalAgentCompactionMessages("call-second", strings.Repeat("second canonical source\n", 400)))
	res, err = ag.CompactLocalNow(context.Background(), 3000)
	if err != nil {
		t.Fatalf("second CompactLocalNow: %v", err)
	}
	if !res.Compacted || legacyWrites != 0 {
		t.Fatalf("updated canonical result=%#v legacy_writes=%d", res, legacyWrites)
	}
	second := secondLedger.snapshot()
	if second == nil || len(second.Replacements) != 1 || second.Replacements[0].CanonicalArtifact == nil || second.Replacements[0].FullArtifact != "" {
		t.Fatalf("second canonical ledger = %#v", second)
	}
	if len(sink.requests) != 2 {
		t.Fatalf("canonical sink requests = %d, want one per distinct source", len(sink.requests))
	}
}

func TestAgentDefaultEnvelopeCodecDoesNotEnableCanonicalCompaction(t *testing.T) {
	ag, err := New(Config{
		LLM: noopCompactionUpdateModel{},
		Compaction: &compaction.Config{
			Enabled: true,
			ToolArtifactWriter: compaction.ArtifactWriterFunc(func(context.Context, compaction.ArtifactRequest) (compaction.ArtifactResult, error) {
				return compaction.ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
			}),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if ag.compactor == nil {
		t.Fatal("compactor is nil")
	}
	got := ag.compactor.Config
	if got.ArtifactOwnerProvider != nil || got.ArtifactSink != nil || got.ArtifactResolver != nil ||
		got.ArtifactResolverCapability.Registered || got.ArtifactEnvelopeCodec != nil {
		t.Fatalf("default envelope codec enabled a partial canonical binding: %#v", got)
	}
}

func canonicalAgentCompactionMessages(callID, output string) []llm.Message {
	return []llm.Message{
		llm.NewUserMessage("old request"),
		llm.NewAssistantMessage("calling tool", []llm.ToolCall{{ID: callID, Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage(callID, "grep", llm.TextContent(output), false),
		llm.NewUserMessage("latest protected"),
	}
}
