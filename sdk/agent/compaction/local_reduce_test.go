package compaction

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type memoryLedgerStore struct {
	ledger *Ledger
	saves  int
	err    error
}

func (s *memoryLedgerStore) Load(context.Context, string) (*Ledger, error) {
	if s.err != nil {
		return nil, s.err
	}
	if s.ledger == nil {
		return NewLedger("sess-local"), nil
	}
	return s.ledger.Clone(), nil
}

func (s *memoryLedgerStore) Save(_ context.Context, _ string, ledger *Ledger) error {
	if s.err != nil {
		return s.err
	}
	s.saves++
	s.ledger = ledger.Clone()
	return nil
}

func TestCompactLocalSnipsOldToolResultAndReusesLedger(t *testing.T) {
	ctx := context.Background()
	store := &memoryLedgerStore{ledger: NewLedger("sess-local")}
	artifactWrites := 0
	svc := NewService(&Config{
		Enabled:       true,
		ContextWindow: 2000,
		LedgerStore:   store,
		SessionID:     "sess-local",
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			artifactWrites++
			return ArtifactResult{Path: ".goode/truncated/tool_grep_1.txt"}, nil
		}),
		ProtectedRecentMessages: 1,
	})
	messages := snipTestMessages(strings.Repeat("hit\n", 400))

	first, firstRes, err := svc.CompactLocal(ctx, messages, &llm.Usage{TotalTokens: 1500})
	if err != nil {
		t.Fatalf("CompactLocal first: %v", err)
	}
	if !firstRes.Compacted {
		t.Fatal("expected local compaction")
	}
	if firstRes.Watermark != "snip" {
		t.Fatalf("watermark = %q, want snip", firstRes.Watermark)
	}
	if len(firstRes.TiersApplied) != 1 || firstRes.TiersApplied[0] != "snip" {
		t.Fatalf("tiers = %#v, want [snip]", firstRes.TiersApplied)
	}
	if artifactWrites != 1 {
		t.Fatalf("artifact writes = %d, want 1", artifactWrites)
	}
	if store.saves != 1 {
		t.Fatalf("ledger saves = %d, want 1", store.saves)
	}
	if len(store.ledger.Replacements) != 1 {
		t.Fatalf("ledger replacements = %#v, want one", store.ledger.Replacements)
	}
	repl := store.ledger.Replacements[0]
	if repl.Tier != "snip" || repl.FullArtifact != ".goode/truncated/tool_grep_1.txt" {
		t.Fatalf("replacement = %#v", repl)
	}
	gotTool := first[2]
	if gotTool.Role != llm.RoleTool || gotTool.ToolCallID != "call-grep" || gotTool.ToolName != "grep" {
		t.Fatalf("tool linkage changed: %#v", gotTool)
	}
	replacementText := gotTool.Content.PlainText()
	for _, want := range []string{"[Tool result snipped:", "grep", "tool_call_id=call-grep", "full_output=.goode/truncated/tool_grep_1.txt"} {
		if !strings.Contains(replacementText, want) {
			t.Fatalf("replacement text %q missing %q", replacementText, want)
		}
	}
	if strings.Contains(replacementText, "hit\nhit\nhit") {
		t.Fatalf("replacement kept bulk output: %q", replacementText)
	}
	assertLocalHistoryProviderValid(t, first)

	second, secondRes, err := svc.CompactLocal(ctx, messages, &llm.Usage{TotalTokens: 1500})
	if err != nil {
		t.Fatalf("CompactLocal second: %v", err)
	}
	if artifactWrites != 1 {
		t.Fatalf("artifact writes after reuse = %d, want still 1", artifactWrites)
	}
	if second[2].Content.PlainText() != replacementText {
		t.Fatalf("replacement not reused byte-for-byte:\nfirst=%q\nsecond=%q", replacementText, second[2].Content.PlainText())
	}
	if secondRes.Compacted != true {
		t.Fatalf("second result = %#v, want compacted", secondRes)
	}
}

func TestCompactLocalSkipsProtectedRecentAndUserMessages(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-local")}
	svc := NewService(&Config{
		Enabled:       true,
		ContextWindow: 2000,
		LedgerStore:   store,
		SessionID:     "sess-local",
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/tool.txt"}, nil
		}),
		ProtectedRecentMessages: 3,
	})
	userText := strings.Repeat("do not compact user text\n", 200)
	oldTool := strings.Repeat("old\n", 300)
	recentTool := strings.Repeat("recent\n", 300)
	messages := []llm.Message{
		llm.NewUserMessage(userText),
		llm.NewAssistantMessage("calling old", []llm.ToolCall{{ID: "call-old", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage("call-old", "grep", llm.TextContent(oldTool), false),
		llm.NewAssistantMessage("calling recent", []llm.ToolCall{{ID: "call-recent", Type: "function", Function: llm.FunctionCall{Name: "bash", Arguments: `{}`}}}),
		llm.NewToolMessage("call-recent", "bash", llm.TextContent(recentTool), false),
		llm.NewUserMessage("latest"),
	}

	got, _, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 1500})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if got[0].Content.PlainText() != userText {
		t.Fatal("user message was compacted")
	}
	if got[2].Content.PlainText() == oldTool {
		t.Fatal("old tool result should be snipped")
	}
	if got[4].Content.PlainText() != recentTool {
		t.Fatal("protected recent tool result was snipped")
	}
}

func TestCompactLocalSkipsProtectedTools(t *testing.T) {
	svc := NewService(&Config{
		Enabled:                 true,
		ContextWindow:           2000,
		LedgerStore:             &memoryLedgerStore{ledger: NewLedger("sess-local")},
		SessionID:               "sess-local",
		ProtectedTools:          []string{"read"},
		ProtectedRecentMessages: 1,
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/tool.txt"}, nil
		}),
	})
	readOutput := strings.Repeat("file\n", 300)
	messages := []llm.Message{
		llm.NewAssistantMessage("read", []llm.ToolCall{{ID: "call-read", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}}),
		llm.NewToolMessage("call-read", "read", llm.TextContent(readOutput), false),
		llm.NewUserMessage("latest"),
	}

	got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 1500})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if res.Compacted {
		t.Fatalf("expected protected tool to skip compaction, got %#v", res)
	}
	if got[1].Content.PlainText() != readOutput {
		t.Fatal("protected read tool was snipped")
	}
}

func TestCompactLocalArtifactFailureLeavesContentAndWarns(t *testing.T) {
	original := strings.Repeat("line\n", 300)
	svc := NewService(&Config{
		Enabled:                 true,
		ContextWindow:           2000,
		LedgerStore:             &memoryLedgerStore{ledger: NewLedger("sess-local")},
		SessionID:               "sess-local",
		ProtectedRecentMessages: 1,
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{}, errors.New("disk full")
		}),
	})
	messages := snipTestMessages(original)

	got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 1500})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if res.Compacted {
		t.Fatalf("expected no compaction when artifact write fails, got %#v", res)
	}
	if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0], "[WARN] Compaction artifact not saved") {
		t.Fatalf("warnings = %#v", res.Warnings)
	}
	if got[2].Content.PlainText() != original {
		t.Fatal("tool content changed despite artifact failure")
	}
}

func TestLocalSnipWatermarkStartsBeforeSummaryThreshold(t *testing.T) {
	svc := NewService(&Config{
		Enabled:        true,
		ContextWindow:  1000,
		ThresholdRatio: 0.85,
	})
	if !svc.ShouldCompact(&llm.Usage{TotalTokens: 700}) {
		t.Fatal("expected snip watermark to request compaction at 70%")
	}
	if got := svc.WatermarkForUsage(&llm.Usage{TotalTokens: 700}); got != "snip" {
		t.Fatalf("watermark at 70%% = %q, want snip", got)
	}
	if got := svc.WatermarkForUsage(&llm.Usage{TotalTokens: 850}); got != "summarize" {
		t.Fatalf("watermark at summary threshold = %q, want summarize", got)
	}
}

func TestCompactAutoUsesSnipBeforeSummaryWhenLocalReductionIsEnough(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-local")}
	svc := NewService(&Config{
		Enabled:        true,
		ContextWindow:  300,
		ThresholdRatio: 0.85,
		SessionID:      "sess-local",
		LedgerStore:    store,
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
		}),
		ProtectedRecentMessages: 1,
	})
	model := &promptCaptureModel{response: "<summary>should not be called</summary>"}

	got, res, err := svc.CompactAuto(context.Background(), model, snipTestMessages(strings.Repeat("hit\n", 300)), &llm.Usage{TotalTokens: 255}, "summarize")
	if err != nil {
		t.Fatalf("CompactAuto: %v", err)
	}
	if model.lastPrompt != "" {
		t.Fatalf("summary model was invoked with prompt %q", model.lastPrompt)
	}
	if !res.Compacted || res.Watermark != "snip" {
		t.Fatalf("result = %#v, want snip compaction", res)
	}
	if len(res.TiersApplied) != 1 || res.TiersApplied[0] != "snip" {
		t.Fatalf("tiers = %#v, want [snip]", res.TiersApplied)
	}
	if !strings.Contains(got[2].Content.PlainText(), "[Tool result snipped:") {
		t.Fatalf("tool result was not snipped: %q", got[2].Content.PlainText())
	}
}

func snipTestMessages(toolOutput string) []llm.Message {
	return []llm.Message{
		llm.NewUserMessage("search for hits"),
		llm.NewAssistantMessage("calling grep", []llm.ToolCall{{ID: "call-grep", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{"pattern":"hit"}`}}}),
		llm.NewToolMessage("call-grep", "grep", llm.TextContent(toolOutput), false),
		llm.NewUserMessage("latest protected"),
	}
}

func assertLocalHistoryProviderValid(t *testing.T, messages []llm.Message) {
	t.Helper()
	pending := map[string]bool{}
	for i, msg := range messages {
		if msg.Role == llm.RoleTool {
			if _, ok := pending[msg.ToolCallID]; !ok {
				t.Fatalf("message %d orphan tool result %#v", i, msg)
			}
			if pending[msg.ToolCallID] {
				t.Fatalf("message %d duplicate tool result id %q", i, msg.ToolCallID)
			}
			pending[msg.ToolCallID] = true
			continue
		}
		for id, seen := range pending {
			if !seen {
				t.Fatalf("assistant tool call %q missing result before message %d", id, i)
			}
			delete(pending, id)
		}
		if msg.Role != llm.RoleAssistant {
			continue
		}
		for _, call := range msg.ToolCalls {
			if strings.TrimSpace(call.ID) == "" {
				t.Fatalf("message %d has empty tool call id", i)
			}
			pending[call.ID] = false
		}
	}
	for id, seen := range pending {
		if !seen {
			t.Fatalf("assistant tool call %q missing trailing result", id)
		}
	}
}
