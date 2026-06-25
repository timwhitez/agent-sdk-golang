package compaction

import (
	"context"
	"errors"
	"fmt"
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

func TestCompactLocalMicrocompactsOldUserCodeBlockWhenEnabled(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-local")}
	artifactWrites := 0
	var savedContent string
	svc := NewService(&Config{
		Enabled:                    true,
		ContextWindow:              2000,
		LedgerStore:                store,
		SessionID:                  "sess-local",
		EnableUserCodeMicrocompact: true,
		ProtectedRecentMessages:    1,
		ToolArtifactWriter: ArtifactWriterFunc(func(_ context.Context, req ArtifactRequest) (ArtifactResult, error) {
			artifactWrites++
			savedContent = req.Content
			return ArtifactResult{Path: ".goode/truncated/user_code_1.md"}, nil
		}),
	})
	oldUser := "Please inspect this file and keep the imports.\n\n```go cmd/main.go\n" + numberedLines("line-", 120) + "```\n"
	messages := []llm.Message{
		llm.NewUserMessage(oldUser),
		llm.NewAssistantMessage("noted", nil),
		llm.NewUserMessage("latest must remain verbatim"),
	}

	got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 1600})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if !res.Compacted {
		t.Fatalf("expected microcompact result: %#v", res)
	}
	if !containsTier(res.TiersApplied, "microcompact") {
		t.Fatalf("tiers = %#v, want microcompact", res.TiersApplied)
	}
	if artifactWrites != 1 {
		t.Fatalf("artifact writes = %d, want 1", artifactWrites)
	}
	if savedContent != oldUser {
		t.Fatalf("artifact content was not the original user message")
	}
	text := got[0].Content.PlainText()
	for _, want := range []string{"Please inspect this file", "[User code block compacted:", "language=go", "hint=cmd/main.go", "lines=120", "full_output=.goode/truncated/user_code_1.md", "line-0", "line-119"} {
		if !strings.Contains(text, want) {
			t.Fatalf("microcompact text missing %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "line-60") {
		t.Fatalf("microcompact retained middle bulk code:\n%s", text)
	}
	if got[2].Content.PlainText() != "latest must remain verbatim" {
		t.Fatalf("latest user changed: %#v", got[2])
	}
	if store.ledger == nil || len(store.ledger.Replacements) != 1 {
		t.Fatalf("ledger replacements = %#v", store.ledger)
	}
	repl := store.ledger.Replacements[0]
	if repl.Role != string(llm.RoleUser) || repl.Tier != "microcompact" || repl.FullArtifact != ".goode/truncated/user_code_1.md" {
		t.Fatalf("ledger replacement = %#v", repl)
	}

	second, _, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 1600})
	if err != nil {
		t.Fatalf("CompactLocal second: %v", err)
	}
	if artifactWrites != 1 {
		t.Fatalf("artifact writes after reuse = %d, want still 1", artifactWrites)
	}
	if second[0].Content.PlainText() != text {
		t.Fatalf("replacement not reused byte-for-byte")
	}
}

func TestCompactLocalMicrocompactKeepsPlainTextAndLatestUser(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-local")}
	svc := NewService(&Config{
		Enabled:                    true,
		ContextWindow:              2000,
		LedgerStore:                store,
		SessionID:                  "sess-local",
		EnableUserCodeMicrocompact: true,
		ProtectedRecentMessages:    1,
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/user_code.md"}, nil
		}),
	})
	plain := strings.Repeat("plain user instruction only\n", 200)
	latestCode := "latest code must stay\n```go\n" + numberedLines("latest-", 120) + "```\n"
	messages := []llm.Message{
		llm.NewUserMessage(plain),
		llm.NewAssistantMessage("ok", nil),
		llm.NewUserMessage(latestCode),
	}

	got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 1600})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if res.Compacted {
		t.Fatalf("expected no user microcompact, got %#v", res)
	}
	if got[0].Content.PlainText() != plain {
		t.Fatal("plain user text was compacted")
	}
	if got[2].Content.PlainText() != latestCode {
		t.Fatal("latest user code was compacted")
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

func TestCompactLocalEstimatedUsesPruneAtSummaryThreshold(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-local-estimated")}
	svc := NewService(&Config{
		Enabled:        true,
		ContextWindow:  100,
		ThresholdRatio: 0.85,
		SessionID:      "sess-local-estimated",
		LedgerStore:    store,
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
		}),
		ProtectedRecentMessages: 1,
	})
	got, res, err := svc.CompactLocalEstimated(context.Background(), snipTestMessages(strings.Repeat("hit\n", 300)), 90)
	if err != nil {
		t.Fatalf("CompactLocalEstimated: %v", err)
	}
	if !res.Compacted || res.Watermark != "prune" {
		t.Fatalf("result = %#v, want prune compaction", res)
	}
	if !containsTier(res.TiersApplied, "prune") {
		t.Fatalf("tiers = %#v, want prune", res.TiersApplied)
	}
	if !strings.Contains(got[2].Content.PlainText(), "[Tool result pruned:") {
		t.Fatalf("tool result was not locally reduced: %q", got[2].Content.PlainText())
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

func numberedLines(prefix string, count int) string {
	var b strings.Builder
	for i := 0; i < count; i++ {
		b.WriteString(prefix)
		b.WriteString(fmt.Sprint(i))
		b.WriteByte('\n')
	}
	return b.String()
}
