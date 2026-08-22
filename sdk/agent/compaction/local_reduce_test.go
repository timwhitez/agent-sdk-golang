package compaction

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type memoryLedgerStore struct {
	ledger  *Ledger
	saves   int
	err     error
	saveErr error
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
	if s.saveErr != nil {
		return s.saveErr
	}
	if s.err != nil {
		return s.err
	}
	s.saves++
	s.ledger = ledger.Clone()
	return nil
}

func TestCompactLocalLedgerSaveFailurePreservesHistory(t *testing.T) {
	ctx := context.Background()
	store := &memoryLedgerStore{
		ledger:  NewLedger("sess-ledger-save-failure"),
		saveErr: errors.New("ledger write denied"),
	}
	svc := NewService(&Config{
		Enabled:       true,
		ContextWindow: 2000,
		LedgerStore:   store,
		SessionID:     "sess-ledger-save-failure",
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/tool_grep_1.txt"}, nil
		}),
		ProtectedRecentMessages: 1,
	})
	messages := snipTestMessages(strings.Repeat("hit\n", 400))

	got, res, err := svc.CompactLocal(ctx, messages, &llm.Usage{TotalTokens: 1500})
	if err == nil || !strings.Contains(err.Error(), "ledger write denied") {
		t.Fatalf("CompactLocal error = %v, want ledger save failure", err)
	}
	if res.Compacted {
		t.Fatalf("failed ledger save reported compaction success: %#v", res)
	}
	if !reflect.DeepEqual(got, messages) {
		t.Fatalf("failed ledger save returned replacement history:\n got=%#v\nwant=%#v", got, messages)
	}
	if store.saves != 0 {
		t.Fatalf("successful ledger saves = %d, want 0", store.saves)
	}
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
	if store.saves != 1 {
		t.Fatalf("ledger saves after replacement reuse = %d, want still 1", store.saves)
	}

	third, thirdRes, err := svc.CompactLocal(ctx, first, &llm.Usage{TotalTokens: 1500})
	if err != nil {
		t.Fatalf("CompactLocal already-snipped history: %v", err)
	}
	if thirdRes.Compacted {
		t.Fatalf("already-snipped history reported another compaction: %#v", thirdRes)
	}
	if !reflect.DeepEqual(third, first) {
		t.Fatalf("already-snipped history changed:\n got=%#v\nwant=%#v", third, first)
	}
	if artifactWrites != 1 || store.saves != 1 || len(store.ledger.Replacements) != 1 {
		t.Fatalf("already-snipped history churned: writes=%d saves=%d replacements=%d", artifactWrites, store.saves, len(store.ledger.Replacements))
	}
}

func TestCompactLocalSameTextLedgerReuseIsNoOp(t *testing.T) {
	ctx := context.Background()
	messages := snipTestMessages(strings.Repeat("same output\n", 200))
	original := messages[2].Content.PlainText()
	key := StableMessageKey(MessageKeyInput{
		Role:           string(messages[2].Role),
		ToolCallID:     messages[2].ToolCallID,
		ToolName:       messages[2].ToolName,
		OriginalText:   original,
		FirstSeenIndex: 2,
	})
	ledger := NewLedger("sess-local-noop")
	ledger.Replacements = []LedgerReplacement{{
		MessageKey:      key,
		PartKey:         "content-0",
		Role:            string(messages[2].Role),
		ToolName:        messages[2].ToolName,
		Tier:            tierSnip,
		OriginalHash:    ContentHash(original),
		ReplacementHash: ContentHash(original),
		ReplacementText: original,
		FullArtifact:    ".goode/truncated/tool_grep.txt",
	}}
	store := &memoryLedgerStore{ledger: ledger}
	artifactWrites := 0
	svc := NewService(&Config{
		Enabled:       true,
		ContextWindow: 2000,
		LedgerStore:   store,
		SessionID:     "sess-local-noop",
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			artifactWrites++
			return ArtifactResult{Path: ".goode/truncated/unexpected.txt"}, nil
		}),
		ProtectedRecentMessages: 1,
	})

	got, res, err := svc.CompactLocal(ctx, messages, &llm.Usage{TotalTokens: 1500})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if res.Compacted {
		t.Fatalf("same-text ledger replacement reported compaction: %#v", res)
	}
	if !reflect.DeepEqual(got, messages) {
		t.Fatalf("same-text ledger replacement changed history:\n got=%#v\nwant=%#v", got, messages)
	}
	if artifactWrites != 0 || store.saves != 0 || len(store.ledger.Replacements) != 1 {
		t.Fatalf("same-text ledger replacement caused churn: writes=%d saves=%d replacements=%d", artifactWrites, store.saves, len(store.ledger.Replacements))
	}
}

func TestExtractTruncationArtifactPathRecognizesGeneratedAndLegacyMarkers(t *testing.T) {
	want := ".goode/truncated/tool_grep_1.txt"
	tests := []string{
		"[Tool result snipped: grep tool_call_id=call-grep lines=10 bytes=100 full_output=" + want + "]",
		"[Tool result snipped: grep tool_call_id=call-grep lines=10 bytes=100 full_output= " + want + "]",
		"output shortened; full output: " + want,
		"output shortened; saved to " + want,
	}
	for _, input := range tests {
		if got := extractTruncationArtifactPath(input); got != want {
			t.Fatalf("extractTruncationArtifactPath(%q) = %q, want %q", input, got, want)
		}
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

func TestCompactDestroyedPlaceholdersRemovesOnlyFullyDestroyedBlocks(t *testing.T) {
	svc := NewService(&Config{Enabled: true, ContextWindow: 1_000_000})
	messages := []llm.Message{
		llm.NewUserMessage("start"),
		llm.NewAssistantMessage("kept assistant evidence", []llm.ToolCall{{
			ID: "destroyed-only", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`},
		}}),
		{Role: llm.RoleTool, ToolCallID: "destroyed-only", ToolName: "read", Destroyed: true, Content: llm.TextContent("[destroyed]")},
		llm.NewAssistantMessage("mixed block", []llm.ToolCall{
			{ID: "mixed-live", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}},
			{ID: "mixed-destroyed", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}},
		}),
		llm.NewToolMessage("mixed-live", "read", llm.TextContent("valuable result"), false),
		{Role: llm.RoleTool, ToolCallID: "mixed-destroyed", ToolName: "read", Destroyed: true, Content: llm.TextContent("[destroyed]")},
		llm.NewUserMessage("latest"),
	}

	got, res, err := svc.CompactDestroyedPlaceholders(context.Background(), messages, &llm.Usage{PromptTokens: 100, TotalTokens: 100})
	if err != nil {
		t.Fatalf("CompactDestroyedPlaceholders: %v", err)
	}
	if !res.Compacted || res.Watermark != "placeholder_cleanup" {
		t.Fatalf("result = %#v", res)
	}
	if len(got) != len(messages)-1 {
		t.Fatalf("messages len = %d, want %d: %#v", len(got), len(messages)-1, got)
	}
	if got[1].Content.PlainText() != "kept assistant evidence" || len(got[1].ToolCalls) != 0 {
		t.Fatalf("destroyed-only assistant was not repaired: %#v", got[1])
	}
	foundMixedDestroyed := false
	for _, msg := range got {
		if msg.ToolCallID == "mixed-destroyed" && msg.Destroyed {
			foundMixedDestroyed = true
		}
	}
	if !foundMixedDestroyed {
		t.Fatal("mixed block with a live result must remain unchanged")
	}
}

func TestCompactLocalEstimatedStopsAfterSnipDropsBelowPruneThreshold(t *testing.T) {
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
	if !res.Compacted || res.Watermark != "snip" {
		t.Fatalf("result = %#v, want snip compaction", res)
	}
	if !containsTier(res.TiersApplied, "snip") || containsTier(res.TiersApplied, "prune") {
		t.Fatalf("tiers = %#v, want snip only after re-estimation", res.TiersApplied)
	}
	if !strings.Contains(got[2].Content.PlainText(), "[Tool result snipped:") {
		t.Fatalf("tool result was not locally reduced: %q", got[2].Content.PlainText())
	}
}

func TestProtectedZoneKeepsWholeCurrentToolHeavyTurn(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-current-turn")}
	svc := NewService(&Config{
		Enabled:                 true,
		ContextWindow:           4000,
		LedgerStore:             store,
		SessionID:               "sess-current-turn",
		ProtectedRecentMessages: 1,
		ToolArtifactWriter: ArtifactWriterFunc(func(_ context.Context, req ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/" + req.ToolCallID + ".txt"}, nil
		}),
	})
	oldOutput := strings.Repeat("old-result\n", 300)
	messages := []llm.Message{
		llm.NewUserMessage("old completed turn"),
		llm.NewAssistantMessage("old call", []llm.ToolCall{{ID: "old", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}}}),
		llm.NewToolMessage("old", "grep", llm.TextContent(oldOutput), false),
		llm.NewUserMessage("current tool-heavy request must remain intact"),
	}
	currentOutputs := make([]string, 0, 5)
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("current-%d", i)
		output := strings.Repeat(fmt.Sprintf("current-result-%d\n", i), 180)
		currentOutputs = append(currentOutputs, output)
		messages = append(messages,
			llm.NewAssistantMessage("current call", []llm.ToolCall{{ID: id, Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}}),
			llm.NewToolMessage(id, "read", llm.TextContent(output), false),
		)
	}

	got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3200})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if !res.Compacted || got[2].Content.PlainText() == oldOutput {
		t.Fatalf("old completed turn was not compacted: result=%#v", res)
	}
	for i, want := range currentOutputs {
		index := 5 + i*2
		if got[index].Content.PlainText() != want {
			t.Fatalf("current turn tool result %d changed at message %d", i, index)
		}
	}
}

func TestProtectedZoneKeepsOpenToolCallResultBlock(t *testing.T) {
	svc := NewService(&Config{
		Enabled:                 true,
		ContextWindow:           2000,
		LedgerStore:             &memoryLedgerStore{ledger: NewLedger("sess-open-block")},
		SessionID:               "sess-open-block",
		ProtectedRecentMessages: 1,
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/open-block.txt"}, nil
		}),
	})
	partialResult := strings.Repeat("partial-result\n", 300)
	messages := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewAssistantMessage("two calls are still in flight", []llm.ToolCall{
			{ID: "call-complete", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}},
			{ID: "call-missing", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{}`}},
		}),
		llm.NewToolMessage("call-complete", "read", llm.TextContent(partialResult), false),
		llm.NewAssistantMessage("streamed continuation while the second result is pending", nil),
	}

	got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 1500})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if res.Compacted {
		t.Fatalf("open tool topology was compacted: %#v", res)
	}
	if got[2].Content.PlainText() != partialResult {
		t.Fatal("tool result inside open call/result block changed")
	}
}

func TestMicrocompactNeverTouchesLatestRealUser(t *testing.T) {
	artifactWrites := 0
	svc := NewService(&Config{
		Enabled:                    true,
		ContextWindow:              2000,
		LedgerStore:                &memoryLedgerStore{ledger: NewLedger("sess-latest-real-user")},
		SessionID:                  "sess-latest-real-user",
		EnableUserCodeMicrocompact: true,
		ProtectedRecentMessages:    1,
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			artifactWrites++
			return ArtifactResult{Path: ".goode/truncated/latest-user.md"}, nil
		}),
	})
	latestRealUser := "latest real user code must remain verbatim\n```go\n" + numberedLines("latest-", 140) + "```\n"
	messages := []llm.Message{
		llm.NewUserMessage(latestRealUser),
		llm.NewAssistantMessage("working", nil),
		{Role: llm.RoleUser, Name: "sdk_internal_require_done", Content: llm.TextContent("Task completion must use the done tool.")},
		llm.NewAssistantMessage("continuing", nil),
	}

	got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 1600})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if res.Compacted || artifactWrites != 0 {
		t.Fatalf("latest real user was microcompacted: result=%#v writes=%d", res, artifactWrites)
	}
	if got[0].Content.PlainText() != latestRealUser {
		t.Fatal("latest real user content changed")
	}
}

func TestProtectedZoneHonorsConfiguredRecentTokenBudget(t *testing.T) {
	svc := NewService(&Config{
		Enabled:                 true,
		ContextWindow:           4000,
		LedgerStore:             &memoryLedgerStore{ledger: NewLedger("sess-token-zone")},
		SessionID:               "sess-token-zone",
		ProtectedRecentMessages: 1,
		ProtectedRecentTokens:   500,
		TokenEstimator: func(text string) int {
			return len(text)
		},
		ToolArtifactWriter: ArtifactWriterFunc(func(_ context.Context, req ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/" + req.ToolCallID + ".txt"}, nil
		}),
	})
	oldOutput := strings.Repeat("o", 1000)
	recentOutput := strings.Repeat("r", 1000)
	messages := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewAssistantMessage("old", []llm.ToolCall{{ID: "old", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}}),
		llm.NewToolMessage("old", "read", llm.TextContent(oldOutput), false),
		llm.NewAssistantMessage("recent", []llm.ToolCall{{ID: "recent", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}}),
		llm.NewToolMessage("recent", "read", llm.TextContent(recentOutput), false),
		llm.NewAssistantMessage("tail", nil),
	}

	got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3200})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if !res.Compacted || got[2].Content.PlainText() == oldOutput {
		t.Fatalf("old material outside token zone was not compacted: %#v", res)
	}
	if got[4].Content.PlainText() != recentOutput {
		t.Fatal("configured recent-token zone did not protect recent tool output")
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
