package compaction

import (
	"context"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestCompactLocalPruneUpgradesSnippedToolReplacement(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-prune")}
	svc := NewService(&Config{
		Enabled:        true,
		ContextWindow:  2000,
		ThresholdRatio: 0.85,
		SessionID:      "sess-prune",
		LedgerStore:    store,
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			return ArtifactResult{Path: ".goode/truncated/tool_grep.txt"}, nil
		}),
		ProtectedRecentMessages: 1,
	})
	messages := snipTestMessages(strings.Repeat("hit\n", 400))
	first, firstRes, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 1500})
	if err != nil {
		t.Fatalf("first CompactLocal: %v", err)
	}
	if !firstRes.Compacted || firstRes.TiersApplied[0] != "snip" {
		t.Fatalf("first result = %#v, want snip", firstRes)
	}
	snipHash := store.ledger.Replacements[0].ReplacementHash

	second, secondRes, err := svc.CompactLocal(context.Background(), first, &llm.Usage{TotalTokens: 1600})
	if err != nil {
		t.Fatalf("second CompactLocal: %v", err)
	}
	if !secondRes.Compacted || secondRes.Watermark != "prune" {
		t.Fatalf("second result = %#v, want prune", secondRes)
	}
	if !containsTier(secondRes.TiersApplied, "prune") {
		t.Fatalf("tiers = %#v, want prune", secondRes.TiersApplied)
	}
	got := second[2].Content.PlainText()
	if !strings.Contains(got, "[Tool result pruned:") || !strings.Contains(got, "full_output=.goode/truncated/tool_grep.txt") {
		t.Fatalf("tool result was not pruned: %q", got)
	}
	if len(store.ledger.Replacements) != 2 {
		t.Fatalf("ledger replacements = %#v, want snip + prune", store.ledger.Replacements)
	}
	prune := store.ledger.Replacements[1]
	if prune.Tier != "prune" || prune.ParentReplacementHash != snipHash {
		t.Fatalf("prune replacement = %#v, want parent hash %q", prune, snipHash)
	}

	third, thirdRes, err := svc.CompactLocal(context.Background(), second, &llm.Usage{TotalTokens: 1600})
	if err != nil {
		t.Fatalf("third CompactLocal: %v", err)
	}
	if thirdRes.Compacted {
		t.Fatalf("already-pruned history reported another compaction: %#v", thirdRes)
	}
	if !reflect.DeepEqual(third, second) || len(store.ledger.Replacements) != 2 || store.saves != 2 {
		t.Fatalf("already-pruned history churned: equal=%t replacements=%d saves=%d", reflect.DeepEqual(third, second), len(store.ledger.Replacements), store.saves)
	}
}

func TestCompactLocalPrunesOldAssistantTextButNotToolCallsOrUsers(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-prune")}
	writes := 0
	svc := NewService(&Config{
		Enabled:        true,
		ContextWindow:  4000,
		ThresholdRatio: 0.85,
		SessionID:      "sess-prune",
		LedgerStore:    store,
		ToolArtifactWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
			writes++
			return ArtifactResult{Path: ".goode/truncated/assistant_text.txt"}, nil
		}),
		ProtectedRecentMessages: 2,
	})
	userText := strings.Repeat("keep user text\n", 200)
	assistantText := "I inspected /tmp/project/main.go and found ERROR_CODE=42.\n" + strings.Repeat("details\n", 300)
	assistantWithCall := llm.NewAssistantMessage("I will inspect a file and this text must stay intact.", []llm.ToolCall{{ID: "call-read", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}})
	messages := []llm.Message{
		llm.NewUserMessage(userText),
		llm.NewAssistantMessage(assistantText, nil),
		assistantWithCall,
		llm.NewToolMessage("call-read", "read", llm.TextContent("ok"), false),
		llm.NewUserMessage("latest"),
	}

	got, res, err := svc.CompactLocal(context.Background(), messages, &llm.Usage{TotalTokens: 3200})
	if err != nil {
		t.Fatalf("CompactLocal: %v", err)
	}
	if !res.Compacted || res.Watermark != "prune" {
		t.Fatalf("result = %#v, want prune", res)
	}
	if writes != 1 {
		t.Fatalf("artifact writes = %d, want 1 assistant artifact", writes)
	}
	if got[0].Content.PlainText() != userText {
		t.Fatal("user message was changed")
	}
	pruned := got[1].Content.PlainText()
	if !strings.Contains(pruned, "[Assistant text compacted:") || !strings.Contains(pruned, "full_output=.goode/truncated/assistant_text.txt") {
		t.Fatalf("assistant text was not compacted: %q", pruned)
	}
	if !strings.Contains(pruned, "/tmp/project/main.go") || !strings.Contains(pruned, "ERROR_CODE=42") {
		t.Fatalf("assistant compacted text lost key tokens: %q", pruned)
	}
	if got[2].Content.PlainText() != assistantWithCall.Content.PlainText() || len(got[2].ToolCalls) != 1 {
		t.Fatalf("assistant tool-call message changed: %#v", got[2])
	}

	again, againRes, err := svc.CompactLocal(context.Background(), got, &llm.Usage{TotalTokens: 3200})
	if err != nil {
		t.Fatalf("CompactLocal already-pruned assistant: %v", err)
	}
	if againRes.Compacted {
		t.Fatalf("already-pruned assistant reported another compaction: %#v", againRes)
	}
	if !reflect.DeepEqual(again, got) || writes != 1 || store.saves != 1 || len(store.ledger.Replacements) != 1 {
		t.Fatalf("already-pruned assistant churned: equal=%t writes=%d saves=%d replacements=%d", reflect.DeepEqual(again, got), writes, store.saves, len(store.ledger.Replacements))
	}
}

func TestAssistantPreviewUsesSharedTokenTruncator(t *testing.T) {
	const path = "/工作区/项目/文件.go"
	text := strings.Repeat("背景", 300) + " " + path + " error=E42"
	got := assistantPreview(text)
	if !utf8.ValidString(got) || strings.ContainsRune(got, utf8.RuneError) {
		t.Fatalf("assistant preview is not valid UTF-8: %q", got)
	}
	if !strings.Contains(strings.ToLower(got), "truncated") {
		t.Fatalf("assistant preview is missing explicit truncation marker: %q", got)
	}
	if !strings.Contains(got, path) {
		t.Fatalf("assistant preview lost exact path %q: %q", path, got)
	}
}

func TestPruneWatermarkBetweenSnipAndSummary(t *testing.T) {
	svc := NewService(&Config{
		Enabled:        true,
		ContextWindow:  1000,
		ThresholdRatio: 0.85,
	})
	if got := svc.WatermarkForUsage(&llm.Usage{TotalTokens: 790}); got != "snip" {
		t.Fatalf("watermark at 79%% = %q, want snip", got)
	}
	if got := svc.WatermarkForUsage(&llm.Usage{TotalTokens: 800}); got != "prune" {
		t.Fatalf("watermark at 80%% = %q, want prune", got)
	}
	if got := svc.WatermarkForUsage(&llm.Usage{TotalTokens: 850}); got != "summarize" {
		t.Fatalf("watermark at 85%% = %q, want summarize", got)
	}
}

func containsTier(tiers []string, want string) bool {
	for _, tier := range tiers {
		if tier == want {
			return true
		}
	}
	return false
}
