package compaction

import (
	"context"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type summaryPromptCaptureModel struct {
	response string
	prompts  []string
}

func (m *summaryPromptCaptureModel) Provider() string { return "mock" }
func (m *summaryPromptCaptureModel) Model() string    { return "mock" }
func (m *summaryPromptCaptureModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	var b strings.Builder
	for _, msg := range req.Messages {
		if b.Len() > 0 {
			b.WriteString("\n---MSG---\n")
		}
		b.WriteString(msg.Content.PlainText())
	}
	m.prompts = append(m.prompts, b.String())
	return &llm.Completion{Content: llm.TextContent(m.response)}, nil
}

func TestCompactUsesPreviousSummaryAndDeltaOnSecondSummary(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-summary")}
	model := &summaryPromptCaptureModel{response: structuredTestSummary("Completed Work", "initial summary")}
	svc := NewService(&Config{
		Enabled:                true,
		SessionID:              "sess-summary",
		LedgerStore:            store,
		SummaryPrompt:          "summarize now",
		KeepRecentUserMessages: 1,
	})
	initialMessages := []llm.Message{
		llm.NewUserMessage("covered old user request"),
		llm.NewAssistantMessage("covered old assistant answer", nil),
		llm.NewUserMessage("latest one"),
	}

	first, firstRes, err := svc.Compact(context.Background(), model, initialMessages)
	if err != nil {
		t.Fatalf("initial Compact: %v", err)
	}
	if !firstRes.Compacted {
		t.Fatal("expected initial summary compaction")
	}
	if store.ledger.Summary == nil || store.ledger.Summary.Version != 1 {
		t.Fatalf("initial ledger summary = %#v", store.ledger.Summary)
	}

	model.response = structuredTestSummary("Completed Work", "merged summary")
	secondMessages := append([]llm.Message{}, first...)
	secondMessages = append(secondMessages,
		llm.NewUserMessage("delta user constraint keep rust files"),
		llm.Message{Role: llm.RoleUser, Name: "sdk_internal_require_done", Content: llm.TextContent("internal reminder must not enter delta")},
		llm.NewAssistantMessage("delta assistant found /tmp/new.go", nil),
		llm.NewUserMessage("delta user follow up"),
	)
	second, secondRes, err := svc.Compact(context.Background(), model, secondMessages)
	if err != nil {
		t.Fatalf("second Compact: %v", err)
	}
	if !secondRes.Compacted {
		t.Fatal("expected second summary compaction")
	}
	if store.ledger.Summary == nil || store.ledger.Summary.Version != 2 {
		t.Fatalf("second ledger summary = %#v", store.ledger.Summary)
	}
	if len(model.prompts) != 2 {
		t.Fatalf("prompt count = %d, want 2", len(model.prompts))
	}
	secondPrompt := model.prompts[1]
	if !strings.Contains(secondPrompt, "Previous Summary") || !strings.Contains(secondPrompt, "initial summary") {
		t.Fatalf("second prompt missing previous summary:\n%s", secondPrompt)
	}
	if !strings.Contains(secondPrompt, "Delta Messages") || !strings.Contains(secondPrompt, "delta assistant found /tmp/new.go") {
		t.Fatalf("second prompt missing delta:\n%s", secondPrompt)
	}
	if !strings.Contains(secondPrompt, "delta user constraint keep rust files") {
		t.Fatalf("second prompt missing non-retained user delta:\n%s", secondPrompt)
	}
	deltaSection := secondPrompt
	if start := strings.Index(deltaSection, "## Delta Messages"); start >= 0 {
		deltaSection = deltaSection[start:]
	}
	if end := strings.Index(deltaSection, endUntrustedMaterial); end >= 0 {
		deltaSection = deltaSection[:end]
	}
	if strings.Contains(deltaSection, "delta user follow up") {
		t.Fatalf("delta section included retained recent user message:\n%s", secondPrompt)
	}
	if strings.Contains(secondPrompt, "internal reminder must not enter delta") {
		t.Fatalf("second prompt included internal user-role delta:\n%s", secondPrompt)
	}
	if strings.Contains(secondPrompt, "covered old assistant answer") || strings.Contains(secondPrompt, "covered old user request") {
		t.Fatalf("second prompt included already-covered raw history:\n%s", secondPrompt)
	}
	if second[0].Name != CompactionSummaryMessageName || !strings.Contains(second[0].Content.PlainText(), "merged summary") {
		t.Fatalf("second compacted messages = %#v", second)
	}
}

func TestCompactSummaryExtractionFailureDoesNotSaveLedgerSummary(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-summary-fail")}
	model := &summaryPromptCaptureModel{response: "missing summary tags"}
	svc := NewService(&Config{
		Enabled:                true,
		SessionID:              "sess-summary-fail",
		LedgerStore:            store,
		SummaryPrompt:          "summarize now",
		KeepRecentUserMessages: 1,
	})
	messages := []llm.Message{
		llm.NewUserMessage("old request"),
		llm.NewAssistantMessage("old answer", nil),
		llm.NewUserMessage("latest"),
	}

	got, res, err := svc.Compact(context.Background(), model, messages)
	if err == nil {
		t.Fatal("expected summary extraction failure")
	}
	if res.Compacted {
		t.Fatalf("result = %#v, want not compacted", res)
	}
	if store.ledger.Summary != nil {
		t.Fatalf("ledger summary was saved on failure: %#v", store.ledger.Summary)
	}
	if len(got) != len(messages) || got[1].Content.PlainText() != messages[1].Content.PlainText() {
		t.Fatalf("history changed on failure: %#v", got)
	}
}

func TestIncrementalSummaryRejectsLedgerHashMismatch(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-summary-hash-mismatch")}
	model := &summaryPromptCaptureModel{response: structuredTestSummary("Completed Work", "initial summary")}
	svc := NewService(&Config{
		Enabled:                true,
		SessionID:              "sess-summary-hash-mismatch",
		LedgerStore:            store,
		KeepRecentUserMessages: 1,
	})
	first, _, err := svc.Compact(context.Background(), model, []llm.Message{
		llm.NewUserMessage("covered request"),
		llm.NewAssistantMessage("covered answer", nil),
		llm.NewUserMessage("latest retained request"),
	})
	if err != nil {
		t.Fatalf("initial Compact: %v", err)
	}
	store.ledger.Summary.SummaryHash = ContentHash("tampered summary")
	model.response = structuredTestSummary("Completed Work", "full rebuild after hash mismatch")
	secondMessages := append(append([]llm.Message(nil), first...), llm.NewAssistantMessage("delta after mismatch", nil))
	_, res, err := svc.Compact(context.Background(), model, secondMessages)
	if err != nil {
		t.Fatalf("second Compact: %v", err)
	}
	if len(model.prompts) != 2 {
		t.Fatalf("prompt count = %d, want 2", len(model.prompts))
	}
	if strings.Contains(model.prompts[1], "## Delta Messages") {
		t.Fatalf("hash-mismatched ledger was used incrementally:\n%s", model.prompts[1])
	}
	if !containsWarningText(res.Warnings, "hash mismatch") || !containsWarningText(res.Warnings, "full rebuild") {
		t.Fatalf("hash mismatch warnings = %#v", res.Warnings)
	}
}

func TestIncrementalSummaryRejectsCoverageMismatch(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-summary-coverage-mismatch")}
	model := &summaryPromptCaptureModel{response: structuredTestSummary("Completed Work", "initial summary")}
	svc := NewService(&Config{
		Enabled:                true,
		SessionID:              "sess-summary-coverage-mismatch",
		LedgerStore:            store,
		KeepRecentUserMessages: 1,
	})
	first, _, err := svc.Compact(context.Background(), model, []llm.Message{
		llm.NewUserMessage("covered request"),
		llm.NewAssistantMessage("covered answer", nil),
		llm.NewUserMessage("latest retained request"),
	})
	if err != nil {
		t.Fatalf("initial Compact: %v", err)
	}
	store.ledger.Summary.CoveredEndKey = StableMessageKey(MessageKeyInput{
		Role:           string(llm.RoleAssistant),
		OriginalText:   "different covered end",
		FirstSeenIndex: 99,
	})
	model.response = structuredTestSummary("Completed Work", "full rebuild after coverage mismatch")
	secondMessages := append(append([]llm.Message(nil), first...), llm.NewAssistantMessage("delta after mismatch", nil))
	_, res, err := svc.Compact(context.Background(), model, secondMessages)
	if err != nil {
		t.Fatalf("second Compact: %v", err)
	}
	if strings.Contains(model.prompts[1], "## Delta Messages") {
		t.Fatalf("coverage-mismatched ledger was used incrementally:\n%s", model.prompts[1])
	}
	if !containsWarningText(res.Warnings, "coverage mismatch") || !containsWarningText(res.Warnings, "full rebuild") {
		t.Fatalf("coverage mismatch warnings = %#v", res.Warnings)
	}
}

func TestIncrementalSummaryUsesStableCoveredEndIdentity(t *testing.T) {
	store := &memoryLedgerStore{ledger: NewLedger("sess-summary-stable-identity")}
	model := &summaryPromptCaptureModel{response: structuredTestSummary("Completed Work", "initial stable summary")}
	svc := NewService(&Config{
		Enabled:                true,
		SessionID:              "sess-summary-stable-identity",
		LedgerStore:            store,
		KeepRecentUserMessages: 1,
	})
	first, _, err := svc.Compact(context.Background(), model, []llm.Message{
		llm.NewUserMessage("covered request"),
		llm.NewAssistantMessage("covered answer", nil),
		llm.NewUserMessage("latest retained request"),
	})
	if err != nil {
		t.Fatalf("initial Compact: %v", err)
	}
	if store.ledger.Summary == nil || strings.TrimSpace(store.ledger.Summary.CheckpointID) == "" {
		t.Fatalf("initial ledger summary has no stable checkpoint identity: %#v", store.ledger.Summary)
	}
	if !strings.Contains(first[0].Content.PlainText(), store.ledger.Summary.CheckpointID) {
		t.Fatalf("summary message does not carry ledger checkpoint identity: %#v", first[0])
	}

	model.response = structuredTestSummary("Completed Work", "merged stable summary")
	shifted := append([]llm.Message{llm.NewSystemMessage("restored system context")}, first...)
	shifted = append(shifted, llm.NewAssistantMessage("delta after shifted summary index", nil))
	_, res, err := svc.Compact(context.Background(), model, shifted)
	if err != nil {
		t.Fatalf("second Compact: %v", err)
	}
	if !strings.Contains(model.prompts[1], "## Delta Messages") || !strings.Contains(model.prompts[1], "delta after shifted summary index") {
		t.Fatalf("stable checkpoint identity did not locate delta boundary:\n%s", model.prompts[1])
	}
	if containsWarningText(res.Warnings, "integrity mismatch") {
		t.Fatalf("shifted but identity-matching history triggered rebuild: %#v", res.Warnings)
	}
}

func TestCompactionSourceSnapshotRecordedOnlyWhenPersisted(t *testing.T) {
	t.Run("persisted", func(t *testing.T) {
		store := &memoryLedgerStore{ledger: NewLedger("sess-summary-snapshot")}
		writes := 0
		svc := NewService(&Config{
			Enabled:     true,
			SessionID:   "sess-summary-snapshot",
			LedgerStore: store,
			SummarySourceWriter: ArtifactWriterFunc(func(_ context.Context, req ArtifactRequest) (ArtifactResult, error) {
				writes++
				if strings.TrimSpace(req.Content) == "" {
					t.Fatal("summary source snapshot content is empty")
				}
				return ArtifactResult{Path: ".goode/truncated/summary-source.md"}, nil
			}),
		})
		_, res, err := svc.Compact(context.Background(), mockCompactModel{response: structuredTestSummary("Completed Work", "snapshot summary")}, []llm.Message{
			llm.NewUserMessage("preserve source evidence"),
			llm.NewAssistantMessage("verified source state", nil),
		})
		if err != nil {
			t.Fatalf("Compact: %v", err)
		}
		if writes != 1 {
			t.Fatalf("snapshot writes = %d, want 1", writes)
		}
		if store.ledger.Summary == nil || store.ledger.Summary.SourceSnapshot != ".goode/truncated/summary-source.md" {
			t.Fatalf("ledger summary snapshot = %#v", store.ledger.Summary)
		}
		if res.SnapshotPath != ".goode/truncated/summary-source.md" {
			t.Fatalf("result snapshot path = %q", res.SnapshotPath)
		}
	})

	t.Run("failed", func(t *testing.T) {
		store := &memoryLedgerStore{ledger: NewLedger("sess-summary-snapshot-fail")}
		svc := NewService(&Config{
			Enabled:     true,
			SessionID:   "sess-summary-snapshot-fail",
			LedgerStore: store,
			SummarySourceWriter: ArtifactWriterFunc(func(context.Context, ArtifactRequest) (ArtifactResult, error) {
				return ArtifactResult{}, errors.New("snapshot disk full")
			}),
		})
		_, res, err := svc.Compact(context.Background(), mockCompactModel{response: structuredTestSummary("Completed Work", "snapshot failure summary")}, []llm.Message{
			llm.NewUserMessage("preserve source evidence"),
		})
		if err != nil {
			t.Fatalf("Compact: %v", err)
		}
		if store.ledger.Summary == nil || store.ledger.Summary.SourceSnapshot != "" || res.SnapshotPath != "" {
			t.Fatalf("failed snapshot was recorded: ledger=%#v result=%#v", store.ledger.Summary, res)
		}
		if !containsWarningText(res.Warnings, "source snapshot") || !containsWarningText(res.Warnings, "snapshot disk full") {
			t.Fatalf("snapshot failure warnings = %#v", res.Warnings)
		}
	})
}

func containsWarningText(warnings []string, needle string) bool {
	needle = strings.ToLower(strings.TrimSpace(needle))
	for _, warning := range warnings {
		if strings.Contains(strings.ToLower(warning), needle) {
			return true
		}
	}
	return false
}

func TestSummaryDeltaTruncationPreservesChinesePath(t *testing.T) {
	const path = "/工作区/项目/文件.go"
	text := strings.Repeat("a", 1300) + " " + path + " error=E42"
	got := truncateSummaryDeltaText(text)
	if !utf8.ValidString(got) || strings.ContainsRune(got, utf8.RuneError) {
		t.Fatalf("summary delta is not valid UTF-8: %q", got)
	}
	if !strings.Contains(got, path) {
		t.Fatalf("summary delta lost exact Chinese path %q: %q", path, got)
	}
}
