package compaction

import (
	"context"
	"strings"
	"testing"

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
	model := &summaryPromptCaptureModel{response: "<summary>initial summary</summary>"}
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

	model.response = "<summary>merged summary</summary>"
	secondMessages := append([]llm.Message{}, first...)
	secondMessages = append(secondMessages,
		llm.NewUserMessage("delta user constraint keep rust files"),
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
	if !strings.Contains(secondPrompt, "Previous summary") || !strings.Contains(secondPrompt, "initial summary") {
		t.Fatalf("second prompt missing previous summary:\n%s", secondPrompt)
	}
	if !strings.Contains(secondPrompt, "Delta messages") || !strings.Contains(secondPrompt, "delta assistant found /tmp/new.go") {
		t.Fatalf("second prompt missing delta:\n%s", secondPrompt)
	}
	if !strings.Contains(secondPrompt, "delta user constraint keep rust files") {
		t.Fatalf("second prompt missing non-retained user delta:\n%s", secondPrompt)
	}
	if strings.Contains(secondPrompt, "delta user follow up") {
		t.Fatalf("second prompt included retained recent user message:\n%s", secondPrompt)
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
