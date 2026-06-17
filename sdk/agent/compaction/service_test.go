package compaction

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type mockCompactModel struct {
	response string
}

func (m mockCompactModel) Provider() string { return "mock" }
func (m mockCompactModel) Model() string    { return "mock" }
func (m mockCompactModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{
		Content: llm.TextContent(m.response),
		Usage:   &llm.Usage{CompletionTokens: 100},
	}, nil
}

type blockingCompactModel struct{}

func (m blockingCompactModel) Provider() string { return "mock" }
func (m blockingCompactModel) Model() string    { return "mock" }
func (m blockingCompactModel) Invoke(ctx context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

type promptCaptureModel struct {
	modelID    string
	response   string
	lastPrompt string
	lastRoles  []llm.Role
}

func (m *promptCaptureModel) Provider() string { return "mock" }
func (m *promptCaptureModel) Model() string {
	if m.modelID == "" {
		return "mock"
	}
	return m.modelID
}
func (m *promptCaptureModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.lastRoles = m.lastRoles[:0]
	for _, msg := range req.Messages {
		m.lastRoles = append(m.lastRoles, msg.Role)
	}
	if n := len(req.Messages); n > 0 {
		m.lastPrompt = req.Messages[n-1].Content.PlainText()
	}
	return &llm.Completion{Content: llm.TextContent(m.response)}, nil
}

func TestWithSummaryPrefix_NoDuplicate(t *testing.T) {
	plain := "Some summary text"
	got := WithSummaryPrefix(plain)
	if !strings.HasPrefix(got, DefaultSummaryPrefix) {
		t.Fatalf("expected prefix, got %q", got[:50])
	}
	// Calling again on already-prefixed text should not duplicate.
	double := WithSummaryPrefix(got)
	if double != got {
		t.Fatalf("expected no duplicate prefix, got length %d vs %d", len(double), len(got))
	}
}

func TestDefaultConfigUsesFiveMinuteCompactionTimeout(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.CompactionTimeout != 300*time.Second {
		t.Fatalf("default compaction timeout = %s, want 300s", cfg.CompactionTimeout)
	}
}

func TestSelectRecentUserMessages_SkipsCompactionSummaryMessage(t *testing.T) {
	messages := []llm.Message{
		llm.NewUserMessage("first real question"),
		{Role: llm.RoleAssistant, Content: llm.TextContent("answer 1")},
		newCompactionSummaryMessage(WithSummaryPrefix("compacted summary")),
		llm.NewUserMessage("second real question"),
		{Role: llm.RoleAssistant, Content: llm.TextContent("answer 2")},
		llm.NewUserMessage("third real question"),
	}

	recent := SelectRecentUserMessages(messages, 3)
	if len(recent) != 3 {
		t.Fatalf("expected 3 recent user messages, got %d", len(recent))
	}
	// The compaction summary message should be skipped.
	for _, m := range recent {
		if m.Name == compactionSummaryMessageName {
			t.Fatal("expected summary message to be skipped")
		}
	}
	// Should be in chronological order: first, second, third.
	if recent[0].Content.PlainText() != "first real question" {
		t.Fatalf("expected first message to be first real question, got %q", recent[0].Content.PlainText())
	}
	if recent[2].Content.PlainText() != "third real question" {
		t.Fatalf("expected third message to be third real question, got %q", recent[2].Content.PlainText())
	}
}

func TestSelectRecentUserMessages_DoesNotSkipLegitimatePrefixedUserMessage(t *testing.T) {
	messages := []llm.Message{
		llm.NewUserMessage(DefaultSummaryPrefix + " user pasted quoted text"),
		llm.NewUserMessage("latest"),
	}

	recent := SelectRecentUserMessages(messages, 2)
	if len(recent) != 2 {
		t.Fatalf("expected 2 user messages, got %d", len(recent))
	}
	if got := recent[0].Content.PlainText(); got != DefaultSummaryPrefix+" user pasted quoted text" {
		t.Fatalf("expected prefixed user message to be kept, got %q", got)
	}
}

func TestExtractSummary_UsesLastSummaryBlock(t *testing.T) {
	text := "<summary>first summary</summary>\nquoted example\n<summary>final summary</summary>"
	if got := ExtractSummary(text); got != "final summary" {
		t.Fatalf("expected last summary block, got %q", got)
	}
}

func TestExtractSummary_UsesLastSummaryOrStructuredBlock(t *testing.T) {
	text := "<summary>first summary</summary>\n<compaction_summary>final summary</compaction_summary>"
	if got := ExtractSummary(text); got != "final summary" {
		t.Fatalf("expected last structured summary block, got %q", got)
	}
}

func TestExtractSummary_NoTagsReturnsEmpty(t *testing.T) {
	if got := ExtractSummary("plain text without summary tags"); got != "" {
		t.Fatalf("expected empty summary when tags are missing, got %q", got)
	}
}

func TestExtractSummary_LogsWarningOnEmptyStructuredSummary(t *testing.T) {
	var buf bytes.Buffer
	origOut := log.Writer()
	origFlags := log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(origOut)
		log.SetFlags(origFlags)
	})

	if got := ExtractSummary("<compaction_summary>  \n\t </compaction_summary>"); got != "" {
		t.Fatalf("expected empty summary for empty structured block, got %q", got)
	}
	if !strings.Contains(buf.String(), "summary extraction failed") {
		t.Fatalf("expected warning log for empty structured block, got %q", buf.String())
	}
}

func TestPrepareForSummary_AddsFallbackWhenEverythingFiltered(t *testing.T) {
	messages := []llm.Message{{
		Role:      llm.RoleAssistant,
		ToolCalls: []llm.ToolCall{{ID: "call-1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{}`}}},
	}}

	prepared := prepareForSummary(messages)
	if len(prepared) != 1 {
		t.Fatalf("expected fallback message, got %d messages", len(prepared))
	}
	if prepared[0].Role != llm.RoleUser {
		t.Fatalf("expected fallback role user, got %s", prepared[0].Role)
	}
	if got := prepared[0].Content.PlainText(); got != fallbackSummaryContext {
		t.Fatalf("expected fallback context %q, got %q", fallbackSummaryContext, got)
	}
}

func TestPrepareForSummary_StripsAssistantToolCallsWhenDestroyedResultFiltered(t *testing.T) {
	messages := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewUserMessage("write file"),
		{
			Role:    llm.RoleAssistant,
			Content: llm.TextContent("I will write the file."),
			ToolCalls: []llm.ToolCall{{
				ID:       "call-write",
				Type:     "function",
				Function: llm.FunctionCall{Name: "write", Arguments: `{"filePath":"x"}`},
			}},
		},
		{
			Role:       llm.RoleTool,
			ToolCallID: "call-write",
			ToolName:   "write",
			Content:    llm.TextContent("[destroyed tool result]"),
			Destroyed:  true,
		},
	}

	prepared := prepareForSummary(messages)
	if len(prepared) != 3 {
		t.Fatalf("prepared message count = %d, want 3 (%#v)", len(prepared), prepared)
	}
	last := prepared[2]
	if last.Role != llm.RoleAssistant {
		t.Fatalf("last role = %s, want assistant", last.Role)
	}
	if len(last.ToolCalls) != 0 {
		t.Fatalf("expected stripped tool calls after destroyed result filtering, got %#v", last.ToolCalls)
	}
	if got := last.Content.PlainText(); !strings.Contains(got, "write the file") {
		t.Fatalf("assistant text should be preserved, got %q", got)
	}
}

func TestPrepareForSummary_KeepsCompleteToolCallResultBlock(t *testing.T) {
	messages := []llm.Message{
		llm.NewUserMessage("read file"),
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{{
				ID:       "call-read",
				Type:     "function",
				Function: llm.FunctionCall{Name: "read", Arguments: `{"filePath":"x"}`},
			}},
		},
		{
			Role:       llm.RoleTool,
			ToolCallID: "call-read",
			ToolName:   "read",
			Content:    llm.TextContent("ok"),
		},
	}

	prepared := prepareForSummary(messages)
	if len(prepared) != 3 {
		t.Fatalf("prepared message count = %d, want 3 (%#v)", len(prepared), prepared)
	}
	if len(prepared[1].ToolCalls) != 1 {
		t.Fatalf("expected assistant tool call to remain, got %#v", prepared[1].ToolCalls)
	}
	if prepared[2].Role != llm.RoleTool || prepared[2].ToolCallID != "call-read" {
		t.Fatalf("expected contiguous tool result to remain, got %#v", prepared[2])
	}
}

func TestPrepareForSummary_ProducesProviderValidHistoryWhenToolPairsAreInvalid(t *testing.T) {
	messages := []llm.Message{
		llm.NewToolMessage("orphan-before", "read", llm.TextContent("orphan result"), false),
		llm.NewUserMessage("inspect files"),
		{
			Role:    llm.RoleAssistant,
			Content: llm.TextContent("I will inspect two files."),
			ToolCalls: []llm.ToolCall{
				{ID: "call-read-a", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{"filePath":"a.go"}`}},
				{ID: "call-read-b", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{"filePath":"b.go"}`}},
			},
		},
		llm.NewToolMessage("call-read-a", "read", llm.TextContent("a.go contents"), false),
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{
				{ID: "call-grep", Type: "function", Function: llm.FunctionCall{Name: "grep", Arguments: `{"pattern":"needle"}`}},
			},
		},
		llm.NewToolMessage("call-grep", "grep", llm.TextContent("needle hit"), false),
	}

	prepared := prepareForSummary(messages)
	assertProviderValidSummaryHistory(t, prepared)

	if len(prepared) != 4 {
		t.Fatalf("prepared message count = %d, want 4 (%#v)", len(prepared), prepared)
	}
	if prepared[0].Role == llm.RoleTool {
		t.Fatalf("orphan tool result should be dropped, got %#v", prepared[0])
	}
	if len(prepared[1].ToolCalls) != 0 {
		t.Fatalf("incomplete assistant tool call block should be stripped, got %#v", prepared[1].ToolCalls)
	}
	if len(prepared[2].ToolCalls) == 0 {
		t.Fatalf("complete assistant tool call should remain, got no tool calls")
	}
	if prepared[2].ToolCalls[0].ID != "call-grep" || prepared[3].ToolCallID != "call-grep" {
		t.Fatalf("complete tool call/result block should remain, got %#v %#v", prepared[2], prepared[3])
	}
}

func assertProviderValidSummaryHistory(t *testing.T, messages []llm.Message) {
	t.Helper()
	pending := map[string]bool{}
	for i, msg := range messages {
		if msg.Role == llm.RoleTool {
			if _, ok := pending[msg.ToolCallID]; !ok {
				t.Fatalf("message %d is orphan tool result %#v", i, msg)
			}
			if pending[msg.ToolCallID] {
				t.Fatalf("message %d duplicates tool result id %q", i, msg.ToolCallID)
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
		if msg.Role != llm.RoleAssistant || len(msg.ToolCalls) == 0 {
			continue
		}
		for _, call := range msg.ToolCalls {
			if strings.TrimSpace(call.ID) == "" {
				t.Fatalf("message %d has empty tool call id", i)
			}
			if _, exists := pending[call.ID]; exists {
				t.Fatalf("message %d duplicates tool call id %q", i, call.ID)
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

func TestIsOverflow_UsesContextWindowMinusReserve(t *testing.T) {
	svc := NewService(&Config{
		Enabled:             true,
		ContextWindow:       100,
		ThresholdRatio:      0.99,
		ReserveOutputTokens: 10,
	})
	if !svc.IsOverflow(&llm.Usage{PromptTokens: 90}) {
		t.Fatal("expected prompt_tokens=90 to overflow when reserve_output_tokens=10")
	}
	if svc.IsOverflow(&llm.Usage{PromptTokens: 89}) {
		t.Fatal("expected prompt_tokens=89 to remain below overflow threshold")
	}
}

func TestIsOverflow_DisabledCompactionReturnsFalse(t *testing.T) {
	svc := NewService(&Config{
		Enabled:             false,
		ContextWindow:       100,
		ReserveOutputTokens: 10,
	})
	if svc.IsOverflow(&llm.Usage{PromptTokens: 99}) {
		t.Fatal("expected disabled compaction to skip overflow checks")
	}
}

func TestCompact_KeepsRecentUsersAndPrefix(t *testing.T) {
	model := mockCompactModel{response: "<summary>test summary</summary>"}
	svc := NewService(&Config{
		Enabled:                true,
		ContextWindow:          128000,
		ThresholdRatio:         0.85,
		SummaryPrompt:          DefaultSummaryPrompt,
		KeepRecentUserMessages: 2,
	})

	messages := []llm.Message{
		llm.NewUserMessage("old question"),
		{Role: llm.RoleAssistant, Content: llm.TextContent("old answer")},
		llm.NewUserMessage("recent question 1"),
		{Role: llm.RoleAssistant, Content: llm.TextContent("recent answer 1")},
		llm.NewUserMessage("recent question 2"),
	}

	newMsgs, res, err := svc.Compact(context.Background(), model, messages)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !res.Compacted {
		t.Fatal("expected compacted=true")
	}
	// First message should be the summary with prefix + compaction marker.
	if !strings.HasPrefix(newMsgs[0].Content.PlainText(), DefaultSummaryPrefix) {
		t.Fatal("expected first message to have summary prefix")
	}
	if newMsgs[0].Name != compactionSummaryMessageName {
		t.Fatalf("expected first message marker %q, got %q", compactionSummaryMessageName, newMsgs[0].Name)
	}
	// Should keep 2 recent user messages.
	userCount := 0
	for _, m := range newMsgs[1:] {
		if m.Role == llm.RoleUser {
			userCount++
		}
	}
	if userCount != 2 {
		t.Fatalf("expected 2 recent user messages, got %d", userCount)
	}
}

func TestCompact_PopulatesSummaryTelemetry(t *testing.T) {
	model := mockCompactModel{response: "<summary>test summary</summary>"}
	svc := NewService(&Config{
		Enabled:                true,
		ContextWindow:          128000,
		ThresholdRatio:         0.85,
		SummaryPrompt:          DefaultSummaryPrompt,
		KeepRecentUserMessages: 1,
	})

	_, res, err := svc.Compact(context.Background(), model, []llm.Message{
		llm.NewUserMessage("old question"),
		llm.NewAssistantMessage("old answer", nil),
		llm.NewUserMessage("recent question"),
	})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if res.Trigger != "manual" {
		t.Fatalf("trigger = %q, want manual", res.Trigger)
	}
	if res.Watermark != "summarize" {
		t.Fatalf("watermark = %q, want summarize", res.Watermark)
	}
	if len(res.TiersApplied) != 1 || res.TiersApplied[0] != "summarize" {
		t.Fatalf("tiers = %#v, want [summarize]", res.TiersApplied)
	}
	if res.Usage == nil || res.Usage.CompletionTokens != 100 {
		t.Fatalf("usage = %#v, want completion tokens from compaction model", res.Usage)
	}
	if res.OriginalTokens <= 0 {
		t.Fatalf("original tokens should be populated, got %d", res.OriginalTokens)
	}
	if res.NewTokens <= 0 {
		t.Fatalf("new tokens should be populated, got %d", res.NewTokens)
	}
}

func TestCompact_UsesDedicatedCompactionRequestInsteadOfRawHistory(t *testing.T) {
	model := &promptCaptureModel{response: "<summary>dedicated summary</summary>"}
	svc := NewService(&Config{
		Enabled:                true,
		ContextWindow:          128000,
		ThresholdRatio:         0.85,
		SummaryPrompt:          "summarize retained material",
		KeepRecentUserMessages: 1,
	})

	messages := []llm.Message{
		llm.NewSystemMessage("system contract: preserve cwd /repo and tests"),
		llm.NewUserMessage("old original goal with /repo/main.go"),
		llm.NewAssistantMessage("I am going to inspect files and then continue", nil),
		llm.NewToolMessage("call-read", "read", llm.TextContent(strings.Repeat("large raw output ", 200)+"/repo/main.go"), false),
		llm.NewAssistantMessage("routine narration with no durable fact", nil),
		llm.NewToolMessage("call-test", "bash", llm.TextContent("go test ./... failed: exit code 1 at /repo/main.go"), true),
		llm.NewUserMessage("latest user goal"),
	}

	_, _, err := svc.Compact(context.Background(), model, messages)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if got := len(model.lastRoles); got != 1 {
		t.Fatalf("compaction model received %d messages, want dedicated single request", got)
	}
	if model.lastRoles[0] != llm.RoleUser {
		t.Fatalf("compaction request role = %s, want user", model.lastRoles[0])
	}
	prompt := model.lastPrompt
	for _, want := range []string{
		"internal context compaction pipeline",
		"summarize retained material",
		"system contract: preserve cwd /repo and tests",
		"latest user goal",
		"tool error bash",
		"go test ./... failed: exit code 1 at /repo/main.go",
		"Recent Tool Results",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("compaction prompt missing %q:\n%s", want, prompt)
		}
	}
	for _, forbidden := range []string{
		"I am going to inspect files and then continue",
		"routine narration with no durable fact",
		strings.Repeat("large raw output ", 20),
	} {
		if strings.Contains(prompt, forbidden) {
			t.Fatalf("compaction prompt included raw/non-durable history %q:\n%s", forbidden, prompt)
		}
	}
}

func TestCompact_AppendsToolContext(t *testing.T) {
	model := mockCompactModel{response: "<summary>tool summary</summary>"}
	svc := NewService(&Config{
		Enabled:                       true,
		ContextWindow:                 128000,
		ThresholdRatio:                0.85,
		SummaryPrompt:                 DefaultSummaryPrompt,
		KeepRecentUserMessages:        1,
		MinSummaryCharsForToolContext: 1,
	})

	messages := []llm.Message{
		llm.NewUserMessage("question"),
		{Role: llm.RoleAssistant, Content: llm.TextContent("calling tool")},
		{Role: llm.RoleTool, ToolName: "read_file", Content: llm.TextContent("file contents here")},
		llm.NewUserMessage("follow up"),
	}

	newMsgs, res, err := svc.Compact(context.Background(), model, messages)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !res.Compacted {
		t.Fatal("expected compacted=true")
	}
	// The summary in the first message should contain tool context.
	summaryText := newMsgs[0].Content.PlainText()
	if !strings.Contains(summaryText, "Recent Tool Results") {
		t.Fatal("expected summary to contain tool context snapshot")
	}
	if !strings.Contains(summaryText, "read_file") {
		t.Fatal("expected summary to reference read_file tool")
	}
}

func TestCompact_SkipsToolContextForShortSummary(t *testing.T) {
	model := mockCompactModel{response: "<summary>short</summary>"}
	svc := NewService(&Config{
		Enabled:                       true,
		ContextWindow:                 128000,
		ThresholdRatio:                0.85,
		SummaryPrompt:                 DefaultSummaryPrompt,
		KeepRecentUserMessages:        1,
		MinSummaryCharsForToolContext: 20,
	})

	messages := []llm.Message{
		llm.NewUserMessage("question"),
		{Role: llm.RoleAssistant, Content: llm.TextContent("calling tool")},
		{Role: llm.RoleTool, ToolName: "read_file", Content: llm.TextContent("file contents here")},
	}

	newMsgs, _, err := svc.Compact(context.Background(), model, messages)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	summaryText := newMsgs[0].Content.PlainText()
	if strings.Contains(summaryText, "Recent Tool Results") {
		t.Fatal("expected short summary to skip tool context snapshot")
	}
}

func TestCompact_UsesRuneCountForToolContextThreshold(t *testing.T) {
	model := mockCompactModel{response: "<summary>界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界界</summary>"}
	svc := NewService(&Config{
		Enabled:                       true,
		ContextWindow:                 128000,
		ThresholdRatio:                0.85,
		SummaryPrompt:                 DefaultSummaryPrompt,
		KeepRecentUserMessages:        1,
		MinSummaryCharsForToolContext: 100,
	})

	messages := []llm.Message{
		llm.NewUserMessage("question"),
		{Role: llm.RoleTool, ToolName: "read_file", Content: llm.TextContent("tool output")},
	}

	newMsgs, _, err := svc.Compact(context.Background(), model, messages)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	summaryText := newMsgs[0].Content.PlainText()
	if strings.Contains(summaryText, "Recent Tool Results") {
		t.Fatal("expected rune-count threshold to skip tool context snapshot")
	}
}

func TestCompact_ErrorsWhenSummaryTagsMissing(t *testing.T) {
	model := mockCompactModel{response: "plain text summary without tags"}
	svc := NewService(&Config{
		Enabled:                true,
		ContextWindow:          128000,
		ThresholdRatio:         0.85,
		SummaryPrompt:          DefaultSummaryPrompt,
		KeepRecentUserMessages: 1,
	})

	messages := []llm.Message{llm.NewUserMessage("hello")}
	newMsgs, res, err := svc.Compact(context.Background(), model, messages)
	if err == nil {
		t.Fatal("expected error when summary tags are missing")
	}
	if !strings.Contains(err.Error(), "summary extraction failed") {
		t.Fatalf("expected summary extraction failure error, got %v", err)
	}
	if res.Compacted {
		t.Fatal("expected compacted=false on extraction failure")
	}
	if len(newMsgs) != len(messages) {
		t.Fatalf("expected original messages on failure, got %d", len(newMsgs))
	}
}

func TestCompact_UsesCompactionTimeout(t *testing.T) {
	svc := NewService(&Config{
		Enabled:                true,
		ContextWindow:          128000,
		ThresholdRatio:         0.85,
		SummaryPrompt:          DefaultSummaryPrompt,
		KeepRecentUserMessages: 1,
		CompactionTimeout:      15 * time.Millisecond,
	})

	messages := []llm.Message{llm.NewUserMessage("hello")}
	start := time.Now()
	_, _, err := svc.Compact(context.Background(), blockingCompactModel{}, messages)
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline exceeded, got %v", err)
	}
	if time.Since(start) > 500*time.Millisecond {
		t.Fatalf("expected timeout to return quickly, took %s", time.Since(start))
	}
}

func TestCompact_UsesModelAwareSummaryPrompt(t *testing.T) {
	model := &promptCaptureModel{
		modelID:  "mock-small",
		response: "<summary>ok</summary>",
	}
	svc := NewService(&Config{
		Enabled:        true,
		ContextWindow:  128000,
		ThresholdRatio: 0.85,
		SummaryPrompt: func(modelID string) string {
			if modelID == "mock-small" {
				return "short prompt"
			}
			return "default prompt"
		},
		KeepRecentUserMessages: 1,
	})

	_, _, err := svc.Compact(context.Background(), model, []llm.Message{llm.NewUserMessage("hello")})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !strings.Contains(model.lastPrompt, "short prompt") {
		t.Fatalf("expected model-aware prompt, got %q", model.lastPrompt)
	}
	if !strings.Contains(model.lastPrompt, "internal context compaction pipeline") {
		t.Fatalf("expected dedicated compaction request, got %q", model.lastPrompt)
	}
}

func TestToolContextSnapshotStopsAtFirstOverflow(t *testing.T) {
	longText := strings.Repeat("z", 500)
	messages := []llm.Message{
		{
			Role:     llm.RoleTool,
			ToolName: "oldest",
			Content:  llm.TextContent(longText),
		},
		{
			Role:     llm.RoleTool,
			ToolName: "overflow-mid-" + strings.Repeat("m", 320),
			Content:  llm.TextContent(longText),
		},
		{
			Role:     llm.RoleTool,
			ToolName: "recent-b-" + strings.Repeat("b", 560),
			Content:  llm.TextContent(longText),
		},
		{
			Role:     llm.RoleTool,
			ToolName: "recent-a-" + strings.Repeat("a", 560),
			Content:  llm.TextContent(longText),
		},
	}

	snap := toolContextSnapshot(messages, nil, 0, 0)
	if snap == "" {
		t.Fatal("expected non-empty snapshot")
	}
	if len(snap) > 2000 {
		t.Fatalf("snapshot exceeds maxChars: %d", len(snap))
	}
	// Most recent entries should be present; the overflow entry should be excluded.
	if !strings.Contains(snap, "recent-a") {
		t.Error("expected recent-a to be included (most recent)")
	}
	if !strings.Contains(snap, "recent-b") {
		t.Error("expected recent-b to be included")
	}
	// overflow-mid should cause total to exceed 2000; verify its exclusion.
	if strings.Contains(snap, "overflow-mid") {
		t.Error("expected overflow-mid to be excluded due to char limit")
	}
}

func TestToolContextSnapshotPrioritizesProtectedTools(t *testing.T) {
	messages := []llm.Message{{
		Role:     llm.RoleTool,
		ToolName: "skill",
		Content:  llm.TextContent("critical state"),
	}}
	for i := 0; i < 6; i++ {
		messages = append(messages, llm.Message{
			Role:     llm.RoleTool,
			ToolName: fmt.Sprintf("tool_%d", i),
			Content:  llm.TextContent(strings.Repeat("x", 160)),
		})
	}

	protected := map[string]struct{}{"skill": {}}
	snap := toolContextSnapshot(messages, protected, 0, 0)
	if !strings.Contains(snap, "**skill**") {
		t.Fatalf("expected protected tool to be retained in snapshot, got %q", snap)
	}
}

func TestNewService_DefaultsToolSnapshotLimits(t *testing.T) {
	svc := NewService(&Config{
		Enabled:                true,
		ContextWindow:          128000,
		ThresholdRatio:         0.85,
		ToolSnapshotMaxEntries: 0,
		ToolSnapshotMaxChars:   0,
	})
	if svc.Config.ToolSnapshotMaxEntries != DefaultToolSnapshotMaxEntries {
		t.Fatalf("expected default snapshot max entries %d, got %d", DefaultToolSnapshotMaxEntries, svc.Config.ToolSnapshotMaxEntries)
	}
	if svc.Config.ToolSnapshotMaxChars != DefaultToolSnapshotMaxChars {
		t.Fatalf("expected default snapshot max chars %d, got %d", DefaultToolSnapshotMaxChars, svc.Config.ToolSnapshotMaxChars)
	}
}

func TestToolContextSnapshotRespectsConfiguredEntryLimit(t *testing.T) {
	messages := []llm.Message{
		{Role: llm.RoleTool, ToolName: "oldest", Content: llm.TextContent("1")},
		{Role: llm.RoleTool, ToolName: "mid", Content: llm.TextContent("2")},
		{Role: llm.RoleTool, ToolName: "newest", Content: llm.TextContent("3")},
	}

	snap := toolContextSnapshot(messages, nil, 2, 1000)
	if strings.Count(snap, "- **") != 2 {
		t.Fatalf("expected exactly 2 tool entries, got snapshot %q", snap)
	}
	if !strings.Contains(snap, "**newest**") {
		t.Fatalf("expected newest tool to be included, got %q", snap)
	}
	if !strings.Contains(snap, "**mid**") {
		t.Fatalf("expected second-most-recent tool to be included, got %q", snap)
	}
	if strings.Contains(snap, "**oldest**") {
		t.Fatalf("expected oldest tool to be excluded by entry limit, got %q", snap)
	}
}

func TestToolContextSnapshotRespectsConfiguredCharLimit(t *testing.T) {
	text := strings.Repeat("x", 30)
	messages := []llm.Message{
		{Role: llm.RoleTool, ToolName: "older", Content: llm.TextContent(text)},
		{Role: llm.RoleTool, ToolName: "newer", Content: llm.TextContent(text)},
	}

	headerLen := len("## Recent Tool Results\n")
	firstLineLen := len(fmt.Sprintf("- **%s**: %s\n", "newer", text))
	maxChars := headerLen + firstLineLen

	snap := toolContextSnapshot(messages, nil, 6, maxChars)
	if !strings.Contains(snap, "**newer**") {
		t.Fatalf("expected most recent tool to be included, got %q", snap)
	}
	if strings.Contains(snap, "**older**") {
		t.Fatalf("expected older tool to be excluded by char limit, got %q", snap)
	}
}
