package agent

import (
	"context"
	"testing"

	sdkaccounting "github.com/timwhitez/agent-sdk-golang/sdk/accounting"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type accountingContractModel struct{ calls int }

func (m *accountingContractModel) Provider() string { return "fixture" }
func (m *accountingContractModel) Model() string    { return "fixture" }

func (m *accountingContractModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls == 1 {
		return &llm.Completion{
			ToolCalls: []llm.ToolCall{{
				ID:   "call-accounting-1",
				Type: "function",
				Function: llm.FunctionCall{
					Name:      "echo",
					Arguments: `{"text":"secret command-like args"}`,
				},
			}},
			Usage:      llm.NewProviderUsage(20, 2, 22),
			ResponseID: "resp-accounting-1",
		}, nil
	}
	return &llm.Completion{
		Content:    llm.TextContent("done"),
		Usage:      llm.NewProviderUsage(30, 3, 33),
		ResponseID: "resp-accounting-2",
	}, nil
}

type estimatedUsageAccountingModel struct{}

func (estimatedUsageAccountingModel) Provider() string { return "fixture" }
func (estimatedUsageAccountingModel) Model() string    { return "fixture-estimated-usage" }
func (estimatedUsageAccountingModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{
		Content: llm.TextContent("done"),
		Usage: &llm.Usage{
			PromptTokens:          40,
			CompletionTokens:      2,
			TotalTokens:           42,
			PromptTokensSource:    llm.PromptTokensSourceEstimate,
			PromptTokensSemantics: llm.PromptTokensSemanticsTotalInputV1,
		},
	}, nil
}

func TestAgentEmitsOneAccountingEventPerUsageAndToolResult(t *testing.T) {
	echo := tools.Func("echo", "echo fixture", func(_ context.Context, args struct {
		Text string `json:"text"`
	}, _ *tools.Container) (any, error) {
		return "tool raw value must stay out of accounting", nil
	})
	estimator := sdkaccounting.Estimator{
		Name:       "fixture",
		Version:    "1",
		PolicyHash: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		EstimateTokens: func(text string) int {
			return len([]rune(text))
		},
	}
	ag, err := New(Config{LLM: &accountingContractModel{}, Tools: []tools.Tool{echo}, AccountingEstimator: estimator})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
	var accountingEvents []AccountingEvent
	toolResultIndex := -1
	toolAccountingIndex := -1
	usageCount := 0
	usageAccountingCount := 0
	for i, event := range events {
		switch e := event.(type) {
		case UsageEvent:
			usageCount++
		case ToolResultEvent:
			toolResultIndex = i
		case AccountingEvent:
			accountingEvents = append(accountingEvents, e)
			switch e.Payload.EventKind {
			case sdkaccounting.EventKindProviderUsage:
				usageAccountingCount++
			case sdkaccounting.EventKindToolResult:
				toolAccountingIndex = i
				if e.ToolCallID != "call-accounting-1" {
					t.Fatalf("tool correlation = %#v", e)
				}
			}
		}
	}
	if usageCount != 2 || usageAccountingCount != 2 {
		t.Fatalf("usage/accounting counts = %d/%d events=%#v", usageCount, usageAccountingCount, events)
	}
	if toolResultIndex < 0 || toolAccountingIndex != toolResultIndex+1 {
		t.Fatalf("tool accounting order result=%d accounting=%d", toolResultIndex, toolAccountingIndex)
	}
	if len(accountingEvents) != 3 {
		t.Fatalf("accounting event count=%d, want 3", len(accountingEvents))
	}
	for i, event := range accountingEvents {
		if event.Sequence != uint64(i+1) {
			t.Fatalf("sequence[%d]=%d", i, event.Sequence)
		}
		if err := event.Payload.Validate(); err != nil {
			t.Fatalf("payload[%d] invalid: %v", i, err)
		}
	}
}

func TestAgentUsageAccountingPinsConfiguredEstimatorForEstimatedPrompt(t *testing.T) {
	estimator := sdkaccounting.Estimator{
		Name:       "fixture_estimator",
		Version:    "2",
		PolicyHash: "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
		EstimateTokens: func(text string) int {
			return len([]rune(text))
		},
	}
	ag, err := New(Config{LLM: estimatedUsageAccountingModel{}, AccountingEstimator: estimator})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
	for _, event := range events {
		accountingEvent, ok := event.(AccountingEvent)
		if !ok || accountingEvent.Payload.EventKind != sdkaccounting.EventKindProviderUsage {
			continue
		}
		if accountingEvent.Payload.Estimator == nil || accountingEvent.Payload.Estimator.PolicyHash != estimator.PolicyHash {
			t.Fatalf("estimated usage accounting estimator = %#v", accountingEvent.Payload.Estimator)
		}
		return
	}
	t.Fatalf("estimated provider usage accounting event missing: %#v", events)
}
