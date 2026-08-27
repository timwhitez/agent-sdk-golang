package openai

import (
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestParseResponsesInfersTotalTokensWhenGatewayOmitsIt(t *testing.T) {
	comp, err := parseResponses([]byte(`{"id":"resp_123","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":11,"output_tokens":7}}`))
	if err != nil {
		t.Fatalf("parseResponses: %v", err)
	}
	if comp.Usage == nil {
		t.Fatal("expected usage")
	}
	if comp.Usage.TotalTokens != 18 {
		t.Fatalf("total tokens = %d, want 18", comp.Usage.TotalTokens)
	}
}

func TestParseResponsesAcceptsPromptCompletionUsageKeys(t *testing.T) {
	comp, err := parseResponses([]byte(`{"id":"resp_456","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))
	if err != nil {
		t.Fatalf("parseResponses: %v", err)
	}
	if comp.Usage == nil {
		t.Fatal("expected usage")
	}
	if comp.Usage.PromptTokens != 5 || comp.Usage.CompletionTokens != 3 || comp.Usage.TotalTokens != 8 {
		t.Fatalf("unexpected usage: %+v", comp.Usage)
	}
}

func TestUsageFromResponsesAcceptsPromptCompletionAliases(t *testing.T) {
	usage := usageFromResponses(map[string]any{
		"usage": map[string]any{
			"prompt_tokens":     9,
			"completion_tokens": 4,
			"total_tokens":      13,
		},
	})
	if usage == nil {
		t.Fatal("expected usage")
	}
	if usage.PromptTokens != 9 || usage.CompletionTokens != 4 || usage.TotalTokens != 13 {
		t.Fatalf("unexpected usage: %+v", usage)
	}
}

func TestUsageFromResponsesKeepsCachedAndImageTokensAsBreakdown(t *testing.T) {
	usage := usageFromResponses(map[string]any{
		"usage": map[string]any{
			"input_tokens":  120,
			"output_tokens": 30,
			"total_tokens":  150,
			"input_tokens_details": map[string]any{
				"cached_tokens": 40,
				"image_tokens":  6,
			},
		},
	})
	if usage == nil {
		t.Fatal("expected usage")
	}
	if usage.PromptTokens != 120 || usage.TotalTokens != 150 {
		t.Fatalf("breakdown was double-counted: %#v", usage)
	}
	if usage.PromptCachedTokens == nil || *usage.PromptCachedTokens != 40 || usage.PromptImageTokens == nil || *usage.PromptImageTokens != 6 {
		t.Fatalf("missing prompt breakdown: %#v", usage)
	}
	if !usage.PromptTokensValid || usage.PromptTokensSource != llm.PromptTokensSourceProvider || usage.PromptTokensSemantics != llm.PromptTokensSemanticsTotalInputV1 {
		t.Fatalf("unexpected normalized usage quality: %#v", usage)
	}
}

func TestParseResponsesUsesFunctionCallIDForToolOutputCorrelation(t *testing.T) {
	comp, err := parseResponses([]byte(`{"id":"resp_tool","status":"completed","output":[{"id":"fc_123","call_id":"call_123","type":"function_call","name":"lookup","arguments":"{\"query\":\"go\"}"}]}`))
	if err != nil {
		t.Fatalf("parseResponses: %v", err)
	}
	if len(comp.ToolCalls) != 1 {
		t.Fatalf("tool calls = %#v, want one", comp.ToolCalls)
	}
	if got := comp.ToolCalls[0].ID; got != "call_123" {
		t.Fatalf("tool call ID = %q, want provider call_id %q", got, "call_123")
	}
}
