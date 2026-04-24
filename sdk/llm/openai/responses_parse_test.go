package openai

import "testing"

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
