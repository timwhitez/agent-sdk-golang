package openai

import (
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestReasoningUsageIsOutputSubset(t *testing.T) {
	for _, responses := range []bool{false, true} {
		data := map[string]any{"prompt_tokens": float64(100), "completion_tokens": float64(50), "total_tokens": float64(150), "completion_tokens_details": map[string]any{"reasoning_tokens": float64(40)}, "output_tokens_details": map[string]any{"reasoning_tokens": float64(40)}}
		usage := parseUsage(data)
		if responses {
			usage = normalizedResponsesUsage(data)
		}
		if usage.CompletionTokens != 50 || usage.TotalTokens != 150 || usage.CompletionReasoningTokens == nil || *usage.CompletionReasoningTokens != 40 {
			t.Fatalf("reasoning is a subset: %+v", usage)
		}
		cloned := llm.CloneUsage(usage)
		*usage.CompletionReasoningTokens = 41
		if *cloned.CompletionReasoningTokens != 40 {
			t.Fatal("reasoning clone aliases input")
		}
		data["output_tokens_details"] = map[string]any{"reasoning_tokens": float64(-1)}
		if normalizedResponsesUsage(data).CompletionReasoningTokens != nil {
			t.Fatal("negative breakdown known")
		}
	}
}
