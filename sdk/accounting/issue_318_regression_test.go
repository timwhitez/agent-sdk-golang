package accounting

import (
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestD3NormalizedMissingUsageKeepsUnknownCounts(t *testing.T) {
	u := llm.NormalizeUsage(&llm.Usage{})
	if u.PromptTokensSource != llm.PromptTokensSourceMissing {
		t.Fatal("missing fixture")
	}
	p := ProjectUsage(*u)
	if p.Usage.EffectivePromptTokens != nil {
		t.Fatalf("all missing normalized usage became effective_prompt_tokens=%v status=%q source=%q", *p.Usage.EffectivePromptTokens, p.Status, p.Usage.PromptTokensSource)
	}
	if p.Status != StatusUnknown {
		t.Fatalf("missing usage status=%q", p.Status)
	}
}

func TestMissingPromptPreservesIndependentUsage(t *testing.T) {
	u := llm.NormalizeUsage(&llm.Usage{CompletionTokens: 7})
	p := ProjectUsage(*u)
	if p.Usage.EffectivePromptTokens != nil || p.Usage.CompletionTokens == nil || *p.Usage.CompletionTokens != 7 || p.Status != StatusSuccess {
		t.Fatalf("completion-only usage = %+v", p)
	}
	zero := ProjectUsage(llm.Usage{PromptTokensValid: true, PromptTokensSource: llm.PromptTokensSourceProvider})
	if zero.Usage.EffectivePromptTokens == nil || *zero.Usage.EffectivePromptTokens != 0 {
		t.Fatal("explicit valid zero prompt was lost")
	}
}
