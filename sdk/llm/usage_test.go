package llm

import "testing"

func TestWithPromptEstimatePreservesRawProviderZero(t *testing.T) {
	usage := NewProviderUsage(0, 9, 9)
	effective := WithPromptEstimate(usage, 123)
	if effective.PromptTokens != 123 || effective.TotalTokens != 132 {
		t.Fatalf("effective usage = %#v", effective)
	}
	if effective.PromptTokensSource != PromptTokensSourceEstimate || effective.PromptTokensValid {
		t.Fatalf("effective quality = source=%q valid=%t", effective.PromptTokensSource, effective.PromptTokensValid)
	}
	if effective.ProviderPromptTokens == nil || *effective.ProviderPromptTokens != 0 {
		t.Fatalf("provider prompt tokens = %#v", effective.ProviderPromptTokens)
	}
	if effective.ProviderTotalTokens == nil || *effective.ProviderTotalTokens != 9 {
		t.Fatalf("provider total tokens = %#v", effective.ProviderTotalTokens)
	}
}

func TestWithPromptEstimateKeepsValidProviderUsage(t *testing.T) {
	usage := NewProviderUsage(80, 10, 90)
	effective := WithPromptEstimate(usage, 999)
	if effective.PromptTokens != 80 || effective.TotalTokens != 90 {
		t.Fatalf("valid provider usage was replaced: %#v", effective)
	}
	if effective.PromptTokensSource != PromptTokensSourceProvider || !effective.PromptTokensValid {
		t.Fatalf("provider quality = source=%q valid=%t", effective.PromptTokensSource, effective.PromptTokensValid)
	}
}

func TestEffectivePromptTokensTreatsPositiveLegacyUsageAsUsable(t *testing.T) {
	usage := NormalizeUsage(&Usage{PromptTokens: 42, CompletionTokens: 8})
	prompt, source := EffectivePromptTokens(usage)
	if prompt != 42 || source != PromptTokensSourceLegacyOrUnknown {
		t.Fatalf("legacy effective usage = %d, %q", prompt, source)
	}
	if usage.TotalTokens != 50 || usage.PromptTokensValid || usage.PromptTokensSemantics != "" {
		t.Fatalf("legacy usage was mis-normalized: %#v", usage)
	}
}
