package llm

import "strings"

const (
	PromptTokensSemanticsTotalInputV1 = "total_input_v1"

	PromptTokensSourceProvider        = "provider"
	PromptTokensSourceEstimate        = "estimate"
	PromptTokensSourceMissing         = "missing"
	PromptTokensSourceLegacyOrUnknown = "legacy_or_unknown"
)

// NewProviderUsage builds the normalized provider usage contract. promptTotal
// must be the complete input size; cache/image values are breakdown only.
func NewProviderUsage(promptTotal, completionTotal, providerTotal int) *Usage {
	valid := promptTotal > 0
	source := PromptTokensSourceProvider
	if !valid {
		source = PromptTokensSourceMissing
	}
	if providerTotal <= 0 && (promptTotal > 0 || completionTotal > 0) {
		providerTotal = promptTotal + completionTotal
	}
	return &Usage{
		PromptTokens:          promptTotal,
		CompletionTokens:      completionTotal,
		TotalTokens:           providerTotal,
		PromptTokensValid:     valid,
		PromptTokensSource:    source,
		PromptTokensSemantics: PromptTokensSemanticsTotalInputV1,
	}
}

// PromptUsageIsProviderValid accepts explicit v1 provider usage and positive
// legacy usage. Positive legacy values remain usable for backwards
// compatibility, but their source is reported as legacy_or_unknown.
func PromptUsageIsProviderValid(u *Usage) bool {
	if u == nil {
		return false
	}
	if u.PromptTokensSemantics == PromptTokensSemanticsTotalInputV1 {
		return u.PromptTokensValid && u.PromptTokens >= 0
	}
	return u.PromptTokens > 0
}

// NormalizeUsage annotates usage values from legacy/custom providers without
// pretending that their prompt-token semantics are known. Provider adapters
// that implement total_input_v1 retain their explicit quality fields.
func NormalizeUsage(u *Usage) *Usage {
	if u == nil {
		return nil
	}
	out := CloneUsage(u)
	if out.TotalTokens <= 0 && (out.PromptTokens > 0 || out.CompletionTokens > 0) {
		out.TotalTokens = maxInt(out.PromptTokens, 0) + maxInt(out.CompletionTokens, 0)
	}
	if out.PromptTokensSemantics == PromptTokensSemanticsTotalInputV1 {
		if strings.TrimSpace(out.PromptTokensSource) == "" {
			if out.PromptTokensValid {
				out.PromptTokensSource = PromptTokensSourceProvider
			} else {
				out.PromptTokensSource = PromptTokensSourceMissing
			}
		}
		return out
	}
	if strings.TrimSpace(out.PromptTokensSource) == "" {
		if out.PromptTokens > 0 {
			out.PromptTokensSource = PromptTokensSourceLegacyOrUnknown
		} else {
			out.PromptTokensSource = PromptTokensSourceMissing
		}
	}
	return out
}

// EffectivePromptTokens returns the normalized/effective prompt total and its
// source. It never adds cache or image breakdowns to PromptTokens.
func EffectivePromptTokens(u *Usage) (int, string) {
	if u == nil {
		return 0, PromptTokensSourceMissing
	}
	source := strings.TrimSpace(u.PromptTokensSource)
	if source == "" {
		if u.PromptTokensSemantics == "" && u.PromptTokens > 0 {
			source = PromptTokensSourceLegacyOrUnknown
		} else {
			source = PromptTokensSourceMissing
		}
	}
	return maxInt(u.PromptTokens, 0), source
}

// WithPromptEstimate returns a copy that replaces invalid/missing provider
// prompt usage with an estimate. Raw provider totals remain available in the
// Provider* fields for diagnostics and persistence.
func WithPromptEstimate(u *Usage, estimatedPrompt int) *Usage {
	if u == nil {
		if estimatedPrompt <= 0 {
			return nil
		}
		return &Usage{
			PromptTokens:          estimatedPrompt,
			TotalTokens:           estimatedPrompt,
			PromptTokensSource:    PromptTokensSourceEstimate,
			PromptTokensSemantics: PromptTokensSemanticsTotalInputV1,
		}
	}
	out := NormalizeUsage(u)
	if PromptUsageIsProviderValid(out) || estimatedPrompt <= 0 {
		if strings.TrimSpace(out.PromptTokensSource) == "" {
			out.PromptTokensSource = PromptTokensSourceLegacyOrUnknown
		}
		return out
	}
	rawPrompt := out.PromptTokens
	rawTotal := out.TotalTokens
	out.ProviderPromptTokens = &rawPrompt
	out.ProviderTotalTokens = &rawTotal
	out.PromptTokens = estimatedPrompt
	out.TotalTokens = estimatedPrompt + maxInt(out.CompletionTokens, 0)
	out.PromptTokensValid = false
	out.PromptTokensSource = PromptTokensSourceEstimate
	out.PromptTokensSemantics = PromptTokensSemanticsTotalInputV1
	return out
}

func CloneUsage(u *Usage) *Usage {
	if u == nil {
		return nil
	}
	out := *u
	out.PromptCachedTokens = cloneIntPtr(u.PromptCachedTokens)
	out.PromptCacheCreationTokens = cloneIntPtr(u.PromptCacheCreationTokens)
	out.PromptImageTokens = cloneIntPtr(u.PromptImageTokens)
	out.PromptUncachedTokens = cloneIntPtr(u.PromptUncachedTokens)
	out.ProviderPromptTokens = cloneIntPtr(u.ProviderPromptTokens)
	out.ProviderTotalTokens = cloneIntPtr(u.ProviderTotalTokens)
	return &out
}

// EstimateMessagesTokens is the SDK fallback used when a compatible gateway
// omits prompt usage. Hosts may use a more exact tokenizer for preflight, but
// all runtime surfaces receive this same effective value from the agent event.
func EstimateMessagesTokens(messages []Message) int {
	total := 0
	for _, msg := range messages {
		total += estimateTextTokens(string(msg.Role))
		total += estimateTextTokens(msg.Name)
		total += estimateTextTokens(msg.ToolCallID)
		total += estimateTextTokens(msg.ToolName)
		total += estimateTextTokens(msg.Content.PlainText())
		for _, call := range msg.ToolCalls {
			total += estimateTextTokens(call.ID)
			total += estimateTextTokens(call.Type)
			total += estimateTextTokens(call.Function.Name)
			total += estimateTextTokens(call.Function.Arguments)
		}
		total += 4
	}
	return total
}

func estimateTextTokens(text string) int {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	return (len(text) + 3) / 4
}

func cloneIntPtr(v *int) *int {
	if v == nil {
		return nil
	}
	out := *v
	return &out
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
