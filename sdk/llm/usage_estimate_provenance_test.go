package llm

import "testing"

func TestRepeatedPromptEstimateKeepsProviderEvidence(t *testing.T) {
	for _, tc := range []struct {
		name    string
		usage   *Usage
		wantRaw bool
	}{
		{"provider_zero", NewProviderUsage(0, 2, 2), true},
		{"no_provider_observation", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			first := WithPromptEstimate(tc.usage, 40)
			second := WithPromptEstimate(first, 50)
			if first.PromptTokens != 40 || second.PromptTokens != 50 {
				t.Fatalf("estimates mutated or not updated: first=%+v second=%+v", first, second)
			}
			if !tc.wantRaw {
				if second.ProviderPromptTokens != nil || second.ProviderTotalTokens != nil {
					t.Fatalf("manufactured provider observation: %+v", second)
				}
				return
			}
			if second.ProviderPromptTokens == nil || *second.ProviderPromptTokens != 0 || second.ProviderTotalTokens == nil || *second.ProviderTotalTokens != 2 {
				t.Fatalf("raw provider counts changed: %+v", second)
			}
			*second.ProviderPromptTokens = 99
			if *first.ProviderPromptTokens != 0 {
				t.Fatal("provider evidence was not cloned")
			}
		})
	}
}
