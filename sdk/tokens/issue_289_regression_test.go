package tokens

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestAuditCreationPrice(t *testing.T) {
	tc := &TokenCost{IncludeCost: true, initialized: true, pricingData: map[string]any{
		"fixture": map[string]any{"input_cost_per_token": 1.0, "output_cost_per_token": 2.0, "cache_read_input_token_cost": 0.5, "cache_creation_input_token_cost": 1.25},
	}}
	raw, read, write := 10, 20, 40
	u := llm.NewProviderUsage(raw+read+write, 5, 75)
	u.PromptUncachedTokens, u.PromptCachedTokens, u.PromptCacheCreationTokens = &raw, &read, &write
	e, err := tc.AddUsage(context.Background(), "fixture", *u)
	if err != nil || e.Cost == nil {
		t.Fatalf("missing cost: %v", err)
	}
	if got := tc.GetUsageSummary().TotalCost; got != 80 || e.Cost.NewPromptTokens != 10 {
		t.Fatalf("new prompt tokens=%d, total cost=%g; want 10 tokens and 80 (10 + 10 + 50 + 10)", e.Cost.NewPromptTokens, got)
	}
}

func TestCacheCreationWithoutSeparatePriceUsesInputPrice(t *testing.T) {
	tc := &TokenCost{IncludeCost: true, initialized: true, pricingData: map[string]any{
		"fixture": map[string]any{"input_cost_per_token": 1.0, "output_cost_per_token": 2.0, "cache_read_input_token_cost": 0.5},
	}}
	read, write := 20, 40
	u := llm.NewProviderUsage(70, 5, 75)
	u.PromptCachedTokens, u.PromptCacheCreationTokens = &read, &write
	entry, err := tc.AddUsage(context.Background(), "fixture", *u)
	if err != nil || entry.Cost == nil || entry.Cost.NewPromptTokens != 50 || tc.GetUsageSummary().TotalCost != 70 {
		t.Fatalf("fallback input pricing changed: %+v %v", entry, err)
	}
}
