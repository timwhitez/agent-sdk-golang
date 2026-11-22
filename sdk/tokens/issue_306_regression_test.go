package tokens

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"testing"
)

func TestAuditReturnedCostOwnership(t *testing.T) {
	tc := &TokenCost{IncludeCost: true, initialized: true, pricingData: map[string]any{
		"fixture": map[string]any{"input_cost_per_token": 1.0, "output_cost_per_token": 2.0},
	}}
	e, err := tc.AddUsage(context.Background(), "fixture", *llm.NewProviderUsage(3, 1, 4))
	if err != nil || e.Cost == nil {
		t.Fatalf("missing cost: %v", err)
	}
	before := tc.GetUsageSummary()
	if before.TotalCost != 5 {
		t.Fatalf("invalid fixture: %+v", before)
	}
	e.Cost.NewPromptCost = 0
	if after := tc.GetUsageSummary(); after != before {
		t.Fatalf("summary changed from %+v to %+v without AddUsage", before, after)
	}
}

func TestReturnedUsageAndNestedCostsOwnTheirValues(t *testing.T) {
	tc := &TokenCost{IncludeCost: true, initialized: true, pricingData: map[string]any{
		"fixture": map[string]any{"input_cost_per_token": 1.0, "output_cost_per_token": 2.0, "cache_read_input_token_cost": 0.5, "cache_creation_input_token_cost": 1.25},
	}}
	read, write := 2, 3
	usage := *llm.NewProviderUsage(10, 1, 11)
	usage.PromptCachedTokens, usage.PromptCacheCreationTokens = &read, &write
	entry, err := tc.AddUsage(context.Background(), "fixture", usage)
	if err != nil || entry.Cost == nil {
		t.Fatalf("AddUsage: %+v %v", entry, err)
	}
	before := tc.GetUsageSummary()
	*entry.Cost.PromptReadCachedCost = 99
	*entry.Cost.PromptCacheCreationCost = 99
	*entry.Cost.PromptReadCachedTokens = 99
	*entry.Cost.PromptCacheCreationTokens = 99
	*entry.Usage.PromptCachedTokens = 99
	*entry.Usage.PromptCacheCreationTokens = 99
	read, write = 999, 999
	if after := tc.GetUsageSummary(); after != before {
		t.Fatalf("nested pointer mutation changed accounting: before=%+v after=%+v", before, after)
	}
	if got := tc.usageHistory[0]; *got.Usage.PromptCachedTokens != 2 || *got.Usage.PromptCacheCreationTokens != 3 || *got.Cost.PromptReadCachedTokens != 2 || *got.Cost.PromptCacheCreationTokens != 3 {
		t.Fatalf("stored usage or cost shared external pointers: %+v", got)
	}
}
