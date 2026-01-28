package tokens

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const pricingURL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

type ModelPricing struct {
	Model string

	InputCostPerToken  *float64
	OutputCostPerToken *float64

	MaxTokens       *int
	MaxInputTokens  *int
	MaxOutputTokens *int

	CacheReadInputTokenCost     *float64
	CacheCreationInputTokenCost *float64
}

type TokenUsageEntry struct {
	Model string
	At    time.Time
	Usage llm.Usage
	Cost  *TokenCostCalculated
}

type TokenCostCalculated struct {
	NewPromptTokens int
	NewPromptCost   float64

	PromptReadCachedTokens *int
	PromptReadCachedCost   *float64

	PromptCacheCreationTokens *int
	PromptCacheCreationCost   *float64

	CompletionTokens int
	CompletionCost   float64
}

type UsageSummary struct {
	TotalTokens int
	TotalCost   float64
}

type cachedPricingData struct {
	Timestamp time.Time       `json:"timestamp"`
	Data      map[string]any  `json:"data"`
}

type TokenCost struct {
	IncludeCost bool

	mu          sync.Mutex
	pricingData map[string]any
	initialized bool

	usageHistory []TokenUsageEntry

	cacheDir string

	HTTPClient *http.Client
}

func New(includeCost bool) *TokenCost {
	return &TokenCost{
		IncludeCost: includeCost || strings.EqualFold(os.Getenv("BU_AGENT_SDK_CALCULATE_COST"), "true"),
		cacheDir:    filepath.Join(xdgCacheHome(), "bu_agent_sdk", "token_cost"),
		HTTPClient:  &http.Client{Timeout: 30 * time.Second},
	}
}

func xdgCacheHome() string {
	if v := os.Getenv("XDG_CACHE_HOME"); v != "" && filepath.IsAbs(v) {
		return v
	}
	h, _ := os.UserHomeDir()
	if h == "" {
		return "/tmp"
	}
	return filepath.Join(h, ".cache")
}

func (tc *TokenCost) Initialize(ctx context.Context) error {
	tc.mu.Lock()
	if tc.initialized {
		tc.mu.Unlock()
		return nil
	}
	tc.mu.Unlock()

	if !tc.IncludeCost {
		tc.mu.Lock()
		tc.initialized = true
		tc.mu.Unlock()
		return nil
	}

	data, err := tc.loadPricingData(ctx)
	if err != nil {
		// fall back to empty
		data = map[string]any{}
	}

	tc.mu.Lock()
	tc.pricingData = data
	tc.initialized = true
	tc.mu.Unlock()
	return nil
}

func (tc *TokenCost) AddUsage(ctx context.Context, model string, usage llm.Usage) (TokenUsageEntry, error) {
	if err := tc.Initialize(ctx); err != nil {
		return TokenUsageEntry{}, err
	}
	entry := TokenUsageEntry{Model: model, At: time.Now(), Usage: usage}
	if tc.IncludeCost {
		calc, _ := tc.calculateCost(ctx, model, usage)
		entry.Cost = calc
	}
	tc.mu.Lock()
	tc.usageHistory = append(tc.usageHistory, entry)
	tc.mu.Unlock()
	return entry, nil
}

func (tc *TokenCost) GetUsageSummary() UsageSummary {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	s := UsageSummary{}
	for _, e := range tc.usageHistory {
		s.TotalTokens += e.Usage.TotalTokens
		if e.Cost != nil {
			s.TotalCost += e.Cost.NewPromptCost + e.Cost.CompletionCost
			if e.Cost.PromptReadCachedCost != nil {
				s.TotalCost += *e.Cost.PromptReadCachedCost
			}
			if e.Cost.PromptCacheCreationCost != nil {
				s.TotalCost += *e.Cost.PromptCacheCreationCost
			}
		}
	}
	return s
}

func (tc *TokenCost) loadPricingData(ctx context.Context) (map[string]any, error) {
	// cache is valid for 24h
	if err := os.MkdirAll(tc.cacheDir, 0o755); err != nil {
		return nil, err
	}
	cacheFile, ok := tc.findValidCache(24 * time.Hour)
	if ok {
		b, err := os.ReadFile(cacheFile)
		if err == nil {
			var cached cachedPricingData
			if json.Unmarshal(b, &cached) == nil {
				if cached.Data != nil {
					return cached.Data, nil
				}
			}
		}
	}

	// fetch
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, pricingURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := tc.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("pricing fetch failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var data map[string]any
	if err := json.Unmarshal(b, &data); err != nil {
		return nil, err
	}

	// write cache
	cache := cachedPricingData{Timestamp: time.Now(), Data: data}
	out, _ := json.MarshalIndent(cache, "", "  ")
	_ = os.WriteFile(filepath.Join(tc.cacheDir, fmt.Sprintf("pricing_%s.json", time.Now().Format("20060102_150405"))), out, 0o644)
	return data, nil
}

func (tc *TokenCost) findValidCache(maxAge time.Duration) (string, bool) {
	entries, err := os.ReadDir(tc.cacheDir)
	if err != nil {
		return "", false
	}
	var best string
	var bestMod time.Time
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".json") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		mod := info.ModTime()
		if time.Since(mod) > maxAge {
			continue
		}
		if mod.After(bestMod) {
			bestMod = mod
			best = filepath.Join(tc.cacheDir, name)
		}
	}
	if best == "" {
		return "", false
	}
	return best, true
}

func (tc *TokenCost) calculateCost(ctx context.Context, model string, usage llm.Usage) (*TokenCostCalculated, error) {
	p, err := tc.GetModelPricing(ctx, model)
	if err != nil || p == nil {
		return nil, err
	}
	if p.InputCostPerToken == nil || p.OutputCostPerToken == nil {
		return nil, nil
	}
	uncachedPrompt := usage.PromptTokens
	if usage.PromptCachedTokens != nil {
		uncachedPrompt -= *usage.PromptCachedTokens
	}
	calc := &TokenCostCalculated{
		NewPromptTokens: usage.PromptTokens,
		NewPromptCost:   float64(uncachedPrompt) * *p.InputCostPerToken,
		CompletionTokens: usage.CompletionTokens,
		CompletionCost:   float64(usage.CompletionTokens) * *p.OutputCostPerToken,
		PromptReadCachedTokens: usage.PromptCachedTokens,
		PromptCacheCreationTokens: usage.PromptCacheCreationTokens,
	}
	if usage.PromptCachedTokens != nil && p.CacheReadInputTokenCost != nil {
		v := float64(*usage.PromptCachedTokens) * *p.CacheReadInputTokenCost
		calc.PromptReadCachedCost = &v
	}
	if usage.PromptCacheCreationTokens != nil && p.CacheCreationInputTokenCost != nil {
		v := float64(*usage.PromptCacheCreationTokens) * *p.CacheCreationInputTokenCost
		calc.PromptCacheCreationCost = &v
	}
	return calc, nil
}

func (tc *TokenCost) GetModelPricing(ctx context.Context, modelName string) (*ModelPricing, error) {
	if err := tc.Initialize(ctx); err != nil {
		return nil, err
	}
	if !tc.IncludeCost {
		return nil, nil
	}
	tc.mu.Lock()
	data := tc.pricingData
	tc.mu.Unlock()
	if data == nil {
		return nil, errors.New("pricing data not loaded")
	}

	// Exact match
	if m, ok := data[modelName]; ok {
		return parseModelPricing(modelName, m)
	}

	// Mapped name
	if mapped, ok := modelToLiteLLM[modelName]; ok {
		if m, ok := data[mapped]; ok {
			return parseModelPricing(modelName, m)
		}
	}

	// Try common prefixes
	for _, prefix := range []string{"anthropic/", "openai/", "google/", "azure/", "bedrock/"} {
		if m, ok := data[prefix+modelName]; ok {
			return parseModelPricing(modelName, m)
		}
	}

	// Strip existing prefix
	if i := strings.Index(modelName, "/"); i >= 0 {
		bare := modelName[i+1:]
		if m, ok := data[bare]; ok {
			return parseModelPricing(modelName, m)
		}
	}

	return nil, nil
}

var modelToLiteLLM = map[string]string{
	"gemini-flash-latest": "gemini/gemini-flash-latest",
}

func parseModelPricing(modelName string, raw any) (*ModelPricing, error) {
	m, ok := raw.(map[string]any)
	if !ok {
		return nil, nil
	}
	getF := func(k string) *float64 {
		if v, ok := m[k].(float64); ok {
			vv := v
			return &vv
		}
		return nil
	}
	getI := func(k string) *int {
		if v, ok := m[k].(float64); ok {
			vv := int(v)
			return &vv
		}
		return nil
	}
	return &ModelPricing{
		Model:                    modelName,
		InputCostPerToken:        getF("input_cost_per_token"),
		OutputCostPerToken:       getF("output_cost_per_token"),
		MaxTokens:                getI("max_tokens"),
		MaxInputTokens:           getI("max_input_tokens"),
		MaxOutputTokens:          getI("max_output_tokens"),
		CacheReadInputTokenCost:  getF("cache_read_input_token_cost"),
		CacheCreationInputTokenCost: getF("cache_creation_input_token_cost"),
	}, nil
}
