package tokens

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	pricingURL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

	defaultMaxPricingBodyBytes       int64 = 8 * 1024 * 1024
	defaultPricingCacheScanCap             = 4096
	defaultPricingCacheScanReadBatch       = 128

	pricingCacheDirMode  os.FileMode = 0o700
	pricingCacheFileMode os.FileMode = 0o600
)

var pricingCacheWarningf = log.Printf
var maxPricingBodyBytes int64 = defaultMaxPricingBodyBytes
var writePricingCacheFile = atomicWritePricingCacheFile
var readPricingCacheFile = readPricingCacheFileLimited
var readPricingCacheDir = os.ReadDir
var pricingCacheScanCap = defaultPricingCacheScanCap
var pricingCacheScanReadBatch = defaultPricingCacheScanReadBatch
var openPricingCacheDir = func(path string) (pricingCacheDirReader, error) {
	return os.Open(path)
}
var pricingCacheEntryInfo = func(entry os.DirEntry) (os.FileInfo, error) {
	return entry.Info()
}

type pricingCacheDirReader interface {
	ReadDir(n int) ([]os.DirEntry, error)
	Close() error
}

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
	Timestamp time.Time      `json:"timestamp"`
	Data      map[string]any `json:"data"`
}

type TokenCost struct {
	IncludeCost bool

	mu          sync.Mutex
	initOnce    sync.Once
	initErr     error
	pricingData map[string]any
	initialized bool

	usageHistory []TokenUsageEntry

	cacheDir string

	HTTPClient *http.Client

	nowFn            func() time.Time
	writeCacheFileFn func(path string, data []byte, mode os.FileMode) error
}

func New(includeCost bool) *TokenCost {
	return &TokenCost{
		IncludeCost: includeCost || strings.EqualFold(os.Getenv("BU_AGENT_SDK_CALCULATE_COST"), "true"),
		cacheDir:    filepath.Join(xdgCacheHome(), "bu_agent_sdk", "token_cost"),
		HTTPClient:  &http.Client{Timeout: 30 * time.Second},
		nowFn:       time.Now,
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

func readPricingCacheFileLimited(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readPricingPayloadLimited(f, fmt.Sprintf("pricing cache file %q", path))
}

func readPricingPayloadLimited(r io.Reader, source string) ([]byte, error) {
	limit := maxPricingBodyBytes
	if limit <= 0 {
		limit = defaultMaxPricingBodyBytes
	}
	lr := &io.LimitedReader{R: r, N: limit + 1}
	data, err := io.ReadAll(lr)
	if err != nil {
		return nil, fmt.Errorf("%s read failed: %w", source, err)
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("%s exceeds size limit (%d bytes read; max %d bytes). Remove oversized input and retry", source, len(data), limit)
	}
	return data, nil
}

func ownerOnlyDirMode(mode os.FileMode) os.FileMode {
	owner := mode & 0o700
	if owner == 0 {
		return pricingCacheDirMode
	}
	return owner
}

func ownerOnlyFileMode(mode os.FileMode) os.FileMode {
	owner := mode & 0o600
	if owner == 0 {
		return pricingCacheFileMode
	}
	return owner
}

func ensurePricingCacheDir(path string) error {
	if err := os.MkdirAll(path, pricingCacheDirMode); err != nil {
		return err
	}
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	current := info.Mode().Perm()
	ownerOnly := ownerOnlyDirMode(current)
	if current == ownerOnly {
		return nil
	}
	return os.Chmod(path, ownerOnly)
}

func resolvePricingCacheFileMode(path string, fallback os.FileMode) (os.FileMode, error) {
	info, err := os.Stat(path)
	if err == nil {
		return ownerOnlyFileMode(info.Mode().Perm()), nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return ownerOnlyFileMode(fallback), nil
	}
	return 0, err
}

func writePricingCacheData(f *os.File, data []byte) error {
	for len(data) > 0 {
		n, err := f.Write(data)
		if n > 0 {
			data = data[n:]
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

func atomicWritePricingCacheFile(path string, data []byte, mode os.FileMode) error {
	return atomicWritePricingCacheFileWithWriter(path, data, mode, writePricingCacheData)
}

func atomicWritePricingCacheFileWithWriter(path string, data []byte, mode os.FileMode, writeData func(*os.File, []byte) error) error {
	if writeData == nil {
		return errors.New("pricing cache writer is nil")
	}
	dir := filepath.Dir(path)
	if err := ensurePricingCacheDir(dir); err != nil {
		return err
	}
	resolvedMode, err := resolvePricingCacheFileMode(path, mode)
	if err != nil {
		return err
	}

	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}()

	if err := tmp.Chmod(resolvedMode); err != nil {
		return err
	}
	if err := writeData(tmp, data); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		if removeErr := os.Remove(path); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			return fmt.Errorf("replace pricing cache file: %w (remove existing: %v)", err, removeErr)
		}
		if retryErr := os.Rename(tmpPath, path); retryErr != nil {
			return retryErr
		}
	}
	return os.Chmod(path, resolvedMode)
}

func (tc *TokenCost) now() time.Time {
	if tc != nil && tc.nowFn != nil {
		return tc.nowFn()
	}
	return time.Now()
}

func (tc *TokenCost) writeCacheFile(path string, data []byte, mode os.FileMode) error {
	if tc != nil && tc.writeCacheFileFn != nil {
		return tc.writeCacheFileFn(path, data, mode)
	}
	return writePricingCacheFile(path, data, mode)
}

func (tc *TokenCost) Initialize(ctx context.Context) error {
	tc.mu.Lock()
	if tc.initialized {
		err := tc.initErr
		tc.mu.Unlock()
		return err
	}
	tc.mu.Unlock()

	tc.initOnce.Do(func() {
		if !tc.IncludeCost {
			tc.mu.Lock()
			tc.initErr = nil
			tc.initialized = true
			tc.mu.Unlock()
			return
		}

		data, err := tc.loadPricingData(ctx)
		if err != nil {
			pricingCacheWarningf("tokens: pricing initialization failed; continuing without cost data (cache_dir=%q): %v", tc.cacheDir, err)
			// Non-fatal fallback: usage history remains available without cost math.
			data = map[string]any{}
		}

		tc.mu.Lock()
		tc.initErr = nil
		tc.pricingData = data
		tc.initialized = true
		tc.mu.Unlock()
	})

	tc.mu.Lock()
	err := tc.initErr
	tc.mu.Unlock()
	return err
}

func (tc *TokenCost) AddUsage(ctx context.Context, model string, usage llm.Usage) (TokenUsageEntry, error) {
	if err := tc.Initialize(ctx); err != nil {
		return TokenUsageEntry{}, err
	}
	entry := TokenUsageEntry{Model: model, At: time.Now(), Usage: usage}
	if tc.IncludeCost {
		calc, err := tc.calculateCost(ctx, model, usage)
		if err != nil {
			pricingCacheWarningf(
				"tokens: cost calculation failed (model=%q prompt=%d completion=%d total=%d): %v",
				model,
				usage.PromptTokens,
				usage.CompletionTokens,
				usage.TotalTokens,
				err,
			)
		} else {
			entry.Cost = calc
		}
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
	if err := ensurePricingCacheDir(tc.cacheDir); err != nil {
		return nil, err
	}
	cacheFile, ok := tc.findValidCache(24 * time.Hour)
	if ok {
		b, err := readPricingCacheFile(cacheFile)
		if err != nil {
			pricingCacheWarningf("tokens: unable to read pricing cache %q: %v", cacheFile, err)
		} else {
			var cached cachedPricingData
			if err := json.Unmarshal(b, &cached); err != nil {
				pricingCacheWarningf("tokens: unable to parse pricing cache %q: %v", cacheFile, err)
			} else if cached.Data != nil {
				return cached.Data, nil
			} else {
				pricingCacheWarningf("tokens: pricing cache %q missing data payload", cacheFile)
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
		b, readErr := readPricingPayloadLimited(resp.Body, "pricing response body")
		if readErr != nil {
			return nil, fmt.Errorf("pricing fetch failed (%d): %w", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("pricing fetch failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	b, err := readPricingPayloadLimited(resp.Body, "pricing response body")
	if err != nil {
		return nil, err
	}
	var data map[string]any
	if err := json.Unmarshal(b, &data); err != nil {
		return nil, err
	}

	// write cache
	now := tc.now()
	cache := cachedPricingData{Timestamp: now, Data: data}
	out, _ := json.MarshalIndent(cache, "", "  ")
	cachePath := filepath.Join(tc.cacheDir, fmt.Sprintf("pricing_%s.json", now.Format("20060102_150405")))
	if err := tc.writeCacheFile(cachePath, out, pricingCacheFileMode); err != nil {
		pricingCacheWarningf("tokens: unable to write pricing cache %q: %v", cachePath, err)
	}
	return data, nil
}

func (tc *TokenCost) findValidCache(maxAge time.Duration) (string, bool) {
	reader, err := openPricingCacheDir(tc.cacheDir)
	if err != nil {
		pricingCacheWarningf("tokens: unable to list pricing cache directory %q: %v", tc.cacheDir, err)
		return "", false
	}
	defer reader.Close()

	var best string
	var bestMod time.Time
	now := tc.now()
	scanCap := pricingCacheScanCap
	readBatchSize := pricingCacheScanReadBatch
	if readBatchSize <= 0 {
		readBatchSize = defaultPricingCacheScanReadBatch
	}
	if readBatchSize <= 0 {
		readBatchSize = 1
	}
	scannedEntries := 0
	scanTruncated := false

	for {
		if scanCap > 0 && scannedEntries >= scanCap {
			scanTruncated = true
			break
		}
		readBatch := readBatchSize
		if scanCap > 0 {
			remaining := scanCap - scannedEntries
			if remaining < readBatch {
				readBatch = remaining
			}
			if readBatch <= 0 {
				scanTruncated = true
				break
			}
		}
		entries, readErr := reader.ReadDir(readBatch)
		for _, e := range entries {
			scannedEntries++
			if e.IsDir() {
				continue
			}
			name := e.Name()
			if !strings.HasSuffix(name, ".json") {
				continue
			}
			info, err := pricingCacheEntryInfo(e)
			if err != nil {
				pricingCacheWarningf("tokens: unable to inspect pricing cache entry %q: %v", filepath.Join(tc.cacheDir, name), err)
				continue
			}
			mod := info.ModTime()
			if now.Sub(mod) > maxAge {
				continue
			}
			if mod.After(bestMod) {
				bestMod = mod
				best = filepath.Join(tc.cacheDir, name)
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			pricingCacheWarningf("tokens: unable to continue pricing cache scan in %q after %d entries: %v", tc.cacheDir, scannedEntries, readErr)
			break
		}
		if len(entries) == 0 {
			break
		}
	}
	if scanTruncated {
		pricingCacheWarningf("[WARN] Pricing cache scan stopped after %d entries (cap %d) in %q (warning_kind=scan_cap scan_truncated=true scanned_entries=%d scan_cap=%d) - Cost tracking continues with available cache/fetch fallback; remove stale cache files or rerun with a cleaner cache directory.", scannedEntries, scanCap, tc.cacheDir, scannedEntries, scanCap)
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
	if uncachedPrompt < 0 {
		uncachedPrompt = 0
	}
	calc := &TokenCostCalculated{
		NewPromptTokens:           uncachedPrompt,
		NewPromptCost:             float64(uncachedPrompt) * *p.InputCostPerToken,
		CompletionTokens:          usage.CompletionTokens,
		CompletionCost:            float64(usage.CompletionTokens) * *p.OutputCostPerToken,
		PromptReadCachedTokens:    usage.PromptCachedTokens,
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
	if m, ok := findModelPricingEntry(data, modelName); ok {
		return parseModelPricing(modelName, m)
	}

	return nil, nil
}

var modelToLiteLLM = map[string]string{
	"gemini-flash-latest":     "gemini/gemini-flash-latest",
	"gemini-pro-latest":       "gemini/gemini-pro-latest",
	"gemini-1.5-flash-latest": "gemini/gemini-1.5-flash-latest",
	"gemini-1.5-pro-latest":   "gemini/gemini-1.5-pro-latest",
	"gemini-2.0-flash":        "gemini/gemini-2.0-flash",
	"gemini-2.0-flash-exp":    "gemini/gemini-2.0-flash-exp",
}

var modelLookupPrefixes = []string{"anthropic/", "openai/", "google/", "azure/", "bedrock/", "gemini/"}

func findModelPricingEntry(data map[string]any, modelName string) (any, bool) {
	trimmed := strings.TrimSpace(modelName)
	if trimmed == "" {
		return nil, false
	}

	if m, ok := data[trimmed]; ok {
		return m, true
	}

	normalized := strings.ToLower(trimmed)
	if m, ok := data[normalized]; ok {
		return m, true
	}

	if mapped, ok := modelToLiteLLM[normalized]; ok {
		if m, ok := data[mapped]; ok {
			return m, true
		}
	}

	for _, prefix := range modelLookupPrefixes {
		if m, ok := data[prefix+normalized]; ok {
			return m, true
		}
	}

	if i := strings.LastIndex(normalized, "/"); i >= 0 {
		bare := normalized[i+1:]
		if m, ok := data[bare]; ok {
			return m, true
		}
		for _, prefix := range modelLookupPrefixes {
			if m, ok := data[prefix+bare]; ok {
				return m, true
			}
		}
	}

	base := modelLookupBase(normalized)
	if base == "" {
		return nil, false
	}

	bestKey := ""
	bestPriority := -1
	for key := range data {
		keyBase := modelLookupBase(key)
		if keyBase == "" || keyBase != base {
			continue
		}
		priority := modelLookupPriority(key)
		if priority > bestPriority || (priority == bestPriority && strings.ToLower(key) > strings.ToLower(bestKey)) {
			bestPriority = priority
			bestKey = key
		}
	}
	if bestKey == "" {
		return nil, false
	}
	m, ok := data[bestKey]
	if !ok {
		return nil, false
	}
	return m, true
}

func modelLookupPriority(key string) int {
	trimmed := strings.ToLower(strings.TrimSpace(key))
	if strings.Contains(trimmed, "latest") {
		return 3
	}
	bare := trimmed
	if i := strings.LastIndex(bare, "/"); i >= 0 {
		bare = bare[i+1:]
	}
	if hasCompactDateSuffix(bare) || hasDashedDateSuffix(bare) || hasAtDateSuffix(bare) {
		return 2
	}
	return 1
}

func modelLookupBase(key string) string {
	base := strings.ToLower(strings.TrimSpace(key))
	if base == "" {
		return ""
	}
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = strings.TrimSpace(base[i+1:])
	}
	for {
		next := trimModelLookupSuffix(base)
		if next == base {
			break
		}
		base = next
	}
	return strings.Trim(base, "-_")
}

func trimModelLookupSuffix(base string) string {
	for _, suffix := range []string{"-latest", "-preview", "-exp"} {
		if strings.HasSuffix(base, suffix) {
			return strings.TrimSuffix(base, suffix)
		}
	}
	if hasAtDateSuffix(base) {
		at := strings.LastIndex(base, "@")
		if at > 0 {
			return base[:at]
		}
	}
	if hasDashedDateSuffix(base) {
		return base[:len(base)-11]
	}
	if hasCompactDateSuffix(base) {
		return base[:len(base)-9]
	}
	return base
}

func hasCompactDateSuffix(s string) bool {
	if len(s) < 9 || s[len(s)-9] != '-' {
		return false
	}
	return isDigits(s[len(s)-8:])
}

func hasDashedDateSuffix(s string) bool {
	if len(s) < 11 || s[len(s)-11] != '-' {
		return false
	}
	part := s[len(s)-10:]
	if part[4] != '-' || part[7] != '-' {
		return false
	}
	return isDigits(part[0:4]) && isDigits(part[5:7]) && isDigits(part[8:10])
}

func hasAtDateSuffix(s string) bool {
	at := strings.LastIndex(s, "@")
	if at <= 0 || at+1 >= len(s) {
		return false
	}
	tail := s[at+1:]
	if len(tail) < 6 {
		return false
	}
	return isDigits(tail)
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
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
		Model:                       modelName,
		InputCostPerToken:           getF("input_cost_per_token"),
		OutputCostPerToken:          getF("output_cost_per_token"),
		MaxTokens:                   getI("max_tokens"),
		MaxInputTokens:              getI("max_input_tokens"),
		MaxOutputTokens:             getI("max_output_tokens"),
		CacheReadInputTokenCost:     getF("cache_read_input_token_cost"),
		CacheCreationInputTokenCost: getF("cache_creation_input_token_cost"),
	}, nil
}
