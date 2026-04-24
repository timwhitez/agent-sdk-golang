package tokens

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestInitializeConcurrentCallsLoadPricingOnce(t *testing.T) {
	t.Parallel()

	var calls int32
	tc := New(true)
	tc.cacheDir = t.TempDir()
	tc.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			atomic.AddInt32(&calls, 1)
			time.Sleep(20 * time.Millisecond)
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"test-model":{"input_cost_per_token":0.01,"output_cost_per_token":0.02}}`)),
				Request:    r,
			}, nil
		}),
	}

	const workers = 8
	start := make(chan struct{})
	errCh := make(chan error, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errCh <- tc.Initialize(context.Background())
		}()
	}

	close(start)
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("initialize failed: %v", err)
		}
	}

	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("expected one pricing fetch, got %d", got)
	}
}

func TestCalculateCostNewPromptTokensExcludeCachedTokens(t *testing.T) {
	t.Parallel()

	tc := &TokenCost{
		IncludeCost: true,
		initialized: true,
		pricingData: map[string]any{
			"test-model": map[string]any{
				"input_cost_per_token":  0.01,
				"output_cost_per_token": 0.02,
			},
		},
	}

	cached := 30
	usage := llm.Usage{PromptTokens: 100, CompletionTokens: 20, PromptCachedTokens: &cached}
	calc, err := tc.calculateCost(context.Background(), "test-model", usage)
	if err != nil {
		t.Fatalf("calculate cost: %v", err)
	}
	if calc == nil {
		t.Fatalf("expected cost calculation")
	}
	if calc.NewPromptTokens != 70 {
		t.Fatalf("expected new prompt tokens 70, got %d", calc.NewPromptTokens)
	}
	if diff := math.Abs(calc.NewPromptCost - 0.7); diff > 1e-9 {
		t.Fatalf("expected new prompt cost 0.7, got %f", calc.NewPromptCost)
	}
}

func TestCalculateCostClampsNegativeNewPromptTokens(t *testing.T) {
	t.Parallel()

	tc := &TokenCost{
		IncludeCost: true,
		initialized: true,
		pricingData: map[string]any{
			"test-model": map[string]any{
				"input_cost_per_token":  0.01,
				"output_cost_per_token": 0.02,
			},
		},
	}

	cached := 200
	usage := llm.Usage{PromptTokens: 100, CompletionTokens: 20, PromptCachedTokens: &cached}
	calc, err := tc.calculateCost(context.Background(), "test-model", usage)
	if err != nil {
		t.Fatalf("calculate cost: %v", err)
	}
	if calc == nil {
		t.Fatalf("expected cost calculation")
	}
	if calc.NewPromptTokens != 0 {
		t.Fatalf("expected clamped new prompt tokens 0, got %d", calc.NewPromptTokens)
	}
	if calc.NewPromptCost != 0 {
		t.Fatalf("expected clamped new prompt cost 0, got %f", calc.NewPromptCost)
	}
}

func TestLoadPricingDataWarnsWhenCacheWriteFails(t *testing.T) {
	origWarn := pricingCacheWarningf
	origWrite := writePricingCacheFile
	defer func() {
		pricingCacheWarningf = origWarn
		writePricingCacheFile = origWrite
	}()

	writePricingCacheFile = func(string, []byte, os.FileMode) error {
		return io.ErrClosedPipe
	}

	var warned bool
	pricingCacheWarningf = func(format string, args ...any) {
		warned = true
	}

	tc := New(true)
	tc.cacheDir = t.TempDir()
	tc.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"test-model":{"input_cost_per_token":0.01,"output_cost_per_token":0.02}}`)),
				Request:    r,
			}, nil
		}),
	}

	data, err := tc.loadPricingData(context.Background())
	if err != nil {
		t.Fatalf("load pricing data: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("expected fetched pricing data")
	}
	if !warned {
		t.Fatalf("expected warning when pricing cache write fails")
	}
}

func TestLoadPricingData_RejectsOversizedCacheFile(t *testing.T) {
	origWarn := pricingCacheWarningf
	defer func() {
		pricingCacheWarningf = origWarn
	}()

	cacheDir := t.TempDir()
	cacheFile := filepath.Join(cacheDir, "pricing_cache.json")
	oversized := make([]byte, int(maxPricingBodyBytes)+1)
	if err := os.WriteFile(cacheFile, oversized, 0o644); err != nil {
		t.Fatalf("write oversized cache file: %v", err)
	}

	var warnings []string
	pricingCacheWarningf = func(format string, args ...any) {
		warnings = append(warnings, sprintf(format, args...))
	}

	tc := New(true)
	tc.cacheDir = cacheDir
	tc.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"fetched-model":{"input_cost_per_token":0.01,"output_cost_per_token":0.02}}`)),
				Request:    r,
			}, nil
		}),
	}

	data, err := tc.loadPricingData(context.Background())
	if err != nil {
		t.Fatalf("load pricing data: %v", err)
	}
	if _, ok := data["fetched-model"]; !ok {
		t.Fatalf("expected fetched model data after oversized cache rejection")
	}
	if !containsWarning(warnings, "exceeds size limit") {
		t.Fatalf("expected oversized cache warning, got %v", warnings)
	}
	if !containsWarning(warnings, cacheFile) {
		t.Fatalf("expected cache path in warning, got %v", warnings)
	}
}

func TestLoadPricingData_RejectsOversizedHTTPBody(t *testing.T) {
	cacheDir := t.TempDir()
	oversized := strings.Repeat("x", int(maxPricingBodyBytes)+8)

	tc := New(true)
	tc.cacheDir = cacheDir
	tc.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(oversized)),
				Request:    r,
			}, nil
		}),
	}

	_, err := tc.loadPricingData(context.Background())
	if err == nil {
		t.Fatalf("expected oversized response body error")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "exceeds size limit") {
		t.Fatalf("expected size-limit diagnostic, got %v", err)
	}
	if !strings.Contains(strings.ToLower(err.Error()), "max") {
		t.Fatalf("expected max-size context, got %v", err)
	}
}

func TestLoadPricingData_AtomicOwnerOnlyCacheWrite(t *testing.T) {
	origWarn := pricingCacheWarningf
	defer func() {
		pricingCacheWarningf = origWarn
	}()

	cacheDir := t.TempDir()
	if err := os.Chmod(cacheDir, 0o755); err == nil {
		if info, statErr := os.Stat(cacheDir); statErr == nil && info.Mode().Perm() != 0o755 {
			t.Skipf("filesystem does not preserve widened directory permissions (got %o)", info.Mode().Perm())
		}
	}

	fixedNow := time.Date(2026, time.February, 15, 2, 0, 0, 0, time.UTC)
	cachePath := filepath.Join(cacheDir, fmt.Sprintf("pricing_%s.json", fixedNow.Format("20060102_150405")))
	if err := os.WriteFile(cachePath, []byte("original\n"), 0o644); err != nil {
		t.Fatalf("seed cache file: %v", err)
	}
	stale := fixedNow.Add(-72 * time.Hour)
	if err := os.Chtimes(cachePath, stale, stale); err != nil {
		t.Fatalf("mark cache stale: %v", err)
	}

	tc := New(true)
	tc.cacheDir = cacheDir
	tc.nowFn = func() time.Time { return fixedNow }
	tc.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"fetched-model":{"input_cost_per_token":0.01,"output_cost_per_token":0.02}}`)),
				Request:    r,
			}, nil
		}),
	}

	data, err := tc.loadPricingData(context.Background())
	if err != nil {
		t.Fatalf("load pricing data: %v", err)
	}
	if _, ok := data["fetched-model"]; !ok {
		t.Fatalf("expected fetched pricing payload")
	}

	dirInfo, err := os.Stat(cacheDir)
	if err != nil {
		t.Fatalf("stat cache dir: %v", err)
	}
	if got := dirInfo.Mode().Perm(); got != 0o700 {
		t.Fatalf("cache dir mode = %o, want %o", got, 0o700)
	}
	fileInfo, err := os.Stat(cachePath)
	if err != nil {
		t.Fatalf("stat cache file: %v", err)
	}
	if got := fileInfo.Mode().Perm(); got != 0o600 {
		t.Fatalf("cache file mode = %o, want %o", got, 0o600)
	}
	before, err := os.ReadFile(cachePath)
	if err != nil {
		t.Fatalf("read cache file before failure injection: %v", err)
	}

	if err := os.Chtimes(cachePath, stale, stale); err != nil {
		t.Fatalf("mark cache stale before rewrite: %v", err)
	}
	tc.writeCacheFileFn = func(path string, data []byte, mode os.FileMode) error {
		return atomicWritePricingCacheFileWithWriter(path, data, mode, func(f *os.File, payload []byte) error {
			if len(payload) == 0 {
				return io.ErrShortWrite
			}
			chunk := len(payload)
			if chunk > 32 {
				chunk = 32
			}
			if _, err := f.Write(payload[:chunk]); err != nil {
				return err
			}
			return io.ErrShortWrite
		})
	}

	var warnings []string
	pricingCacheWarningf = func(format string, args ...any) {
		warnings = append(warnings, sprintf(format, args...))
	}

	data, err = tc.loadPricingData(context.Background())
	if err != nil {
		t.Fatalf("load pricing data after write failure: %v", err)
	}
	if _, ok := data["fetched-model"]; !ok {
		t.Fatalf("expected fetched pricing payload after write failure")
	}
	if !containsWarning(warnings, "unable to write pricing cache") {
		t.Fatalf("expected cache write warning, got %v", warnings)
	}

	after, err := os.ReadFile(cachePath)
	if err != nil {
		t.Fatalf("read cache file after failure injection: %v", err)
	}
	if string(after) != string(before) {
		t.Fatalf("expected existing cache file unchanged on write failure")
	}
}

func TestFindValidCache_ScanCapPreventsFullDirectoryMaterializationWithActionableWarning(t *testing.T) {
	origWarn := pricingCacheWarningf
	origScanCap := pricingCacheScanCap
	origScanBatch := pricingCacheScanReadBatch
	origEntryInfo := pricingCacheEntryInfo
	defer func() {
		pricingCacheWarningf = origWarn
		pricingCacheScanCap = origScanCap
		pricingCacheScanReadBatch = origScanBatch
		pricingCacheEntryInfo = origEntryInfo
	}()

	cacheDir := t.TempDir()
	now := time.Date(2026, time.February, 16, 12, 0, 0, 0, time.UTC)
	for i := 0; i < 6; i++ {
		p := filepath.Join(cacheDir, fmt.Sprintf("pricing_%02d.json", i))
		if err := os.WriteFile(p, []byte("{}"), 0o600); err != nil {
			t.Fatalf("write cache file %d: %v", i, err)
		}
		mod := now.Add(-10*time.Minute + time.Duration(i)*time.Second)
		if err := os.Chtimes(p, mod, mod); err != nil {
			t.Fatalf("chtimes cache file %d: %v", i, err)
		}
	}

	var warnings []string
	pricingCacheWarningf = func(format string, args ...any) {
		warnings = append(warnings, sprintf(format, args...))
	}
	var infoCalls int
	pricingCacheEntryInfo = func(entry os.DirEntry) (os.FileInfo, error) {
		infoCalls++
		return entry.Info()
	}
	pricingCacheScanCap = 2
	pricingCacheScanReadBatch = 16

	tc := New(true)
	tc.cacheDir = cacheDir
	tc.nowFn = func() time.Time { return now }

	cachePath, ok := tc.findValidCache(24 * time.Hour)
	if !ok {
		t.Fatalf("expected valid cache path under scan cap")
	}
	if cachePath == "" {
		t.Fatalf("expected non-empty cache path")
	}
	if infoCalls != pricingCacheScanCap {
		t.Fatalf("expected %d scanned entries, got %d", pricingCacheScanCap, infoCalls)
	}
	if !containsWarning(warnings, "[WARN] Pricing cache scan stopped after 2 entries (cap 2)") {
		t.Fatalf("expected scan-cap warning prefix, got %v", warnings)
	}
	if !containsWarning(warnings, "warning_kind=scan_cap") {
		t.Fatalf("expected warning_kind metadata, got %v", warnings)
	}
	if !containsWarning(warnings, "scan_truncated=true") || !containsWarning(warnings, "scanned_entries=2") || !containsWarning(warnings, "scan_cap=2") {
		t.Fatalf("expected scan-cap metadata fields, got %v", warnings)
	}
}

func TestLoadPricingData_ScanCapWarningPreservesFetchFallback(t *testing.T) {
	origWarn := pricingCacheWarningf
	origScanCap := pricingCacheScanCap
	origScanBatch := pricingCacheScanReadBatch
	defer func() {
		pricingCacheWarningf = origWarn
		pricingCacheScanCap = origScanCap
		pricingCacheScanReadBatch = origScanBatch
	}()

	cacheDir := t.TempDir()
	now := time.Date(2026, time.February, 16, 12, 0, 0, 0, time.UTC)
	stale := now.Add(-72 * time.Hour)
	for i := 0; i < 3; i++ {
		p := filepath.Join(cacheDir, fmt.Sprintf("pricing_%02d.json", i))
		if err := os.WriteFile(p, []byte(`{"timestamp":"2026-01-01T00:00:00Z","data":{"old-model":{"input_cost_per_token":0.01}}}`), 0o600); err != nil {
			t.Fatalf("write stale cache file %d: %v", i, err)
		}
		if err := os.Chtimes(p, stale, stale); err != nil {
			t.Fatalf("mark stale cache file %d: %v", i, err)
		}
	}

	var warnings []string
	pricingCacheWarningf = func(format string, args ...any) {
		warnings = append(warnings, sprintf(format, args...))
	}
	pricingCacheScanCap = 1
	pricingCacheScanReadBatch = 16

	var fetchCalls int32
	tc := New(true)
	tc.cacheDir = cacheDir
	tc.nowFn = func() time.Time { return now }
	tc.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			atomic.AddInt32(&fetchCalls, 1)
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"fetched-model":{"input_cost_per_token":0.01,"output_cost_per_token":0.02}}`)),
				Request:    r,
			}, nil
		}),
	}

	data, err := tc.loadPricingData(context.Background())
	if err != nil {
		t.Fatalf("load pricing data: %v", err)
	}
	if _, ok := data["fetched-model"]; !ok {
		t.Fatalf("expected fetched model data after scan-cap fallback")
	}
	if got := atomic.LoadInt32(&fetchCalls); got != 1 {
		t.Fatalf("expected one network fetch after scan-cap fallback, got %d", got)
	}
	if !containsWarning(warnings, "[WARN] Pricing cache scan stopped after 1 entries (cap 1)") {
		t.Fatalf("expected scan-cap warning prefix, got %v", warnings)
	}
	if !containsWarning(warnings, "warning_kind=scan_cap") {
		t.Fatalf("expected warning_kind metadata, got %v", warnings)
	}
	if !containsWarning(warnings, "scan_truncated=true") || !containsWarning(warnings, "scanned_entries=1") || !containsWarning(warnings, "scan_cap=1") {
		t.Fatalf("expected scan-cap metadata fields, got %v", warnings)
	}
}

func TestGetModelPricingMatchesAliasesAndVersionFamilies(t *testing.T) {
	t.Parallel()

	tc := &TokenCost{
		IncludeCost: true,
		initialized: true,
		pricingData: map[string]any{
			"openai/gpt-4o-mini": map[string]any{
				"input_cost_per_token":  0.04,
				"output_cost_per_token": 0.08,
			},
			"openrouter/openai/gpt-4o-mini-2024-07-18": map[string]any{
				"input_cost_per_token":  0.05,
				"output_cost_per_token": 0.09,
			},
			"gemini/gemini-flash-latest": map[string]any{
				"input_cost_per_token":  0.02,
				"output_cost_per_token": 0.03,
			},
			"anthropic/claude-3-5-sonnet-20240620": map[string]any{
				"input_cost_per_token":  0.01,
				"output_cost_per_token": 0.02,
			},
			"anthropic/claude-3-5-sonnet-20241022": map[string]any{
				"input_cost_per_token":  0.011,
				"output_cost_per_token": 0.021,
			},
		},
	}

	tests := []struct {
		name      string
		modelName string
		wantInput float64
	}{
		{name: "case-insensitive openai model", modelName: "GPT-4O-MINI", wantInput: 0.04},
		{name: "multi-prefix model family fallback", modelName: "openrouter/openai/gpt-4o-mini-latest", wantInput: 0.05},
		{name: "mapped gemini alias", modelName: "gemini-flash-latest", wantInput: 0.02},
		{name: "latest family falls back to newest dated variant", modelName: "claude-3-5-sonnet-latest", wantInput: 0.011},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			pricing, err := tc.GetModelPricing(context.Background(), tt.modelName)
			if err != nil {
				t.Fatalf("GetModelPricing(%q): %v", tt.modelName, err)
			}
			if pricing == nil || pricing.InputCostPerToken == nil {
				t.Fatalf("expected pricing for %q", tt.modelName)
			}
			if diff := math.Abs(*pricing.InputCostPerToken - tt.wantInput); diff > 1e-12 {
				t.Fatalf("unexpected input cost for %q: got %f want %f", tt.modelName, *pricing.InputCostPerToken, tt.wantInput)
			}
		})
	}
}

func TestInitializeWarnsAndFallsBackWhenPricingLoadFails(t *testing.T) {
	origWarn := pricingCacheWarningf
	defer func() {
		pricingCacheWarningf = origWarn
	}()

	var warnings []string
	pricingCacheWarningf = func(format string, args ...any) {
		warnings = append(warnings, sprintf(format, args...))
	}

	tc := New(true)
	tc.cacheDir = t.TempDir()
	tc.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return nil, io.ErrUnexpectedEOF
		}),
	}

	if err := tc.Initialize(context.Background()); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	if !tc.initialized {
		t.Fatalf("expected initialized=true")
	}
	if tc.pricingData == nil || len(tc.pricingData) != 0 {
		t.Fatalf("expected empty pricing fallback, got %#v", tc.pricingData)
	}
	if !containsWarning(warnings, "pricing initialization failed") {
		t.Fatalf("expected initialization warning, got %v", warnings)
	}
	if !containsWarning(warnings, tc.cacheDir) {
		t.Fatalf("expected cache_dir context in warning, got %v", warnings)
	}
}

func TestLoadPricingDataWarnsWhenCacheReadFails(t *testing.T) {
	origWarn := pricingCacheWarningf
	origReadFile := readPricingCacheFile
	defer func() {
		pricingCacheWarningf = origWarn
		readPricingCacheFile = origReadFile
	}()

	cacheDir := t.TempDir()
	cacheFile := filepath.Join(cacheDir, "pricing_cache.json")
	if err := os.WriteFile(cacheFile, []byte(`{"timestamp":"2026-02-07T00:00:00Z","data":{"cached-model":{"input_cost_per_token":0.01}}}`), 0o644); err != nil {
		t.Fatalf("write cache file: %v", err)
	}

	readPricingCacheFile = func(path string) ([]byte, error) {
		if path == cacheFile {
			return nil, io.ErrUnexpectedEOF
		}
		return os.ReadFile(path)
	}

	var warnings []string
	pricingCacheWarningf = func(format string, args ...any) {
		warnings = append(warnings, sprintf(format, args...))
	}

	tc := New(true)
	tc.cacheDir = cacheDir
	tc.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"fetched-model":{"input_cost_per_token":0.01,"output_cost_per_token":0.02}}`)),
				Request:    r,
			}, nil
		}),
	}

	data, err := tc.loadPricingData(context.Background())
	if err != nil {
		t.Fatalf("load pricing data: %v", err)
	}
	if _, ok := data["fetched-model"]; !ok {
		t.Fatalf("expected fetched pricing data after cache read failure")
	}
	if !containsWarning(warnings, "unable to read pricing cache") {
		t.Fatalf("expected cache read warning, got %v", warnings)
	}
	if !containsWarning(warnings, cacheFile) {
		t.Fatalf("expected cache path in warning, got %v", warnings)
	}
}

func TestLoadPricingDataWarnsWhenCacheStatFails(t *testing.T) {
	origWarn := pricingCacheWarningf
	origReadDir := readPricingCacheDir
	origEntryInfo := pricingCacheEntryInfo
	defer func() {
		pricingCacheWarningf = origWarn
		readPricingCacheDir = origReadDir
		pricingCacheEntryInfo = origEntryInfo
	}()

	cacheDir := t.TempDir()
	cacheFile := filepath.Join(cacheDir, "pricing_cache.json")
	if err := os.WriteFile(cacheFile, []byte(`{"timestamp":"2026-02-07T00:00:00Z","data":{"cached-model":{"input_cost_per_token":0.01}}}`), 0o644); err != nil {
		t.Fatalf("write cache file: %v", err)
	}

	readPricingCacheDir = os.ReadDir
	pricingCacheEntryInfo = func(entry os.DirEntry) (os.FileInfo, error) {
		if entry.Name() == filepath.Base(cacheFile) {
			return nil, io.ErrClosedPipe
		}
		return entry.Info()
	}

	var warnings []string
	pricingCacheWarningf = func(format string, args ...any) {
		warnings = append(warnings, sprintf(format, args...))
	}

	tc := New(true)
	tc.cacheDir = cacheDir
	tc.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"fetched-model":{"input_cost_per_token":0.01,"output_cost_per_token":0.02}}`)),
				Request:    r,
			}, nil
		}),
	}

	data, err := tc.loadPricingData(context.Background())
	if err != nil {
		t.Fatalf("load pricing data: %v", err)
	}
	if _, ok := data["fetched-model"]; !ok {
		t.Fatalf("expected fetched pricing data after cache stat failure")
	}
	if !containsWarning(warnings, "unable to inspect pricing cache entry") {
		t.Fatalf("expected cache stat warning, got %v", warnings)
	}
	if !containsWarning(warnings, cacheFile) {
		t.Fatalf("expected cache path in warning, got %v", warnings)
	}
}

func TestLoadPricingDataWarnsWhenCacheUnmarshalFails(t *testing.T) {
	origWarn := pricingCacheWarningf
	defer func() {
		pricingCacheWarningf = origWarn
	}()

	cacheDir := t.TempDir()
	cacheFile := filepath.Join(cacheDir, "pricing_cache.json")
	if err := os.WriteFile(cacheFile, []byte("not-json"), 0o644); err != nil {
		t.Fatalf("write cache file: %v", err)
	}

	var warnings []string
	pricingCacheWarningf = func(format string, args ...any) {
		warnings = append(warnings, sprintf(format, args...))
	}

	tc := New(true)
	tc.cacheDir = cacheDir
	tc.HTTPClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"fetched-model":{"input_cost_per_token":0.01,"output_cost_per_token":0.02}}`)),
				Request:    r,
			}, nil
		}),
	}

	data, err := tc.loadPricingData(context.Background())
	if err != nil {
		t.Fatalf("load pricing data: %v", err)
	}
	if _, ok := data["fetched-model"]; !ok {
		t.Fatalf("expected fetched pricing data after cache parse failure")
	}
	if !containsWarning(warnings, "unable to parse pricing cache") {
		t.Fatalf("expected cache parse warning, got %v", warnings)
	}
	if !containsWarning(warnings, cacheFile) {
		t.Fatalf("expected cache path in warning, got %v", warnings)
	}
}

func TestAddUsageWarnsWhenCostCalculationFails(t *testing.T) {
	origWarn := pricingCacheWarningf
	defer func() {
		pricingCacheWarningf = origWarn
	}()

	var warnings []string
	pricingCacheWarningf = func(format string, args ...any) {
		warnings = append(warnings, sprintf(format, args...))
	}

	tc := &TokenCost{
		IncludeCost: true,
		initialized: true,
		pricingData: nil,
	}

	usage := llm.Usage{PromptTokens: 12, CompletionTokens: 3, TotalTokens: 15}
	entry, err := tc.AddUsage(context.Background(), "test-model", usage)
	if err != nil {
		t.Fatalf("add usage: %v", err)
	}
	if entry.Cost != nil {
		t.Fatalf("expected nil cost when calculation fails")
	}
	if !containsWarning(warnings, "cost calculation failed") {
		t.Fatalf("expected cost calculation warning, got %v", warnings)
	}
	if !containsWarning(warnings, "model=\"test-model\"") {
		t.Fatalf("expected model context in warning, got %v", warnings)
	}

	summary := tc.GetUsageSummary()
	if summary.TotalTokens != usage.TotalTokens {
		t.Fatalf("expected usage history to record tokens=%d, got %d", usage.TotalTokens, summary.TotalTokens)
	}
}

func containsWarning(warnings []string, fragment string) bool {
	for _, warning := range warnings {
		if strings.Contains(warning, fragment) {
			return true
		}
	}
	return false
}

func sprintf(format string, args ...any) string {
	return fmt.Sprintf(format, args...)
}
