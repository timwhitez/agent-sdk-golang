package agent

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestReadToolResultDumpIndex_RejectsOversizedIndexWithActionableDiagnostic(t *testing.T) {
	indexPath := writeOversizedToolResultDumpIndex(t)

	_, err := readToolResultDumpIndex(indexPath)
	if err == nil {
		t.Fatal("expected oversized index read to fail")
	}
	assertToolResultDumpIndexTooLargeDiagnostic(t, err, indexPath)
}

func TestPruneToolResultDumpIndex_RejectsOversizedIndexWithActionableDiagnostic(t *testing.T) {
	indexPath := writeOversizedToolResultDumpIndex(t)

	_, _, _, err := pruneToolResultDumpIndex(indexPath, time.Now(), time.Minute)
	if err == nil {
		t.Fatal("expected oversized index prune to fail")
	}
	assertToolResultDumpIndexTooLargeDiagnostic(t, err, indexPath)
}

func TestReclaimToolResultDumpOrphans_OversizedIndexFallsBackToMTimeCleanup(t *testing.T) {
	const dumpTTL = time.Minute
	dir := t.TempDir()
	now := time.Unix(1700000000, 0).UTC()

	expiredDump := filepath.Join(dir, "agent-tool-result-expired.txt")
	if err := os.WriteFile(expiredDump, []byte("expired"), 0o600); err != nil {
		t.Fatalf("write expired dump: %v", err)
	}
	expiredMTime := now.Add(-2 * dumpTTL)
	if err := os.Chtimes(expiredDump, expiredMTime, expiredMTime); err != nil {
		t.Fatalf("chtimes expired dump: %v", err)
	}

	indexPath := writeOversizedToolResultDumpIndexWithPath(t, filepath.Join(dir, "agent-tool-result-index-oversized.json"))

	if err := reclaimToolResultDumpOrphans(dir, now, dumpTTL); err != nil {
		t.Fatalf("reclaim orphans: %v", err)
	}
	if _, err := os.Stat(indexPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected oversized index to be removed, got err=%v", err)
	}
	if _, err := os.Stat(expiredDump); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected expired dump to be reclaimed via mtime fallback, got err=%v", err)
	}
}

func TestReclaimToolResultDumpOrphans_ScanCapPreventsFullGlobMaterializationWithActionableWarning(t *testing.T) {
	const dumpTTL = time.Minute
	dir := t.TempDir()
	now := time.Unix(1700000000, 0).UTC()

	dumpPaths := make([]string, 0, 6)
	for i := 0; i < 6; i++ {
		path := filepath.Join(dir, fmt.Sprintf("agent-tool-result-expired-%02d.txt", i))
		if err := os.WriteFile(path, []byte("expired"), 0o600); err != nil {
			t.Fatalf("write expired dump %d: %v", i, err)
		}
		old := now.Add(-2 * dumpTTL)
		if err := os.Chtimes(path, old, old); err != nil {
			t.Fatalf("chtimes expired dump %d: %v", i, err)
		}
		dumpPaths = append(dumpPaths, path)
	}

	prevCap := toolResultDumpOrphanScanCap
	prevBatch := toolResultDumpOrphanScanBatch
	toolResultDumpOrphanScanCap = 2
	toolResultDumpOrphanScanBatch = 1
	t.Cleanup(func() {
		toolResultDumpOrphanScanCap = prevCap
		toolResultDumpOrphanScanBatch = prevBatch
	})

	var logBuf bytes.Buffer
	prevWriter := log.Writer()
	prevFlags := log.Flags()
	prevPrefix := log.Prefix()
	log.SetOutput(&logBuf)
	log.SetFlags(0)
	log.SetPrefix("")
	t.Cleanup(func() {
		log.SetOutput(prevWriter)
		log.SetFlags(prevFlags)
		log.SetPrefix(prevPrefix)
	})

	if err := reclaimToolResultDumpOrphans(dir, now, dumpTTL); err != nil {
		t.Fatalf("reclaim orphans: %v", err)
	}

	remaining := 0
	for _, path := range dumpPaths {
		if _, err := os.Stat(path); err == nil {
			remaining++
			continue
		} else if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("stat dump %q: %v", path, err)
		}
	}
	if remaining == 0 {
		t.Fatalf("expected some orphan dumps to remain when scan cap is hit")
	}

	logged := logBuf.String()
	if !strings.Contains(logged, "[WARN] Tool result dump cleanup scan stopped after 2 entries (cap 2)") {
		t.Fatalf("expected scan-cap warning prefix, got %q", logged)
	}
	if !strings.Contains(logged, "warning_kind=scan_cap") {
		t.Fatalf("expected warning_kind=scan_cap in warning, got %q", logged)
	}
	if !strings.Contains(logged, "scan_truncated=true") {
		t.Fatalf("expected scan_truncated=true in warning, got %q", logged)
	}
	if !strings.Contains(logged, "scanned_entries=2") || !strings.Contains(logged, "scan_cap=2") {
		t.Fatalf("expected scan-cap metadata in warning, got %q", logged)
	}
}

func writeOversizedToolResultDumpIndex(t *testing.T) string {
	t.Helper()
	return writeOversizedToolResultDumpIndexWithPath(t, filepath.Join(t.TempDir(), "agent-tool-result-index-oversized.json"))
}

func writeOversizedToolResultDumpIndexWithPath(t *testing.T, path string) string {
	t.Helper()
	payload := strings.Repeat("x", toolResultDumpIndexMaxSize+1)
	if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
		t.Fatalf("write oversized index: %v", err)
	}
	return path
}

func assertToolResultDumpIndexTooLargeDiagnostic(t *testing.T, err error, path string) {
	t.Helper()
	if !errors.Is(err, errToolResultDumpIndexTooLarge) {
		t.Fatalf("expected errToolResultDumpIndexTooLarge, got %v", err)
	}
	var tooLarge *toolResultDumpIndexTooLargeError
	if !errors.As(err, &tooLarge) {
		t.Fatalf("expected toolResultDumpIndexTooLargeError, got %T", err)
	}
	if tooLarge.path != path {
		t.Fatalf("oversized diagnostic path = %q, want %q", tooLarge.path, path)
	}
	if tooLarge.max != toolResultDumpIndexMaxSize {
		t.Fatalf("oversized max = %d, want %d", tooLarge.max, toolResultDumpIndexMaxSize)
	}
	msg := err.Error()
	if !strings.HasPrefix(msg, "[ERROR] ") {
		t.Fatalf("expected severity prefix, got %q", msg)
	}
	if !strings.Contains(msg, " - ") {
		t.Fatalf("expected summary/action separator, got %q", msg)
	}
	if !strings.Contains(msg, "error_kind=tool_result_dump_index_too_large") {
		t.Fatalf("expected machine-readable error kind, got %q", msg)
	}
}

func TestPruneToolResultDumpIndex_RetainsEntryWhenRemoveFails(t *testing.T) {
	// Bug fix #1: Verify that when os.Remove fails (e.g., permission denied),
	// the entry is retained in the index for retry in the next cleanup cycle.
	//
	// Testing strategy: We'll test the behavior by using a non-existent directory
	// path that will definitely fail os.Remove with an error other than ErrNotExist.
	// Actually, os.Remove on a non-empty directory fails with a different error.
	// But we need a file, not a directory.
	//
	// Better approach: Create the test to verify the code logic directly.
	// We'll use a file in /proc or /sys which typically can't be deleted.

	// Try to use a file in /proc that we can read but not delete
	procFile := "/proc/version"
	if _, err := os.Stat(procFile); err != nil {
		t.Skipf("Cannot access %s for testing: %v", procFile, err)
	}

	dir := t.TempDir()
	indexPath := filepath.Join(dir, "agent-tool-result-index-test.json")

	now := time.Now().UTC()
	expiresAt := now.Add(-time.Minute) // Already expired

	// Write an index with an expired entry pointing to a file we can't delete
	idx := toolResultDumpIndexFile{
		Version:   toolResultDumpIndexVersion,
		SessionID: "test-session",
		UpdatedAt: now.UTC().Format(time.RFC3339),
		Dumps: []toolResultDumpIndexEntry{
			{
				Path:      procFile,
				CreatedAt: now.Add(-time.Hour).UTC().Format(time.RFC3339),
				ExpiresAt: expiresAt.UTC().Format(time.RFC3339),
			},
		},
	}
	b, err := jsonMarshal(idx)
	if err != nil {
		t.Fatalf("marshal index: %v", err)
	}
	if err := os.WriteFile(indexPath, b, 0o600); err != nil {
		t.Fatalf("write index: %v", err)
	}

	// Capture log output to verify warning is logged
	var logBuf bytes.Buffer
	prevWriter := log.Writer()
	prevFlags := log.Flags()
	prevPrefix := log.Prefix()
	log.SetOutput(&logBuf)
	log.SetFlags(0)
	log.SetPrefix("")
	defer func() {
		log.SetOutput(prevWriter)
		log.SetFlags(prevFlags)
		log.SetPrefix(prevPrefix)
	}()

	// Prune should fail to remove the /proc file but keep the entry in the index
	_, kept, changed, err := pruneToolResultDumpIndex(indexPath, now, time.Minute)
	if err != nil {
		t.Fatalf("prune index: %v", err)
	}

	// The entry should be kept in the index despite the failed removal
	if len(kept) == 0 {
		t.Fatal("expected entry to be kept in index when os.Remove fails")
	}
	if _, ok := kept[procFile]; !ok {
		t.Fatalf("expected proc file path %q to be in kept entries", procFile)
	}
	// Verify the kept entry has correct expiration time
	entry := kept[procFile]
	if !entry.ExpiresAt.Before(now) || entry.ExpiresAt.Sub(expiresAt) > time.Second {
		t.Errorf("kept entry has unexpected ExpiresAt: %v (expected around %v)", entry.ExpiresAt, expiresAt)
	}

	// Verify warning was logged
	logged := logBuf.String()
	if !strings.Contains(logged, "failed to cleanup expired tool result dump") {
		t.Errorf("expected warning about failed cleanup, got: %s", logged)
	}

	// changed should be false because the entry is retained
	if changed {
		t.Error("changed should be false when entry is retained due to failed removal")
	}
}

func jsonMarshal(v any) ([]byte, error) {
	return json.Marshal(v)
}
