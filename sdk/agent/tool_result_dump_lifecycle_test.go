package agent

import (
	"bytes"
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
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
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
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
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
