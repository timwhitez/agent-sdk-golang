package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"time"
)

const (
	toolResultDumpDirName      = "agent-tool-result-dumps"
	toolResultDumpIndexPattern = "agent-tool-result-index-*.json"
	toolResultDumpIndexVersion = 1
	toolResultDumpExpiryPolicy = "ttl_then_mtime_scan"
	toolResultDumpIndexMaxSize = 1 << 20
)

var (
	toolResultDumpNow = time.Now
	toolResultDumpDir = defaultToolResultDumpDir

	toolResultDumpSessionCounter   uint64
	errToolResultDumpIndexTooLarge = errors.New("tool_result_dump_index_too_large")
	toolResultDumpOrphanScanCap    = 10_000
	toolResultDumpOrphanScanBatch  = 256
)

type toolResultDumpLifecycleEntry struct {
	CreatedAt time.Time
	ExpiresAt time.Time
}

type toolResultDumpIndexFile struct {
	Version   int                        `json:"version"`
	SessionID string                     `json:"session_id,omitempty"`
	UpdatedAt string                     `json:"updated_at,omitempty"`
	Dumps     []toolResultDumpIndexEntry `json:"dumps,omitempty"`
}

type toolResultDumpIndexEntry struct {
	Path      string `json:"path"`
	CreatedAt string `json:"created_at"`
	ExpiresAt string `json:"expires_at"`
}

type toolResultDumpIndexTooLargeError struct {
	path string
	size int64
	max  int64
}

func (e *toolResultDumpIndexTooLargeError) Error() string {
	return fmt.Sprintf("[ERROR] Tool result dump index file is too large - Reduce %q to <= %d bytes (current %d), then retry startup cleanup. (error_kind=tool_result_dump_index_too_large)", e.path, e.max, e.size)
}

func (e *toolResultDumpIndexTooLargeError) Unwrap() error {
	return errToolResultDumpIndexTooLarge
}

func defaultToolResultDumpDir() string {
	return filepath.Join(os.TempDir(), toolResultDumpDirName)
}

func resolveToolResultDumpDir() (string, error) {
	if toolResultDumpDir == nil {
		return "", fmt.Errorf("tool result dump dir provider is nil")
	}
	dir := strings.TrimSpace(toolResultDumpDir())
	if dir == "" {
		return "", fmt.Errorf("tool result dump dir is empty")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	return dir, nil
}

func (a *Agent) initToolResultDumpLifecycle(now time.Time) {
	dir, err := resolveToolResultDumpDir()
	if err != nil {
		a.warnf("warning: failed to initialize tool result dump lifecycle directory: %v", err)
		return
	}
	if err := reclaimToolResultDumpOrphansWithWarning(dir, now, a.toolResultDumpTTL, a.warnf); err != nil {
		a.warnf("warning: failed to cleanup orphan tool result dumps: %v", err)
	}

	a.toolResultDumpsMu.Lock()
	a.toolResultDumpDir = dir
	a.toolResultDumpID = newToolResultDumpSessionID(now)
	a.toolResultDumpIdx = filepath.Join(dir, fmt.Sprintf("agent-tool-result-index-%s.json", a.toolResultDumpID))
	a.toolResultDumpsMu.Unlock()
}

func newToolResultDumpSessionID(now time.Time) string {
	seq := atomic.AddUint64(&toolResultDumpSessionCounter, 1)
	return fmt.Sprintf("%d-%d-%d", os.Getpid(), now.UnixNano(), seq)
}

func (a *Agent) ensureToolResultDumpIndexLocked(now time.Time) {
	if strings.TrimSpace(a.toolResultDumpIdx) != "" {
		return
	}
	dir := strings.TrimSpace(a.toolResultDumpDir)
	if dir == "" {
		resolved, err := resolveToolResultDumpDir()
		if err != nil {
			a.warnf("warning: failed to resolve tool result dump directory for index: %v", err)
			return
		}
		dir = resolved
		a.toolResultDumpDir = dir
	}
	if strings.TrimSpace(a.toolResultDumpID) == "" {
		a.toolResultDumpID = newToolResultDumpSessionID(now)
	}
	a.toolResultDumpIdx = filepath.Join(dir, fmt.Sprintf("agent-tool-result-index-%s.json", a.toolResultDumpID))
}

func (a *Agent) registerToolResultDump(path string, now time.Time) toolResultDumpLifecycleEntry {
	if strings.TrimSpace(path) == "" {
		return toolResultDumpLifecycleEntry{}
	}
	entry := toolResultDumpLifecycleEntry{
		CreatedAt: now,
		ExpiresAt: now.Add(a.toolResultDumpTTL),
	}
	a.toolResultDumpsMu.Lock()
	if a.toolResultDumps == nil {
		a.toolResultDumps = make(map[string]toolResultDumpLifecycleEntry)
	}
	a.ensureToolResultDumpIndexLocked(now)
	a.toolResultDumps[path] = entry
	snapshot := cloneToolResultDumpEntries(a.toolResultDumps)
	indexPath := a.toolResultDumpIdx
	sessionID := a.toolResultDumpID
	a.toolResultDumpsMu.Unlock()

	if err := writeToolResultDumpIndex(indexPath, sessionID, snapshot, now); err != nil {
		a.warnf("warning: failed to persist tool result dump index %q: %v", indexPath, err)
	}
	return entry
}

func (a *Agent) cleanupToolResultDumps(now time.Time, removeAll bool) {
	a.toolResultDumpsMu.Lock()
	indexPath := a.toolResultDumpIdx
	sessionID := a.toolResultDumpID
	if len(a.toolResultDumps) == 0 {
		a.toolResultDumpsMu.Unlock()
		if removeAll && strings.TrimSpace(indexPath) != "" {
			if err := writeToolResultDumpIndex(indexPath, sessionID, nil, now); err != nil {
				a.warnf("warning: failed to cleanup tool result dump index %q: %v", indexPath, err)
			}
		}
		return
	}
	candidates := make([]string, 0, len(a.toolResultDumps))
	for path, entry := range a.toolResultDumps {
		if removeAll || !entry.ExpiresAt.After(now) {
			candidates = append(candidates, path)
		}
	}
	a.toolResultDumpsMu.Unlock()

	changed := false
	for _, path := range candidates {
		err := os.Remove(path)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			a.warnf("warning: failed to cleanup tool result dump %q: %v", path, err)
			continue
		}
		a.toolResultDumpsMu.Lock()
		if _, ok := a.toolResultDumps[path]; ok {
			delete(a.toolResultDumps, path)
			changed = true
		}
		a.toolResultDumpsMu.Unlock()
	}
	if !changed && !removeAll {
		return
	}
	a.toolResultDumpsMu.Lock()
	snapshot := cloneToolResultDumpEntries(a.toolResultDumps)
	indexPath = a.toolResultDumpIdx
	sessionID = a.toolResultDumpID
	a.toolResultDumpsMu.Unlock()
	if err := writeToolResultDumpIndex(indexPath, sessionID, snapshot, now); err != nil {
		a.warnf("warning: failed to update tool result dump index %q: %v", indexPath, err)
	}
}

func cloneToolResultDumpEntries(src map[string]toolResultDumpLifecycleEntry) map[string]toolResultDumpLifecycleEntry {
	if len(src) == 0 {
		return map[string]toolResultDumpLifecycleEntry{}
	}
	out := make(map[string]toolResultDumpLifecycleEntry, len(src))
	for path, entry := range src {
		out[path] = entry
	}
	return out
}

func reclaimToolResultDumpOrphans(dir string, now time.Time, ttl time.Duration) error {
	return reclaimToolResultDumpOrphansWithWarning(dir, now, ttl, log.Printf)
}

func reclaimToolResultDumpOrphansWithWarning(dir string, now time.Time, ttl time.Duration, warnf func(string, ...any)) error {
	if warnf == nil {
		warnf = func(string, ...any) {}
	}
	scanResult, err := scanToolResultDumpPaths(dir, toolResultDumpOrphanScanCap, toolResultDumpOrphanScanBatch)
	if err != nil {
		return fmt.Errorf("scan dump directory: %w", err)
	}
	if scanResult.scanTruncated {
		warnf("warning: %s", toolResultDumpScanCapWarning(scanResult.scannedEntries, scanResult.scanCap))
	}
	indexPaths := scanResult.indexPaths
	tracked := make(map[string]struct{})
	for _, indexPath := range indexPaths {
		sessionID, kept, changed, err := pruneToolResultDumpIndexWithWarning(indexPath, now, ttl, warnf)
		if err != nil {
			var oversizedErr *toolResultDumpIndexTooLargeError
			if errors.As(err, &oversizedErr) {
				warnf("warning: %v", err)
				if rmErr := os.Remove(indexPath); rmErr != nil && !errors.Is(rmErr, os.ErrNotExist) {
					warnf("warning: failed to remove oversized tool result dump index %q: %v", indexPath, rmErr)
				}
				continue
			}
			warnf("warning: failed to read tool result dump index %q: %v", indexPath, err)
			continue
		}
		if len(kept) == 0 {
			if err := os.Remove(indexPath); err != nil && !errors.Is(err, os.ErrNotExist) {
				warnf("warning: failed to cleanup empty tool result dump index %q: %v", indexPath, err)
			}
			continue
		}
		if changed {
			if err := writeToolResultDumpIndex(indexPath, sessionID, kept, now); err != nil {
				warnf("warning: failed to compact tool result dump index %q: %v", indexPath, err)
			}
		}
		for path := range kept {
			tracked[path] = struct{}{}
		}
	}

	dumpPaths := scanResult.dumpPaths
	for _, path := range dumpPaths {
		if _, ok := tracked[path]; ok {
			continue
		}
		expired, err := dumpExpiredByMTime(path, now, ttl)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			warnf("warning: failed to inspect orphan tool result dump %q: %v", path, err)
			continue
		}
		if !expired {
			continue
		}
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			warnf("warning: failed to cleanup orphan tool result dump %q: %v", path, err)
		}
	}
	return nil
}

type toolResultDumpPathScanResult struct {
	indexPaths     []string
	dumpPaths      []string
	scannedEntries int
	scanCap        int
	scanTruncated  bool
}

func scanToolResultDumpPaths(dir string, scanCap, batchSize int) (toolResultDumpPathScanResult, error) {
	result := toolResultDumpPathScanResult{
		scanCap: scanCap,
	}
	if batchSize <= 0 {
		batchSize = 1
	}

	f, err := os.Open(dir)
	if err != nil {
		return result, err
	}
	defer f.Close()

	for {
		entries, readErr := f.ReadDir(batchSize)
		for _, entry := range entries {
			if scanCap > 0 && result.scannedEntries >= scanCap {
				result.scanTruncated = true
				break
			}
			result.scannedEntries++
			if entry.IsDir() {
				continue
			}
			name := entry.Name()
			if ok, err := filepath.Match(toolResultDumpIndexPattern, name); err == nil && ok {
				result.indexPaths = append(result.indexPaths, filepath.Join(dir, name))
			}
			if ok, err := filepath.Match(toolResultDumpPattern, name); err == nil && ok {
				result.dumpPaths = append(result.dumpPaths, filepath.Join(dir, name))
			}
		}
		if result.scanTruncated {
			break
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return result, readErr
		}
		if len(entries) == 0 {
			break
		}
	}

	sort.Strings(result.indexPaths)
	sort.Strings(result.dumpPaths)
	return result, nil
}

func toolResultDumpScanCapWarning(scannedEntries, scanCap int) string {
	return fmt.Sprintf("[WARN] Tool result dump cleanup scan stopped after %d entries (cap %d) - Re-run cleanup to continue reclaiming stale dump artifacts. (warning_kind=scan_cap scan_truncated=true scanned_entries=%d scan_cap=%d)", scannedEntries, scanCap, scannedEntries, scanCap)
}

func pruneToolResultDumpIndex(indexPath string, now time.Time, ttl time.Duration) (string, map[string]toolResultDumpLifecycleEntry, bool, error) {
	return pruneToolResultDumpIndexWithWarning(indexPath, now, ttl, log.Printf)
}

func pruneToolResultDumpIndexWithWarning(indexPath string, now time.Time, ttl time.Duration, warnf func(string, ...any)) (string, map[string]toolResultDumpLifecycleEntry, bool, error) {
	if warnf == nil {
		warnf = func(string, ...any) {}
	}
	idx, err := readToolResultDumpIndex(indexPath)
	if err != nil {
		return "", nil, false, err
	}
	kept := make(map[string]toolResultDumpLifecycleEntry, len(idx.Dumps))
	changed := false
	for _, raw := range idx.Dumps {
		path := strings.TrimSpace(raw.Path)
		if path == "" {
			changed = true
			continue
		}
		entry, err := parseToolResultDumpIndexEntry(raw)
		if err != nil {
			derived, derr := dumpLifecycleFromMTime(path, ttl)
			if derr != nil {
				if errors.Is(derr, os.ErrNotExist) {
					changed = true
					continue
				}
				warnf("warning: failed to derive lifecycle for tool result dump %q: %v", path, derr)
				changed = true
				continue
			}
			entry = derived
			changed = true
		}
		if !entry.ExpiresAt.After(now) {
			if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
				warnf("warning: failed to cleanup expired tool result dump %q: %v", path, err)
				// Keep the entry in the index for retry in the next cleanup cycle
				kept[path] = entry
			} else {
				// Only mark changed if the file was successfully removed or didn't exist
				changed = true
			}
			continue
		}
		if _, err := os.Stat(path); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				changed = true
				continue
			}
			warnf("warning: failed to stat indexed tool result dump %q: %v", path, err)
		}
		kept[path] = entry
	}
	if len(kept) != len(idx.Dumps) {
		changed = true
	}
	return strings.TrimSpace(idx.SessionID), kept, changed, nil
}

func readToolResultDumpIndex(path string) (toolResultDumpIndexFile, error) {
	var idx toolResultDumpIndexFile
	b, err := readToolResultDumpIndexBytes(path, toolResultDumpIndexMaxSize)
	if err != nil {
		return idx, err
	}
	if err := json.Unmarshal(b, &idx); err != nil {
		return idx, err
	}
	return idx, nil
}

func readToolResultDumpIndexBytes(path string, maxSize int64) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.Mode().IsRegular() && info.Size() > maxSize {
		return nil, &toolResultDumpIndexTooLargeError{path: path, size: info.Size(), max: maxSize}
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	b, err := io.ReadAll(io.LimitReader(f, maxSize+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > maxSize {
		return nil, &toolResultDumpIndexTooLargeError{path: path, size: int64(len(b)), max: maxSize}
	}
	return b, nil
}

func parseToolResultDumpIndexEntry(entry toolResultDumpIndexEntry) (toolResultDumpLifecycleEntry, error) {
	createdAt, err := time.Parse(time.RFC3339, strings.TrimSpace(entry.CreatedAt))
	if err != nil {
		return toolResultDumpLifecycleEntry{}, fmt.Errorf("parse created_at: %w", err)
	}
	expiresAt, err := time.Parse(time.RFC3339, strings.TrimSpace(entry.ExpiresAt))
	if err != nil {
		return toolResultDumpLifecycleEntry{}, fmt.Errorf("parse expires_at: %w", err)
	}
	if !expiresAt.After(createdAt) {
		return toolResultDumpLifecycleEntry{}, fmt.Errorf("expires_at must be after created_at")
	}
	return toolResultDumpLifecycleEntry{CreatedAt: createdAt, ExpiresAt: expiresAt}, nil
}

func dumpLifecycleFromMTime(path string, ttl time.Duration) (toolResultDumpLifecycleEntry, error) {
	info, err := os.Stat(path)
	if err != nil {
		return toolResultDumpLifecycleEntry{}, err
	}
	createdAt := info.ModTime()
	return toolResultDumpLifecycleEntry{
		CreatedAt: createdAt,
		ExpiresAt: createdAt.Add(ttl),
	}, nil
}

func dumpExpiredByMTime(path string, now time.Time, ttl time.Duration) (bool, error) {
	info, err := os.Stat(path)
	if err != nil {
		return false, err
	}
	return !info.ModTime().Add(ttl).After(now), nil
}

func writeToolResultDumpIndex(indexPath, sessionID string, dumps map[string]toolResultDumpLifecycleEntry, now time.Time) error {
	indexPath = strings.TrimSpace(indexPath)
	if indexPath == "" {
		return nil
	}
	if len(dumps) == 0 {
		if err := os.Remove(indexPath); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return nil
	}
	keys := make([]string, 0, len(dumps))
	for path := range dumps {
		path = strings.TrimSpace(path)
		if path == "" {
			continue
		}
		keys = append(keys, path)
	}
	sort.Strings(keys)

	entries := make([]toolResultDumpIndexEntry, 0, len(keys))
	for _, path := range keys {
		entry := dumps[path]
		createdAt := entry.CreatedAt
		if createdAt.IsZero() {
			createdAt = now
		}
		expiresAt := entry.ExpiresAt
		if expiresAt.IsZero() || !expiresAt.After(createdAt) {
			expiresAt = createdAt.Add(time.Second)
		}
		entries = append(entries, toolResultDumpIndexEntry{
			Path:      path,
			CreatedAt: createdAt.UTC().Format(time.RFC3339),
			ExpiresAt: expiresAt.UTC().Format(time.RFC3339),
		})
	}
	if len(entries) == 0 {
		if err := os.Remove(indexPath); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return nil
	}

	state := toolResultDumpIndexFile{
		Version:   toolResultDumpIndexVersion,
		SessionID: strings.TrimSpace(sessionID),
		UpdatedAt: now.UTC().Format(time.RFC3339),
		Dumps:     entries,
	}
	b, err := json.Marshal(state)
	if err != nil {
		return err
	}
	return writeFileAtomic(indexPath, b, 0o600)
}

func writeFileAtomic(path string, data []byte, mode os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".tool-result-index-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}
	if err := tmp.Chmod(mode); err != nil {
		cleanup()
		return err
	}
	n, err := tmp.Write(data)
	if err != nil {
		cleanup()
		return err
	}
	if n != len(data) {
		cleanup()
		return io.ErrShortWrite
	}
	if err := tmp.Sync(); err != nil {
		cleanup()
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		if removeErr := os.Remove(path); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			_ = os.Remove(tmpPath)
			return fmt.Errorf("rename index: %w (remove old index: %v)", err, removeErr)
		}
		if retryErr := os.Rename(tmpPath, path); retryErr != nil {
			_ = os.Remove(tmpPath)
			return retryErr
		}
	}
	return nil
}
