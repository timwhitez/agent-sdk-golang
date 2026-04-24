package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/bmatcuk/doublestar/v4"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// globArgs are the arguments for the glob tool.
type globArgs struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path,omitempty"`
}

// Constants for glob tool behavior.
const (
	globSkippedSampleLimit = 3
	partialScanWarningKind = "partial_scan_failure"
	scanCapWarningKind     = "scan_cap_reached"
	maxGlobScanFiles       = 200
)

// Variables for testing overrides.
var (
	globStatFile          = os.Lstat
	globWalkDir           = filepath.WalkDir
	errGlobScanCapReached = errors.New("glob scan cap reached")
)

// globSkippedDiagnostics tracks skipped paths during glob scanning.
type globSkippedDiagnostics struct {
	count   int
	samples []string
}

// add adds a path to the diagnostics, up to the sample limit.
func (d *globSkippedDiagnostics) add(path string) {
	if d == nil {
		return
	}
	d.count++
	if len(d.samples) >= globSkippedSampleLimit {
		return
	}
	d.samples = append(d.samples, path)
}

// globSkippedWarning creates a warning for paths skipped due to stat errors.
func globSkippedWarning(diag globSkippedDiagnostics) string {
	if diag.count == 0 {
		return ""
	}
	summary := ""
	if len(diag.samples) == 0 {
		summary = fmt.Sprintf("glob skipped %d matched path(s) due stat errors", diag.count)
	} else {
		summary = fmt.Sprintf(
			"glob skipped %d matched path(s) due stat errors: %s",
			diag.count,
			strings.Join(diag.samples, ", "),
		)
	}
	return formatWarningDiagnostic(
		summary,
		"Review skipped paths and permissions, then rerun glob if you need complete results.",
	)
}

// globSymlinkWarning creates a warning for paths skipped due to being symbolic links.
func globSymlinkWarning(diag globSkippedDiagnostics) string {
	if diag.count == 0 {
		return ""
	}
	summary := ""
	if len(diag.samples) == 0 {
		summary = fmt.Sprintf("glob skipped %d matched path(s) that are symbolic links", diag.count)
	} else {
		summary = fmt.Sprintf(
			"glob skipped %d matched path(s) that are symbolic links: %s",
			diag.count,
			strings.Join(diag.samples, ", "),
		)
	}
	return formatWarningDiagnostic(
		summary,
		"Inspect symbolic-link targets and read real files inside the sandbox to get complete results.",
	)
}

// globScanCapWarning creates a warning when the scan cap is reached.
func globScanCapWarning(scanned, cap int) string {
	if cap <= 0 || scanned <= 0 {
		return ""
	}
	return formatWarningDiagnostic(
		fmt.Sprintf("glob scan stopped after %d file candidate(s) at cap %d", scanned, cap),
		"Narrow the path or pattern and rerun to inspect remaining matches.",
	)
}

// appendGlobWarning appends warnings to a body string.
func appendGlobWarning(body string, warnings ...string) string {
	out := body
	for _, warning := range warnings {
		if strings.TrimSpace(warning) == "" {
			continue
		}
		if strings.TrimSpace(out) == "" {
			out = warning
			continue
		}
		out += "\n" + warning
	}
	return out
}

// resultPathForDisplay returns a display path relative to the sandbox root if possible.
func resultPathForDisplay(s *Sandbox, abs string) string {
	if s != nil && isWithinRoot(abs, s.RootDir) {
		if rel, err := filepath.Rel(s.RootDir, abs); err == nil {
			return filepath.ToSlash(rel)
		}
	}
	return filepath.ToSlash(abs)
}

// resultPathForMatch returns a display path for a match, trying sandbox root first.
func resultPathForMatch(s *Sandbox, base string, abs string) string {
	if s != nil && isWithinRoot(abs, s.RootDir) {
		if rel, err := filepath.Rel(s.RootDir, abs); err == nil {
			return filepath.ToSlash(rel)
		}
	}
	if strings.TrimSpace(base) != "" {
		if rel, err := filepath.Rel(base, abs); err == nil {
			return filepath.ToSlash(rel)
		}
	}
	return filepath.ToSlash(abs)
}

// globTool returns the glob tool.
func globTool() tools.Tool {
	return tools.Func[globArgs]("glob", "Find files matching a glob pattern", func(ctx context.Context, a globArgs, deps *tools.Container) (any, error) {
		const maxGlobResults = 50
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		basePath := "."
		if strings.TrimSpace(a.Path) != "" {
			basePath = a.Path
		}
		baseAccessPath, err := s.ResolveAccessPath(basePath)
		if err != nil {
			return "", fmt.Errorf("Security error: %w", err)
		}
		base, err := s.RevalidateAccessPath(baseAccessPath)
		if err != nil {
			return "", fmt.Errorf("Security error: %w", err)
		}
		pat := strings.TrimSpace(a.Pattern)
		if pat == "" {
			return "", fmt.Errorf("empty pattern")
		}
		// Support ** patterns (doublestar) and normal * patterns.
		pat = filepath.ToSlash(pat)
		files := make([]string, 0, maxGlobResults)
		totalFiles := 0
		scannedCandidates := 0
		scanTruncated := false
		skippedStat := globSkippedDiagnostics{}
		skippedSymlink := globSkippedDiagnostics{}
		walkErr := globWalkDir(base, func(path string, d os.DirEntry, walkErr error) error {
			if walkErr != nil {
				if d != nil && d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			if d.IsDir() {
				return nil
			}
			if scannedCandidates >= maxGlobScanFiles {
				scanTruncated = true
				return errGlobScanCapReached
			}
			scannedCandidates++

			rel, err := filepath.Rel(base, path)
			if err != nil {
				return nil
			}
			if !doublestar.MatchUnvalidated(pat, filepath.ToSlash(rel)) {
				return nil
			}

			displayPath := resultPathForDisplay(s, path)
			st, err := globStatFile(path)
			if err != nil {
				skippedStat.add(displayPath)
				return nil
			}
			if st.Mode()&os.ModeSymlink != 0 {
				skippedSymlink.add(displayPath)
				return nil
			}
			candidateAccessPath, err := s.ResolveAccessPath(path)
			if err != nil {
				if isSymlinkSecurityError(err) {
					skippedSymlink.add(displayPath)
				} else {
					skippedStat.add(displayPath)
				}
				return nil
			}
			candidatePath, err := s.RevalidateAccessPath(candidateAccessPath)
			if err != nil {
				if isSymlinkSecurityError(err) {
					skippedSymlink.add(displayPath)
				} else {
					skippedStat.add(displayPath)
				}
				return nil
			}
			f, openedInfo, err := openFileNoFollow(candidatePath)
			if err != nil {
				if isSymlinkSecurityError(err) {
					skippedSymlink.add(displayPath)
				} else {
					skippedStat.add(displayPath)
				}
				return nil
			}
			_ = f.Close()
			if openedInfo.IsDir() {
				return nil
			}
			totalFiles++
			if len(files) < maxGlobResults {
				files = append(files, displayPath)
			}
			return nil
		})
		if walkErr != nil && !errors.Is(walkErr, errGlobScanCapReached) {
			return "", fmt.Errorf("glob scan failed: %w", walkErr)
		}
		statWarning := globSkippedWarning(skippedStat)
		symlinkWarning := globSymlinkWarning(skippedSymlink)
		scanWarning := ""
		if scanTruncated {
			scanWarning = globScanCapWarning(scannedCandidates, maxGlobScanFiles)
		}
		skippedCount := skippedStat.count + skippedSymlink.count
		skippedReason := ""
		switch {
		case skippedStat.count > 0 && skippedSymlink.count > 0:
			skippedReason = "multiple"
		case skippedStat.count > 0:
			skippedReason = "stat_error"
		case skippedSymlink.count > 0:
			skippedReason = "symlink_target"
		}
		meta := map[string]any{
			"count":     totalFiles,
			"truncated": totalFiles > len(files) || scanTruncated,
		}
		if skippedCount > 0 {
			meta["has_errors"] = true
			meta["warning_kind"] = partialScanWarningKind
			meta["skipped_count"] = skippedCount
			meta["skipped_reason"] = skippedReason
		}
		if skippedStat.count > 0 {
			meta["skipped_paths"] = skippedStat.count
			meta["skipped_path_samples"] = append([]string(nil), skippedStat.samples...)
		}
		if skippedSymlink.count > 0 {
			meta["skipped_symlink_paths"] = skippedSymlink.count
			meta["skipped_symlink_samples"] = append([]string(nil), skippedSymlink.samples...)
		}
		if scanTruncated {
			meta["scan_truncated"] = true
			meta["scan_cap"] = maxGlobScanFiles
			meta["scanned_candidates"] = scannedCandidates
			meta["scanned_files"] = scannedCandidates
			meta["skipped_due_to_cap"] = true
			if skippedCount == 0 {
				meta["warning_kind"] = scanCapWarningKind
			}
		}
		tools.UpsertToolResultMetadata(ctx, meta)
		if totalFiles == 0 {
			return appendGlobWarning("No files match pattern: "+a.Pattern, statWarning, symlinkWarning, scanWarning), nil
		}
		header := ""
		if scanTruncated {
			header = fmt.Sprintf("Found at least %d file(s). Showing first %d (limit %d, scan cap %d). Refine the pattern or path to see more.", totalFiles, len(files), maxGlobResults, maxGlobScanFiles)
		} else if totalFiles > len(files) {
			header = fmt.Sprintf("Found %d file(s). Showing first %d (limit %d). Refine the pattern or path to see more.", totalFiles, len(files), maxGlobResults)
		} else {
			header = fmt.Sprintf("Found %d file(s):", totalFiles)
		}
		return appendGlobWarning(fmt.Sprintf("%s\n%s", header, strings.Join(files, "\n")), statWarning, symlinkWarning, scanWarning), nil
	})
}
