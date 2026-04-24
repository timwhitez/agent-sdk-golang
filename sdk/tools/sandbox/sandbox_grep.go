package sandbox

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/bmatcuk/doublestar/v4"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type grepArgs struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path,omitempty"`

	Glob        string `json:"glob,omitempty"` // e.g. "*.go" or "**/*.ts"
	IgnoreCase  bool   `json:"ignore_case,omitempty"`
	Before      int    `json:"before,omitempty"`
	After       int    `json:"after,omitempty"`
	Context     int    `json:"context,omitempty"`
	MaxResults  int    `json:"max_results,omitempty"`  // output lines or entries, default 50
	OutputMode  string `json:"output_mode,omitempty"`  // "content"|"files_with_matches"|"count"
	LineNumbers *bool  `json:"line_numbers,omitempty"` // default true
}

var (
	grepWalkDirFn = filepath.WalkDir
	grepOpenFile  = func(path string) (*os.File, error) {
		f, _, err := openFileNoFollow(path)
		if err != nil {
			return nil, err
		}
		return f, nil
	}
	grepReadSample = readSampleBytes
	grepSeekStart  = func(f *os.File) error {
		_, err := f.Seek(0, io.SeekStart)
		return err
	}
	errGrepStop = errors.New("grep stop")
)

func grepTool() tools.Tool {
	return tools.Func[grepArgs]("grep", "Search file contents with regex", func(ctx context.Context, a grepArgs, deps *tools.Container) (any, error) {
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
		pat := a.Pattern
		if a.IgnoreCase {
			pat = "(?i)" + pat
		}
		re, err := regexp.Compile(pat)
		if err != nil {
			return "", fmt.Errorf("Invalid regex: %w", err)
		}
		before := a.Before
		after := a.After
		if a.Context > 0 {
			before = a.Context
			after = a.Context
		}
		if before < 0 {
			before = 0
		}
		if after < 0 {
			after = 0
		}
		maxOut := a.MaxResults
		if maxOut <= 0 {
			maxOut = 50
		}
		mode := strings.TrimSpace(a.OutputMode)
		if mode == "" {
			mode = "content"
		}
		showLineNumbers := true
		if a.LineNumbers != nil {
			showLineNumbers = *a.LineNumbers
		}

		globFilter, err := validateGlobArgPattern("grep", "glob", a.Glob)
		if err != nil {
			return "", err
		}

		results := []string{}
		files := []string{}
		counts := map[string]int{}
		scanTruncated := false
		permissionDenied := []string{}
		permissionDeniedSeen := map[string]struct{}{}
		scanFailures := []string{}
		scanFailureSeen := map[string]struct{}{}
		stopped := false
		scanCap := maxOut
		scannedCandidates := 0
		recordPermissionDenied := func(path string) {
			displayPath := resultPathForDisplay(s, path)
			if _, ok := permissionDeniedSeen[displayPath]; ok {
				return
			}
			permissionDeniedSeen[displayPath] = struct{}{}
			permissionDenied = append(permissionDenied, displayPath)
		}
		recordScanFailure := func(path, stage string, err error) {
			if err == nil {
				return
			}
			if errors.Is(err, os.ErrPermission) || os.IsPermission(err) {
				recordPermissionDenied(path)
				return
			}
			displayPath := resultPathForDisplay(s, path)
			entry := fmt.Sprintf("%s (%s: %v)", displayPath, stage, err)
			if _, ok := scanFailureSeen[entry]; ok {
				return
			}
			scanFailureSeen[entry] = struct{}{}
			scanFailures = append(scanFailures, entry)
		}
		walkErr := grepWalkDirFn(base, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				recordScanFailure(path, "walk", err)
				if d != nil && d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			if d.IsDir() {
				return nil
			}
			if stopped {
				return errGrepStop
			}
			if scannedCandidates >= scanCap {
				scanTruncated = true
				stopped = true
				return errGrepStop
			}
			scannedCandidates++
			// filter file path
			matchPath := resultPathForMatch(s, base, path)
			displayPath := resultPathForDisplay(s, path)
			if globFilter != "" {
				if !doublestar.MatchUnvalidated(globFilter, matchPath) {
					return nil
				}
			}

			f, err := grepOpenFile(path)
			if err != nil {
				recordScanFailure(path, "open", err)
				return nil
			}
			defer f.Close()
			sample, err := grepReadSample(f, binaryDetectSampleBytes)
			if err != nil {
				recordScanFailure(path, "read", err)
				return nil
			}
			if isBinaryData(path, sample) {
				return nil // skip binary
			}
			if err := grepSeekStart(f); err != nil {
				recordScanFailure(path, "seek", err)
				return nil
			}
			scanner := bufio.NewScanner(f)
			scannerBuf := make([]byte, 0, 64*1024)
			scanner.Buffer(scannerBuf, 1024*1024)
			type prevLine struct {
				no   int
				text string
			}
			prev := make([]prevLine, 0, before)
			pushPrev := func(no int, text string) {
				if before <= 0 {
					return
				}
				if len(prev) == before {
					copy(prev, prev[1:])
					prev[before-1] = prevLine{no: no, text: text}
					return
				}
				prev = append(prev, prevLine{no: no, text: text})
			}

			lineNo := 0
			afterRemain := 0
			lastEmitted := 0
			fileMatched := false
			matchCount := 0
			for scanner.Scan() {
				lineNo++
				line := scanner.Text()
				isMatch := re.MatchString(line)
				if isMatch {
					fileMatched = true
					matchCount++
					if mode == "files_with_matches" {
						files = append(files, displayPath)
						break
					}
					// emit before context
					for _, pl := range prev {
						if pl.no <= lastEmitted {
							continue
						}
						results = append(results, formatGrepLine(displayPath, pl.no, pl.text, showLineNumbers))
						lastEmitted = pl.no
						if len(results) >= maxOut {
							stopped = true
							break
						}
					}
					if stopped {
						break
					}
					// emit match line
					if lineNo > lastEmitted {
						results = append(results, formatGrepLine(displayPath, lineNo, line, showLineNumbers))
						lastEmitted = lineNo
						if len(results) >= maxOut {
							stopped = true
							break
						}
					}
					if afterRemain < after {
						afterRemain = after
					}
				} else if afterRemain > 0 {
					if mode == "content" {
						if lineNo > lastEmitted {
							results = append(results, formatGrepLine(displayPath, lineNo, line, showLineNumbers))
							lastEmitted = lineNo
							if len(results) >= maxOut {
								stopped = true
								break
							}
						}
					}
					afterRemain--
				}
				pushPrev(lineNo, line)
			}
			if err := scanner.Err(); err != nil {
				recordScanFailure(path, "scan", err)
			}
			if mode == "count" {
				if matchCount > 0 {
					counts[displayPath] += matchCount
					if len(counts) >= scanCap {
						scanTruncated = true
						stopped = true
						return errGrepStop
					}
				}
				return nil
			}
			if stopped {
				return errGrepStop
			}
			if fileMatched && mode == "files_with_matches" {
				if len(files) >= scanCap {
					scanTruncated = true
					stopped = true
					return errGrepStop
				}
				// already added
				return nil
			}
			return nil
		})
		if walkErr != nil && !errors.Is(walkErr, errGrepStop) {
			recordScanFailure(base, "walk", walkErr)
		}
		warning := grepPermissionWarning(permissionDenied)
		scanWarning := grepScanWarning(scanFailures)
		capWarning := ""
		if scanTruncated {
			capWarning = formatWarningDiagnostic(
				fmt.Sprintf("grep scan stopped after %d file candidate(s) at cap %d", scannedCandidates, scanCap),
				"Narrow the path/glob and rerun to inspect remaining matches.",
			)
		}
		if skipped := len(permissionDenied) + len(scanFailures); skipped > 0 {
			meta := map[string]any{
				"warning_kind":  partialScanWarningKind,
				"skipped_count": skipped,
			}
			switch {
			case len(permissionDenied) > 0 && len(scanFailures) > 0:
				meta["skipped_reason"] = "multiple"
			case len(permissionDenied) > 0:
				meta["skipped_reason"] = "permission_denied"
			default:
				meta["skipped_reason"] = "scan_error"
			}
			if len(permissionDenied) > 0 {
				meta["skipped_permission_denied"] = len(permissionDenied)
				meta["skipped_permission_samples"] = sampleWarningValues(permissionDenied, 5)
			}
			if len(scanFailures) > 0 {
				meta["skipped_scan_errors"] = len(scanFailures)
				meta["skipped_scan_samples"] = sampleWarningValues(scanFailures, 5)
			}
			tools.UpsertToolResultMetadata(ctx, meta)
		}
		if scanTruncated {
			meta := map[string]any{
				"scan_truncated":     true,
				"scan_cap":           scanCap,
				"scanned_candidates": scannedCandidates,
				"scanned_files":      scannedCandidates,
				"skipped_due_to_cap": true,
			}
			if len(permissionDenied) == 0 && len(scanFailures) == 0 {
				meta["warning_kind"] = scanCapWarningKind
			}
			tools.UpsertToolResultMetadata(ctx, meta)
		}
		switch mode {
		case "files_with_matches":
			if len(files) == 0 {
				return appendGrepWarning("No matches for: "+a.Pattern, warning, scanWarning, capWarning), nil
			}
			// unique + cap
			uniq := []string{}
			seen := map[string]struct{}{}
			for _, f := range files {
				if _, ok := seen[f]; ok {
					continue
				}
				seen[f] = struct{}{}
				uniq = append(uniq, f)
				if len(uniq) >= maxOut {
					break
				}
			}
			return appendGrepWarning(strings.Join(uniq, "\n"), warning, scanWarning, capWarning), nil
		case "count":
			if len(counts) == 0 {
				return appendGrepWarning("No matches for: "+a.Pattern, warning, scanWarning, capWarning), nil
			}
			lines := []string{}
			keys := make([]string, 0, len(counts))
			for k := range counts {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				lines = append(lines, fmt.Sprintf("%s: %d", k, counts[k]))
				if len(lines) >= maxOut {
					break
				}
			}
			return appendGrepWarning(strings.Join(lines, "\n"), warning, scanWarning, capWarning), nil
		default:
			if len(results) == 0 {
				return appendGrepWarning("No matches for: "+a.Pattern, warning, scanWarning, capWarning), nil
			}
			return appendGrepWarning(strings.Join(results, "\n"), warning, scanWarning, capWarning), nil
		}
	})
}

func grepPermissionWarning(paths []string) string {
	if len(paths) == 0 {
		return ""
	}
	sort.Strings(paths)
	shown := paths
	extra := 0
	if len(shown) > 5 {
		extra = len(shown) - 5
		shown = shown[:5]
	}
	msg := fmt.Sprintf("grep skipped %d path(s) due permission denied: %s", len(paths), strings.Join(shown, ", "))
	if extra > 0 {
		msg += fmt.Sprintf(" (+%d more)", extra)
	}
	return formatWarningDiagnostic(msg, "Adjust path permissions or narrow the search scope, then retry for complete results.")
}

func grepScanWarning(failures []string) string {
	if len(failures) == 0 {
		return ""
	}
	sort.Strings(failures)
	shown := failures
	extra := 0
	if len(shown) > 5 {
		extra = len(shown) - 5
		shown = shown[:5]
	}
	msg := fmt.Sprintf("grep skipped %d path(s) due scan errors: %s", len(failures), strings.Join(shown, ", "))
	if extra > 0 {
		msg += fmt.Sprintf(" (+%d more)", extra)
	}
	return formatWarningDiagnostic(msg, "Check skipped files and retry the search to confirm complete results.")
}

func appendGrepWarning(body string, warnings ...string) string {
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

func sampleWarningValues(values []string, limit int) []string {
	if len(values) == 0 || limit <= 0 {
		return nil
	}
	copied := append([]string(nil), values...)
	sort.Strings(copied)
	if len(copied) > limit {
		return append([]string(nil), copied[:limit]...)
	}
	return copied
}

func formatGrepLine(file string, lineNo int, line string, showLineNumbers bool) string {
	line = truncate(line, 200)
	if !showLineNumbers {
		return fmt.Sprintf("%s: %s", file, line)
	}
	return fmt.Sprintf("%s:%d: %s", file, lineNo, line)
}
