package sandbox

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/bmatcuk/doublestar/v4"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// ============================================================================
// ls tool: List files and directories
// ============================================================================

// lsArgs are the arguments for the ls tool.
type lsArgs struct {
	Path   string   `json:"path,omitempty"`
	Ignore []string `json:"ignore,omitempty"`
}

// ls tool constants
const (
	maxLsResults       = 200
	maxLsScanEntries   = 512
	lsReadDirBatchSize = 128
)

// lsTool returns the ls tool implementation.
func lsTool() tools.Tool {
	return tools.Func[lsArgs]("ls", "List files and directories in a given path", func(ctx context.Context, a lsArgs, deps *tools.Container) (any, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return "", err
		}
		p := strings.TrimSpace(a.Path)
		if p == "" {
			p = "."
		}
		accessPath, err := s.ResolveAccessPath(p)
		if err != nil {
			return "", fmt.Errorf("Security error: %w", err)
		}
		f, st, _, err := s.OpenReadAccessPath(accessPath)
		if err != nil {
			return "", fmt.Errorf("stat %s: %w", accessPath.Abs(), err)
		}
		defer f.Close()
		if !st.IsDir() {
			return "", fmt.Errorf("Path is not a directory: %s", p)
		}
		ignore := make([]string, 0, len(a.Ignore))
		for _, ig := range a.Ignore {
			pat, err := validateGlobArgPattern("ls", "ignore", ig)
			if err != nil {
				return "", err
			}
			if pat != "" {
				ignore = append(ignore, pat)
			}
		}
		matchIgnore := func(name string) bool {
			if len(ignore) == 0 {
				return false
			}
			for _, pat := range ignore {
				if doublestar.MatchUnvalidated(pat, name) {
					return true
				}
			}
			return false
		}
		items := make([]string, 0, maxLsResults)
		scannedEntries := 0
		scanTruncated := false
		truncatedReason := ""

	scanLoop:
		for {
			if scannedEntries >= maxLsScanEntries {
				scanTruncated = true
				truncatedReason = "scan_cap"
				break
			}

			batchSize := lsReadDirBatchSize
			remaining := maxLsScanEntries - scannedEntries
			if remaining < batchSize {
				batchSize = remaining
			}
			if batchSize <= 0 {
				scanTruncated = true
				truncatedReason = "scan_cap"
				break
			}

			ents, readErr := f.ReadDir(batchSize)
			if readErr != nil && !errors.Is(readErr, io.EOF) {
				return "", fmt.Errorf("read dir %s: %w", accessPath.Abs(), readErr)
			}
			for _, e := range ents {
				scannedEntries++
				name := e.Name()
				if matchIgnore(name) {
					continue
				}
				if e.IsDir() {
					name += "/"
				}
				items = append(items, name)
				if len(items) >= maxLsResults {
					scanTruncated = true
					truncatedReason = "result_cap"
					break scanLoop
				}
			}
			if errors.Is(readErr, io.EOF) {
				break
			}
			if len(ents) == 0 {
				break
			}
		}

		sort.Strings(items)

		meta := map[string]any{
			"count": len(items),
		}
		warning := ""
		if scanTruncated {
			meta["warning_kind"] = scanCapWarningKind
			meta["scan_truncated"] = true
			meta["scan_cap"] = maxLsScanEntries
			meta["result_cap"] = maxLsResults
			meta["scanned_entries"] = scannedEntries
			meta["skipped_due_to_cap"] = true
			meta["truncated_reason"] = truncatedReason
			warning = lsScanCapWarning(scannedEntries, maxLsScanEntries, maxLsResults, truncatedReason)
		}
		tools.UpsertToolResultMetadata(ctx, meta)

		if len(items) == 0 {
			return appendGlobWarning("(empty)", warning), nil
		}
		return appendGlobWarning(strings.Join(items, "\n"), warning), nil
	})
}

// lsScanCapWarning returns a warning message when ls scan cap is reached.
func lsScanCapWarning(scannedEntries, scanCap, resultCap int, reason string) string {
	switch reason {
	case "result_cap":
		return formatWarningDiagnostic(
			fmt.Sprintf("ls output stopped after collecting %d item(s) at result cap %d", resultCap, resultCap),
			"Narrow the path or adjust ignore patterns, then rerun ls to inspect remaining entries.",
		)
	default:
		return formatWarningDiagnostic(
			fmt.Sprintf("ls scan stopped after %d directory entries at scan cap %d", scannedEntries, scanCap),
			"Narrow the path or adjust ignore patterns, then rerun ls to inspect remaining entries.",
		)
	}
}
