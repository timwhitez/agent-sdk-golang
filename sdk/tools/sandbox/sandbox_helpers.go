package sandbox

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/bmatcuk/doublestar/v4"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// Constants for truncation and confirmation
const (
	maxConfirmDiffChars = 120_000
	maxConfirmRawChars  = 40_000
	maxConfirmDiffLines = 2_000
)

// ============================================================================
// Path utilities
// ============================================================================

// pathsEqual compares two paths for equality, handling case-insensitivity on Windows.
func pathsEqual(a, b string) bool {
	a = filepath.Clean(a)
	b = filepath.Clean(b)
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}

// evalSymlinksForPath evaluates symlinks for a path, handling non-existent ancestors.
func evalSymlinksForPath(abs string) (string, error) {
	resolved, err := filepath.EvalSymlinks(abs)
	if err == nil {
		return resolved, nil
	}
	if !os.IsNotExist(err) {
		return "", err
	}
	parent := abs
	for {
		_, statErr := os.Lstat(parent)
		if statErr == nil {
			break
		}
		if !os.IsNotExist(statErr) {
			return "", statErr
		}
		dir := filepath.Dir(parent)
		if dir == parent {
			return "", statErr
		}
		parent = dir
	}
	resolvedParent, err := filepath.EvalSymlinks(parent)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(parent, abs)
	if err != nil {
		return "", err
	}
	if rel == "." {
		return resolvedParent, nil
	}
	return filepath.Join(resolvedParent, rel), nil
}

// isWithinRoot checks if a path is within a root directory.
func isWithinRoot(path, root string) bool {
	path = filepath.Clean(path)
	root = filepath.Clean(root)
	if path == "" || root == "" {
		return false
	}
	if path == root {
		return true
	}
	sep := string(os.PathSeparator)
	if root == sep {
		return strings.HasPrefix(path, root)
	}
	return strings.HasPrefix(path, root+sep)
}

// isSymlinkSecurityError checks if an error is a symlink-related security error.
func isSymlinkSecurityError(err error) bool {
	var secErr *SecurityError
	if !errors.As(err, &secErr) {
		return false
	}
	return strings.Contains(strings.ToLower(secErr.Error()), "symlink")
}

// ============================================================================
// Truncation utilities
// ============================================================================

// truncate truncates a string to a maximum length, adding "..." if truncated.
func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	if max <= 3 {
		return s[:max]
	}
	return s[:max-3] + "..."
}

// truncateLine truncates a string to a maximum length for a single line.
// Returns the truncated string and a boolean indicating if truncation occurred.
func truncateLine(s string, max int) (string, bool) {
	if max <= 0 || len(s) <= max {
		return s, false
	}
	if max <= 3 {
		return s[:max], true
	}
	return s[:max-3] + "...", true
}

// truncateForMeta truncates a string for metadata fields.
func truncateForMeta(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	if max <= 3 {
		return s[:max]
	}
	return s[:max-3] + "..."
}

// truncateOneLine collapses multiline text to a single line and truncates.
func truncateOneLine(s string, max int) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	return truncateForMeta(s, max)
}

// ============================================================================
// Diagnostic formatting
// ============================================================================

// formatErrorDiagnostic formats an error diagnostic message.
func formatErrorDiagnostic(summary, action string) string {
	summary = strings.TrimSpace(summary)
	if summary == "" {
		summary = "operation failed"
	}
	action = strings.TrimSpace(action)
	if action == "" {
		action = "Review the diagnostic details and retry."
	}
	return fmt.Sprintf("[ERROR] %s - %s", summary, action)
}

// formatErrorDiagnosticFromErr formats an error diagnostic from an error.
func formatErrorDiagnosticFromErr(summary string, err error, action string) string {
	summary = strings.TrimSpace(summary)
	if err != nil {
		detail := strings.TrimSpace(err.Error())
		if detail != "" {
			if summary == "" {
				summary = detail
			} else {
				summary = fmt.Sprintf("%s: %s", summary, detail)
			}
		}
	}
	return formatErrorDiagnostic(summary, action)
}

// formatWarningDiagnostic formats a warning diagnostic message.
func formatWarningDiagnostic(summary, action string) string {
	summary = strings.TrimSpace(summary)
	if summary == "" {
		summary = "operation warning"
	}
	action = strings.TrimSpace(action)
	if action == "" {
		action = "Review the warning details and retry if needed."
	}
	return fmt.Sprintf("[WARN] %s - %s", summary, action)
}

// isSeverityActionDiagnostic checks if text is a severity-action diagnostic.
func isSeverityActionDiagnostic(text string) bool {
	text = strings.TrimSpace(text)
	if !strings.HasPrefix(text, "[") {
		return false
	}
	end := strings.Index(text, "]")
	if end <= 1 {
		return false
	}
	severity := strings.ToUpper(strings.TrimSpace(text[1:end]))
	switch severity {
	case "INFO", "WARN", "ERROR":
	default:
		return false
	}
	return strings.Contains(strings.TrimSpace(text[end+1:]), " - ")
}

// ============================================================================
// Confirmation and metadata
// ============================================================================

// attachToolCallMeta attaches tool call metadata to a metadata map.
func attachToolCallMeta(ctx context.Context, meta map[string]any) map[string]any {
	if meta == nil {
		meta = map[string]any{}
	}
	if id := tools.ToolCallID(ctx); id != "" {
		meta["tool_call_id"] = id
	}
	return meta
}

// buildConfirmDetail builds a confirmation detail string from metadata.
func buildConfirmDetail(meta map[string]any) string {
	if meta == nil {
		return ""
	}
	if v, ok := meta["diff"].(string); ok {
		meta["diff"] = truncateForMeta(v, maxConfirmDiffChars)
	}
	if v, ok := meta["raw"].(string); ok {
		meta["raw"] = truncateForMeta(v, maxConfirmRawChars)
	}
	b, err := json.Marshal(meta)
	if err != nil {
		// Fallback: keep confirm usable even if meta is malformed.
		if s, ok := meta["summary"].(string); ok {
			return s
		}
		return "(confirm)"
	}
	return string(b)
}

// ============================================================================
// Tool denial
// ============================================================================

// denyToolResult returns a denied tool result with metadata.
func denyToolResult(ctx context.Context, toolName, reason string) (llm.Content, error) {
	toolLabel := strings.TrimSpace(toolName)
	if toolLabel == "" {
		toolLabel = "tool"
	}
	deniedReason := strings.TrimSpace(reason)
	if deniedReason == "" {
		deniedReason = "request denied by confirmation policy"
	}
	tools.UpsertToolResultMetadata(ctx, map[string]any{
		"error_kind":    "denied",
		"denied_tool":   toolLabel,
		"denied_reason": deniedReason,
	})
	msg := formatErrorDiagnostic(
		fmt.Sprintf("%s request denied: %s", toolLabel, deniedReason),
		"Adjust permission/confirmation settings and retry.",
	)
	return llm.TextContent(msg), deniedToolError(toolLabel, deniedReason)
}

// toolDeniedError is an error type for denied tool actions.
type toolDeniedError struct {
	tool   string
	reason string
}

func (e *toolDeniedError) Error() string {
	tool := strings.TrimSpace(e.tool)
	if tool == "" {
		tool = "tool"
	}
	reason := strings.TrimSpace(e.reason)
	if reason == "" {
		reason = "request denied by confirmation policy"
	}
	return fmt.Sprintf("%s request denied: %s", tool, reason)
}

func (e *toolDeniedError) Unwrap() error { return ErrToolDenied }

// deniedToolError creates a new tool denied error.
func deniedToolError(toolName, reason string) error {
	return &toolDeniedError{tool: strings.TrimSpace(toolName), reason: strings.TrimSpace(reason)}
}

// ============================================================================
// Tool wrapper
// ============================================================================

// toolWithArgs creates a Tool from a function with typed arguments.
func toolWithArgs[Args any](name, description string, fn func(ctx context.Context, args Args, deps *tools.Container) (llm.Content, error)) tools.Tool {
	schema := tools.SchemaFor[Args]()
	return tools.Tool{
		Name:        name,
		Description: description,
		Schema:      schema,
		Handler: func(ctx context.Context, raw json.RawMessage, deps *tools.Container) (llm.Content, error) {
			var a Args
			dec := json.NewDecoder(bytes.NewReader(raw))
			dec.DisallowUnknownFields()
			if err := dec.Decode(&a); err != nil {
				msg := formatErrorDiagnosticFromErr("tool arguments are invalid", err, "Fix tool arguments and retry.")
				return llm.TextContent(msg), err
			}
			return fn(ctx, a, deps)
		},
	}
}

// ============================================================================
// Diff utilities
// ============================================================================

// splitLines splits a string into lines, normalizing line endings.
func splitLines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	if s == "" {
		return []string{}
	}
	return strings.Split(s, "\n")
}

// fullReplaceDiff returns a unified-diff-like preview by treating the change as a full-file replacement.
// It's meant for human preview in interactive clients (not for applying).
func fullReplaceDiff(filePath, oldContent, newContent string) string {
	oldLines := splitLines(oldContent)
	newLines := splitLines(newContent)
	oldN := len(oldLines)
	newN := len(newLines)

	var b strings.Builder
	b.WriteString("--- a/")
	b.WriteString(strings.TrimSpace(filePath))
	b.WriteString("\n+++ b/")
	b.WriteString(strings.TrimSpace(filePath))
	b.WriteString("\n")

	// Hunk header.
	if oldN == 0 && newN == 0 {
		b.WriteString("@@ -0,0 +0,0 @@\n")
		return b.String()
	}

	oldStart := 1
	newStart := 1
	if oldN == 0 {
		oldStart = 0
	}
	if newN == 0 {
		newStart = 0
	}
	b.WriteString(fmt.Sprintf("@@ -%d,%d +%d,%d @@\n", oldStart, oldN, newStart, newN))

	// Emit lines (bounded).
	lineBudget := maxConfirmDiffLines
	for _, l := range oldLines {
		if lineBudget <= 0 {
			b.WriteString("... (diff truncated)\n")
			return b.String()
		}
		b.WriteString("-")
		b.WriteString(l)
		b.WriteString("\n")
		lineBudget--
	}
	for _, l := range newLines {
		if lineBudget <= 0 {
			b.WriteString("... (diff truncated)\n")
			return b.String()
		}
		b.WriteString("+")
		b.WriteString(l)
		b.WriteString("\n")
		lineBudget--
	}
	return b.String()
}

// ============================================================================
// File reading utilities
// ============================================================================

var (
	sandboxReadAll          = io.ReadAll
	errFileReadLimitReached = errors.New("file too large")
)

// readAllBounded reads from r with a byte limit.
func readAllBounded(r io.Reader, maxBytes int64) ([]byte, error) {
	if maxBytes <= 0 {
		return sandboxReadAll(r)
	}
	limited := &io.LimitedReader{R: r, N: maxBytes + 1}
	b, err := sandboxReadAll(limited)
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > maxBytes {
		return nil, errFileReadLimitReached
	}
	return b, nil
}

// readPreviewBounded reads a preview from r with a byte limit.
func readPreviewBounded(r io.Reader, maxBytes int64) ([]byte, bool, error) {
	if maxBytes <= 0 {
		b, err := sandboxReadAll(r)
		return b, false, err
	}
	limited := &io.LimitedReader{R: r, N: maxBytes + 1}
	b, err := sandboxReadAll(limited)
	if err != nil {
		return nil, false, err
	}
	truncated := int64(len(b)) > maxBytes
	if truncated {
		b = b[:maxBytes]
	}
	return b, truncated, nil
}

// ============================================================================
// Glob pattern validation (shared by ls, glob, grep)
// ============================================================================

// validateGlobArgPattern validates a glob argument pattern for a tool.
// Returns the normalized pattern (with forward slashes) or an error.
// Empty patterns are allowed and return an empty string.
func validateGlobArgPattern(toolName, argName, raw string) (string, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return "", nil
	}
	normalized := filepath.ToSlash(trimmed)
	if !doublestar.ValidatePattern(normalized) {
		return "", fmt.Errorf("invalid %s pattern for %s tool: %q", argName, toolName, trimmed)
	}
	return normalized, nil
}
