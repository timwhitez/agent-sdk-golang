package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type writeArgs struct {
	FilePath string `json:"file_path"`
	Content  string `json:"content"`
}

func writeTool() tools.Tool {
	return toolWithArgs[writeArgs]("write", "Write content to a file", func(ctx context.Context, a writeArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		conf := getConfirmer(deps, ctx)
		p, err := s.resolveForAccess(a.FilePath)
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
			return llm.TextContent(msg), err
		}
		// Build confirm meta with a diff preview.
		raw := fmt.Sprintf("%s (%d bytes)", a.FilePath, len(a.Content))
		oldContent := ""
		diffTruncated := false
		if b, st, _, truncated, err := s.readPathPreviewBounded(p, maxWriteDiffBytes); err == nil {
			if !st.IsDir() {
				oldContent = string(b)
				diffTruncated = truncated
			}
		} else if !os.IsNotExist(err) {
			var secErr *SecurityError
			if errors.As(err, &secErr) {
				msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
				return llm.TextContent(msg), err
			}
			msg := formatErrorDiagnosticFromErr("Unable to preview existing file before write", err, "Check file permissions/path and retry.")
			return llm.TextContent(msg), err
		}
		diff := fullReplaceDiff(a.FilePath, oldContent, a.Content)
		if diffTruncated {
			diff += fmt.Sprintf("... (existing file preview truncated after %d bytes)\n", maxWriteDiffBytes)
		}
		diffMetaTruncated := diffTruncated || len(diff) > maxConfirmDiffChars
		meta := attachToolCallMeta(ctx, map[string]any{
			"category":         "filesystem_write",
			"summary":          raw,
			"file_path":        strings.TrimSpace(a.FilePath),
			"diff":             diff,
			"diff_truncated":   diffMetaTruncated,
			"diff_bytes_limit": maxWriteDiffBytes,
			"raw":              raw,
		})
		ok, err := conf.Confirm(ctx, "write", buildConfirmDetail(meta))
		if err != nil {
			msg := formatErrorDiagnosticFromErr("write confirmation failed", err, "Retry after confirmation policy is available.")
			return llm.TextContent(msg), err
		}
		if !ok {
			return denyToolResult(ctx, "write", "user denied request")
		}
		resolvedPath, err := s.revalidatePathForAccess(p)
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
			return llm.TextContent(msg), err
		}
		if err := writeFilePreserveMode(resolvedPath, []byte(a.Content), 0o644); err != nil {
			msg := formatErrorDiagnosticFromErr("Unable to write file", err, "Check file permissions/path and retry.")
			return llm.TextContent(msg), err
		}
		return llm.TextContent(fmt.Sprintf("Wrote %d bytes to %s", len(a.Content), a.FilePath)), nil
	})
}

type editArgs struct {
	FilePath   string `json:"file_path"`
	OldString  string `json:"old_string"`
	NewString  string `json:"new_string"`
	ReplaceAll bool   `json:"replace_all,omitempty"`
}

func editTool() tools.Tool {
	return toolWithArgs[editArgs]("edit", "Replace text in a file", func(ctx context.Context, a editArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		conf := getConfirmer(deps, ctx)
		p, err := s.resolveForAccess(a.FilePath)
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
			return llm.TextContent(msg), err
		}
		b, st, _, err := s.readAllPathBounded(p, maxEditFileBytes)
		if err != nil {
			if errors.Is(err, errFileReadLimitReached) {
				size := maxEditFileBytes + 1
				if st != nil && st.Size() > 0 {
					size = st.Size()
				}
				msg := fmt.Sprintf("[ERROR] edit refuses to load %s (%d bytes) - max %d bytes; edit externally or split the file", a.FilePath, size, maxEditFileBytes)
				return llm.TextContent(msg), fmt.Errorf("file too large")
			}
			var secErr *SecurityError
			if errors.As(err, &secErr) {
				msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
				return llm.TextContent(msg), err
			}
			if os.IsNotExist(err) {
				msg := formatErrorDiagnostic(fmt.Sprintf("File not found: %s", a.FilePath), "Verify the path exists (use ls/glob) and retry.")
				return llm.TextContent(msg), err
			}
			msg := formatErrorDiagnosticFromErr("Unable to read file for edit", err, "Check file permissions/path and retry.")
			return llm.TextContent(msg), err
		}
		if st.IsDir() {
			err := fmt.Errorf("is a directory")
			msg := formatErrorDiagnostic(fmt.Sprintf("Path is a directory: %s", a.FilePath), "Provide a file path (not a directory) and retry.")
			return llm.TextContent(msg), err
		}
		content := string(b)
		if !strings.Contains(content, a.OldString) {
			err := fmt.Errorf("string not found")
			msg := formatErrorDiagnostic(fmt.Sprintf("String not found in %s", a.FilePath), "Check old_string matches file content and retry.")
			return llm.TextContent(msg), err
		}
		count := strings.Count(content, a.OldString)
		if count != 1 && !a.ReplaceAll {
			msg := fmt.Sprintf("[ERROR] edit expects a single match in %s - found %d occurrences; make old_string unique or set replace_all=true", a.FilePath, count)
			return llm.TextContent(msg), fmt.Errorf("edit requires unique match")
		}
		newContent := strings.Replace(content, a.OldString, a.NewString, count)
		raw := fmt.Sprintf("%s (replace %d occurrence(s))", a.FilePath, count)
		meta := attachToolCallMeta(ctx, map[string]any{
			"category":  "filesystem_write",
			"summary":   raw,
			"file_path": strings.TrimSpace(a.FilePath),
			"diff":      fullReplaceDiff(a.FilePath, content, newContent),
			"raw":       raw + "\nold_string: " + truncateForMeta(a.OldString, 600) + "\nnew_string: " + truncateForMeta(a.NewString, 600),
		})
		ok, err := conf.Confirm(ctx, "edit", buildConfirmDetail(meta))
		if err != nil {
			msg := formatErrorDiagnosticFromErr("edit confirmation failed", err, "Retry after confirmation policy is available.")
			return llm.TextContent(msg), err
		}
		if !ok {
			return denyToolResult(ctx, "edit", "user denied request")
		}
		resolvedPath, err := s.revalidatePathForAccess(p)
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
			return llm.TextContent(msg), err
		}
		if err := writeFilePreserveMode(resolvedPath, []byte(newContent), 0o644); err != nil {
			msg := formatErrorDiagnosticFromErr("Unable to write edited file", err, "Check file permissions/path and retry.")
			return llm.TextContent(msg), err
		}
		return llm.TextContent(fmt.Sprintf("Replaced %d occurrence(s) in %s", count, a.FilePath)), nil
	})
}

type multiEditItem struct {
	OldString  string `json:"old_string"`
	NewString  string `json:"new_string"`
	ReplaceAll bool   `json:"replace_all,omitempty"`
}

type multieditArgs struct {
	FilePath string          `json:"file_path"`
	Edits    []multiEditItem `json:"edits"`
}

func multieditTool() tools.Tool {
	return toolWithArgs[multieditArgs]("multiedit", "Apply multiple text replacements to a file (in order)", func(ctx context.Context, a multieditArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		conf := getConfirmer(deps, ctx)
		p, err := s.resolveForAccess(a.FilePath)
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
			return llm.TextContent(msg), err
		}
		if strings.TrimSpace(a.FilePath) == "" {
			err := fmt.Errorf("missing file_path")
			msg := formatErrorDiagnostic("multiedit requires file_path", "Provide a non-empty file_path and retry.")
			return llm.TextContent(msg), err
		}
		if len(a.Edits) == 0 {
			err := fmt.Errorf("empty edits")
			msg := formatErrorDiagnostic("multiedit requires at least one edit", "Provide a non-empty edits array and retry.")
			return llm.TextContent(msg), err
		}
		b, st, _, err := s.readAllPathBounded(p, maxEditFileBytes)
		if err != nil {
			if errors.Is(err, errFileReadLimitReached) {
				size := maxEditFileBytes + 1
				if st != nil && st.Size() > 0 {
					size = st.Size()
				}
				msg := fmt.Sprintf("[ERROR] multiedit refuses to load %s (%d bytes) - max %d bytes; edit externally or split the file", a.FilePath, size, maxEditFileBytes)
				return llm.TextContent(msg), fmt.Errorf("file too large")
			}
			var secErr *SecurityError
			if errors.As(err, &secErr) {
				msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
				return llm.TextContent(msg), err
			}
			if os.IsNotExist(err) {
				msg := formatErrorDiagnostic(fmt.Sprintf("File not found: %s", a.FilePath), "Verify the path exists (use ls/glob) and retry.")
				return llm.TextContent(msg), err
			}
			msg := formatErrorDiagnosticFromErr("Unable to read file for multiedit", err, "Check file permissions/path and retry.")
			return llm.TextContent(msg), err
		}
		if st.IsDir() {
			err := fmt.Errorf("is a directory")
			msg := formatErrorDiagnostic(fmt.Sprintf("Path is a directory: %s", a.FilePath), "Provide a file path (not a directory) and retry.")
			return llm.TextContent(msg), err
		}
		orig := string(b)
		content := orig
		counts := make([]int, 0, len(a.Edits))
		for i, e := range a.Edits {
			if e.OldString == "" {
				err := fmt.Errorf("empty old_string")
				msg := formatErrorDiagnostic(fmt.Sprintf("multiedit edits[%d] requires old_string", i), "Provide a non-empty edits[].old_string and retry.")
				return llm.TextContent(msg), err
			}
			if !strings.Contains(content, e.OldString) {
				err := fmt.Errorf("string not found")
				msg := formatErrorDiagnostic(fmt.Sprintf("String not found for edits[%d] in %s", i, a.FilePath), "Check edits[].old_string matches file content and retry.")
				return llm.TextContent(msg), err
			}
			c := strings.Count(content, e.OldString)
			if c != 1 && !e.ReplaceAll {
				msg := fmt.Sprintf("[ERROR] multiedit edits[%d] expects a single match in %s - found %d occurrences; make old_string unique or set replace_all=true", i, a.FilePath, c)
				return llm.TextContent(msg), fmt.Errorf("multiedit requires unique match")
			}
			replaced := 1
			if e.ReplaceAll {
				replaced = c
			}
			counts = append(counts, replaced)
			content = strings.Replace(content, e.OldString, e.NewString, replaced)
		}

		summary := fmt.Sprintf("%s (multiedit %d step(s))", a.FilePath, len(a.Edits))
		rawLines := []string{summary}
		for i, c := range counts {
			rawLines = append(rawLines, fmt.Sprintf("- step %d: replace %d occurrence(s)", i+1, c))
		}
		meta := attachToolCallMeta(ctx, map[string]any{
			"category":  "filesystem_write",
			"summary":   summary,
			"file_path": strings.TrimSpace(a.FilePath),
			"diff":      fullReplaceDiff(a.FilePath, orig, content),
			"raw":       strings.Join(rawLines, "\n"),
		})
		ok, err := conf.Confirm(ctx, "multiedit", buildConfirmDetail(meta))
		if err != nil {
			msg := formatErrorDiagnosticFromErr("multiedit confirmation failed", err, "Retry after confirmation policy is available.")
			return llm.TextContent(msg), err
		}
		if !ok {
			return denyToolResult(ctx, "multiedit", "user denied request")
		}
		resolvedPath, err := s.revalidatePathForAccess(p)
		if err != nil {
			msg := formatErrorDiagnosticFromErr("Security error", err, "Use a file path inside the sandbox root and retry.")
			return llm.TextContent(msg), err
		}
		if err := writeFilePreserveMode(resolvedPath, []byte(content), 0o644); err != nil {
			msg := formatErrorDiagnosticFromErr("Unable to write multiedit result", err, "Check file permissions/path and retry.")
			return llm.TextContent(msg), err
		}
		return llm.TextContent(fmt.Sprintf("Updated %s with %d edit step(s)", a.FilePath, len(a.Edits))), nil
	})
}
