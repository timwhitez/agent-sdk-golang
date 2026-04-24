package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// applyPatchArgs holds the patch content.
type applyPatchArgs struct {
	Patch string `json:"patch"`
}

// applyPatchTool returns a tool that applies unified-diff-style patches.
func applyPatchTool() tools.Tool {
	return toolWithArgs[applyPatchArgs]("apply_patch", "Apply a patch in apply_patch format (add/update/delete files)", func(ctx context.Context, a applyPatchArgs, deps *tools.Container) (llm.Content, error) {
		s, err := tools.Get(deps, ctx, Key)
		if err != nil {
			return llm.TextContent(""), err
		}
		conf := getConfirmer(deps, ctx)
		paths := []string{}
		if ops, err := parsePatchOpsForPreview(a.Patch); err == nil {
			seen := map[string]struct{}{}
			for _, op := range ops {
				p := strings.TrimSpace(op.path)
				if p != "" {
					if _, ok := seen[p]; !ok {
						seen[p] = struct{}{}
						paths = append(paths, p)
					}
				}
				mt := strings.TrimSpace(op.moveTo)
				if mt != "" {
					if _, ok := seen[mt]; !ok {
						seen[mt] = struct{}{}
						paths = append(paths, mt)
					}
				}
			}
			sort.Strings(paths)
		}
		raw := fmt.Sprintf("apply_patch (%d bytes)", len(a.Patch))
		meta := attachToolCallMeta(ctx, map[string]any{
			"category": "filesystem_write",
			"summary":  raw,
			"paths":    paths,
			"diff":     a.Patch,
			"raw":      raw,
		})
		ok, err := conf.Confirm(ctx, "apply_patch", buildConfirmDetail(meta))
		if err != nil {
			msg := formatErrorDiagnosticFromErr("apply_patch confirmation failed", err, "Retry after confirmation policy is available.")
			return llm.TextContent(msg), err
		}
		if !ok {
			return denyToolResult(ctx, "apply_patch", "user denied request")
		}
		res, err := applyPatchToSandbox(s, a.Patch)
		if err != nil {
			msg := strings.TrimSpace(err.Error())
			if !isSeverityActionDiagnostic(msg) {
				msg = formatErrorDiagnosticFromErr("apply_patch failed", err, "Fix patch content and retry.")
			}
			return llm.TextContent(msg), err
		}
		return llm.TextContent(res), nil
	})
}

// patchOp represents a single patch operation (add/update/delete/move).
type patchOp struct {
	kind   string // add|update|delete
	path   string
	moveTo string
	// For add: lines are content lines without prefix.
	addLines []string
	// For update: list of hunks, each with raw prefixed lines.
	hunks []patchHunk
}

// patchHunk represents a single diff hunk with line numbers.
type patchHunk struct {
	lines          []string
	oldStart       int
	oldLen         int
	newStart       int
	newLen         int
	hasLineNumbers bool
}

// parsePatchOpsForPreview parses patch operations for preview during confirmation.
func parsePatchOpsForPreview(patch string) ([]patchOp, error) {
	norm := strings.ReplaceAll(patch, "\r\n", "\n")
	norm = strings.ReplaceAll(norm, "\r", "\n")
	lines := strings.Split(norm, "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "*** Begin Patch" {
		return nil, fmt.Errorf("patch must start with '*** Begin Patch'")
	}
	if strings.TrimSpace(lines[len(lines)-1]) != "*** End Patch" {
		return nil, fmt.Errorf("patch must end with '*** End Patch'")
	}
	inner := lines[1 : len(lines)-1]
	return parsePatchOps(inner)
}

// applyPatchToSandbox applies a patch to the sandbox filesystem.
func applyPatchToSandbox(s *Sandbox, patch string) (string, error) {
	if s == nil {
		return "", fmt.Errorf("nil sandbox")
	}
	norm := strings.ReplaceAll(patch, "\r\n", "\n")
	norm = strings.ReplaceAll(norm, "\r", "\n")
	lines := strings.Split(norm, "\n")
	// tolerate trailing newline
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "*** Begin Patch" {
		return "", fmt.Errorf("patch must start with '*** Begin Patch'")
	}
	if strings.TrimSpace(lines[len(lines)-1]) != "*** End Patch" {
		return "", fmt.Errorf("patch must end with '*** End Patch'")
	}
	lines = lines[1 : len(lines)-1]
	ops, err := parsePatchOps(lines)
	if err != nil {
		return "", err
	}
	if len(ops) == 0 {
		return "", fmt.Errorf("patch contains no operations; include at least one *** Add File, *** Update File, or *** Delete File section")
	}

	changed := 0
	for _, op := range ops {
		switch op.kind {
		case "add":
			if err := applyAddFile(s, op.path, op.addLines); err != nil {
				return "", err
			}
			changed++
		case "delete":
			if err := applyDeleteFile(s, op.path); err != nil {
				return "", err
			}
			changed++
		case "update":
			if err := applyUpdateFile(s, op.path, op.hunks); err != nil {
				return "", err
			}
			if strings.TrimSpace(op.moveTo) != "" {
				if err := applyMoveFile(s, op.path, op.moveTo); err != nil {
					return "", err
				}
			}
			changed++
		default:
			return "", fmt.Errorf("unknown patch op: %s", op.kind)
		}
	}
	return fmt.Sprintf("Applied patch: %d file(s) updated", changed), nil
}

// parsePatchOps parses patch operations from normalized lines.
func parsePatchOps(lines []string) ([]patchOp, error) {
	ops := []patchOp{}
	for i := 0; i < len(lines); {
		rawLine := lines[i]
		line := strings.TrimSpace(rawLine)
		if line == "" {
			i++
			continue
		}
		switch {
		case strings.HasPrefix(line, "*** Add File: "):
			p := strings.TrimSpace(strings.TrimPrefix(line, "*** Add File: "))
			if p == "" {
				return nil, fmt.Errorf("missing add file path")
			}
			i++
			addLines := []string{}
			for i < len(lines) {
				l := lines[i]
				if strings.HasPrefix(strings.TrimSpace(l), "*** ") {
					break
				}
				if !strings.HasPrefix(l, "+") {
					return nil, fmt.Errorf("add file content must start with '+': %q", l)
				}
				addLines = append(addLines, strings.TrimPrefix(l, "+"))
				i++
			}
			ops = append(ops, patchOp{kind: "add", path: p, addLines: addLines})
		case strings.HasPrefix(line, "*** Delete File: "):
			p := strings.TrimSpace(strings.TrimPrefix(line, "*** Delete File: "))
			if p == "" {
				return nil, fmt.Errorf("missing delete file path")
			}
			i++
			ops = append(ops, patchOp{kind: "delete", path: p})
		case strings.HasPrefix(line, "*** Update File: "):
			p := strings.TrimSpace(strings.TrimPrefix(line, "*** Update File: "))
			if p == "" {
				return nil, fmt.Errorf("missing update file path")
			}
			i++
			moveTo := ""
			if i < len(lines) {
				l2 := strings.TrimSpace(lines[i])
				if strings.HasPrefix(l2, "*** Move to: ") {
					moveTo = strings.TrimSpace(strings.TrimPrefix(l2, "*** Move to: "))
					i++
				}
			}
			hunks := []patchHunk{}
			for i < len(lines) {
				l := lines[i]
				lt := strings.TrimSpace(l)
				if strings.HasPrefix(lt, "*** ") {
					break
				}
				if lt == "" {
					i++
					continue
				}
				if strings.HasPrefix(lt, "@@") {
					oldStart, oldLen, newStart, newLen, ok := parseHunkHeader(lt)
					if !ok && lt != "@@" {
						return nil, fmt.Errorf("invalid hunk header in update %q: %q", p, l)
					}
					i++
					h := []string{}
					for i < len(lines) {
						ll := lines[i]
						llt := strings.TrimSpace(ll)
						if strings.HasPrefix(llt, "@@") || strings.HasPrefix(llt, "*** ") {
							break
						}
						if ll == "*** End of File" {
							i++
							break
						}
						if ll == "" {
							// Whitespace-stripping editors can turn a blank context line " " into "".
							ll = " "
						}
						pref := ll[0]
						if pref != ' ' && pref != '-' && pref != '+' {
							return nil, fmt.Errorf("invalid hunk line prefix: %q", ll)
						}
						h = append(h, ll)
						i++
					}
					if len(h) == 0 {
						return nil, fmt.Errorf("empty hunk in update %q", p)
					}
					hunks = append(hunks, patchHunk{
						lines:          h,
						oldStart:       oldStart,
						oldLen:         oldLen,
						newStart:       newStart,
						newLen:         newLen,
						hasLineNumbers: ok,
					})
					continue
				}
				return nil, fmt.Errorf("unexpected line in update %q: %q", p, l)
			}
			if len(hunks) == 0 {
				return nil, fmt.Errorf("update file %q must include at least one hunk", p)
			}
			ops = append(ops, patchOp{kind: "update", path: p, moveTo: moveTo, hunks: hunks})
		default:
			return nil, fmt.Errorf("unexpected patch content: %q", rawLine)
		}
	}
	if len(ops) == 0 {
		return nil, fmt.Errorf("patch contains no operations; include at least one *** Add File, *** Update File, or *** Delete File section")
	}
	return ops, nil
}

// parseHunkHeader parses a unified diff hunk header (@@ -old,old +new,new @@).
func parseHunkHeader(line string) (int, int, int, int, bool) {
	if !strings.HasPrefix(line, "@@") {
		return 0, 0, 0, 0, false
	}
	rest := line[2:]
	end := strings.Index(rest, "@@")
	if end == -1 {
		return 0, 0, 0, 0, false
	}
	header := strings.TrimSpace(rest[:end])
	parts := strings.Fields(header)
	if len(parts) < 2 {
		return 0, 0, 0, 0, false
	}
	oldStart, oldLen, ok := parseHunkRange(parts[0], '-')
	if !ok {
		return 0, 0, 0, 0, false
	}
	newStart, newLen, ok := parseHunkRange(parts[1], '+')
	if !ok {
		return 0, 0, 0, 0, false
	}
	return oldStart, oldLen, newStart, newLen, true
}

// parseHunkRange parses a single hunk range (e.g., "-1,10" or "+5").
func parseHunkRange(tok string, prefix byte) (int, int, bool) {
	if tok == "" || tok[0] != prefix {
		return 0, 0, false
	}
	tok = tok[1:]
	parts := strings.SplitN(tok, ",", 2)
	start, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, false
	}
	length := 1
	if len(parts) == 2 {
		length, err = strconv.Atoi(parts[1])
		if err != nil {
			return 0, 0, false
		}
	}
	return start, length, true
}

// applyAddFile creates a new file with the given content.
func applyAddFile(s *Sandbox, relPath string, lines []string) error {
	p, err := s.resolveForAccess(relPath)
	if err != nil {
		return err
	}
	content := strings.Join(lines, "\n")
	if len(lines) > 0 {
		content += "\n"
	}
	resolvedPath, err := s.revalidatePathForAccess(p)
	if err != nil {
		return err
	}
	if err := writeFilePreserveMode(resolvedPath, []byte(content), 0o644); err != nil {
		return err
	}
	return nil
}

// applyDeleteFile removes a file from the sandbox.
func applyDeleteFile(s *Sandbox, relPath string) error {
	p, err := s.resolveForAccess(relPath)
	if err != nil {
		return err
	}
	resolvedPath, err := s.revalidatePathForAccess(p)
	if err != nil {
		return err
	}
	if err := os.Remove(resolvedPath); err != nil {
		return err
	}
	return nil
}

// applyMoveFile renames a file within the sandbox.
func applyMoveFile(s *Sandbox, fromRel, toRel string) error {
	fromAbs, err := s.resolveForAccess(fromRel)
	if err != nil {
		return err
	}
	toAbs, err := s.resolveForAccess(toRel)
	if err != nil {
		return err
	}
	fromPath, err := s.revalidatePathForAccess(fromAbs)
	if err != nil {
		return err
	}
	toPath, err := s.revalidatePathForAccess(toAbs)
	if err != nil {
		return err
	}
	toDir := filepath.Dir(toPath)
	if info, err := os.Lstat(toDir); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", toDir)}
	} else if err != nil && !os.IsNotExist(err) {
		return err
	}
	if err := os.MkdirAll(toDir, 0o755); err != nil {
		return err
	}
	if dirFile, _, err := openFileNoFollow(toDir); err != nil {
		return err
	} else {
		_ = dirFile.Close()
	}
	return os.Rename(fromPath, toPath)
}

// applyUpdateFile applies hunks to an existing file.
func applyUpdateFile(s *Sandbox, relPath string, hunks []patchHunk) error {
	p, err := s.resolveForAccess(relPath)
	if err != nil {
		return err
	}
	b, st, resolvedPath, err := s.readAllPathBounded(p, maxEditFileBytes)
	if err != nil {
		if errors.Is(err, errFileReadLimitReached) {
			size := maxEditFileBytes + 1
			if st != nil && st.Size() > 0 {
				size = st.Size()
			}
			return fmt.Errorf("[ERROR] apply_patch refuses to load %s (%d bytes) - max %d bytes; patch smaller files or split changes", relPath, size, maxEditFileBytes)
		}
		return err
	}
	if st.IsDir() {
		return fmt.Errorf("path is a directory: %s", relPath)
	}
	content := strings.ReplaceAll(string(b), "\r\n", "\n")
	content = strings.ReplaceAll(content, "\r", "\n")
	hasTrailingNewline := strings.HasSuffix(content, "\n")
	if hasTrailingNewline {
		content = strings.TrimSuffix(content, "\n")
	}
	lines := []string{}
	if content != "" {
		lines = strings.Split(content, "\n")
	}
	offset := 0
	for hunkIdx, h := range hunks {
		hunkNumber := hunkIdx + 1
		oldLines, newLines := hunkLines(h)
		if len(oldLines) == 0 && !h.hasLineNumbers {
			return fmt.Errorf("hunk %d failed to apply to %s (no context)", hunkNumber, relPath)
		}
		matchIdx := -1
		if h.hasLineNumbers {
			expected := h.oldStart - 1 + offset
			if expected < 0 {
				expected = 0
			}
			if len(oldLines) == 0 {
				if expected <= len(lines) {
					matchIdx = expected
				}
			} else if expected+len(oldLines) <= len(lines) && linesMatch(lines, expected, oldLines) {
				matchIdx = expected
			}
		}
		if matchIdx == -1 {
			matches := findHunkMatches(lines, oldLines)
			if len(matches) == 1 {
				matchIdx = matches[0]
			} else if len(matches) == 0 {
				return fmt.Errorf("hunk %d failed to apply to %s (context not found)", hunkNumber, relPath)
			} else {
				return fmt.Errorf("hunk %d failed to apply to %s (context is ambiguous)", hunkNumber, relPath)
			}
		}
		lines = applyHunk(lines, matchIdx, oldLines, newLines)
		offset += len(newLines) - len(oldLines)
	}
	out := strings.Join(lines, "\n")
	if hasTrailingNewline {
		out += "\n"
	}
	currentPath, err := s.revalidatePathForAccess(p)
	if err != nil {
		return err
	}
	if !pathsEqual(currentPath, resolvedPath) {
		return &SecurityError{Message: fmt.Sprintf("path changed during patch apply: %q (was %q, now %q)", relPath, resolvedPath, currentPath)}
	}
	return writeFilePreserveMode(currentPath, []byte(out), 0o644)
}

// hunkLines splits a hunk into old (removed/context) and new (added/context) lines.
func hunkLines(h patchHunk) ([]string, []string) {
	oldLines := []string{}
	newLines := []string{}
	for _, l := range h.lines {
		if l == "" {
			continue
		}
		pref := l[0]
		body := l[1:]
		switch pref {
		case ' ':
			oldLines = append(oldLines, body)
			newLines = append(newLines, body)
		case '-':
			oldLines = append(oldLines, body)
		case '+':
			newLines = append(newLines, body)
		}
	}
	return oldLines, newLines
}

// linesMatch checks if oldLines match lines starting at start.
func linesMatch(lines []string, start int, oldLines []string) bool {
	if start < 0 || start+len(oldLines) > len(lines) {
		return false
	}
	for i, line := range oldLines {
		if lines[start+i] != line {
			return false
		}
	}
	return true
}

// findHunkMatches finds all positions where oldLines match in lines.
func findHunkMatches(lines []string, oldLines []string) []int {
	if len(oldLines) == 0 {
		return nil
	}
	matches := []int{}
	for i := 0; i+len(oldLines) <= len(lines); i++ {
		if linesMatch(lines, i, oldLines) {
			matches = append(matches, i)
		}
	}
	return matches
}

// applyHunk applies a single hunk to lines at position start.
func applyHunk(lines []string, start int, oldLines []string, newLines []string) []string {
	out := make([]string, 0, len(lines)-len(oldLines)+len(newLines))
	out = append(out, lines[:start]...)
	out = append(out, newLines...)
	out = append(out, lines[start+len(oldLines):]...)
	return out
}
