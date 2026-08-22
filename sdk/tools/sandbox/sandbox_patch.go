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

// maxHunkLineDrift bounds how far a fuzzy hunk match may sit from the line
// number declared in the hunk header. Without this bound a hunk whose context
// happens to be unique elsewhere in the file would be applied at that unrelated
// position, producing a change that compiles but is semantically wrong.
const maxHunkLineDrift = 500

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

	// Validate and stage every operation in memory first so a patch that fails
	// halfway (e.g. a hunk that no longer matches) leaves the filesystem
	// untouched instead of half-applied.
	planner := newPatchPlanner(s)
	changed := 0
	for _, op := range ops {
		switch op.kind {
		case "add":
			if err := planner.planAdd(op.path, op.addLines); err != nil {
				return "", err
			}
			changed++
		case "delete":
			if err := planner.planDelete(op.path); err != nil {
				return "", err
			}
			changed++
		case "update":
			if err := planner.planUpdate(op.path, op.hunks); err != nil {
				return "", err
			}
			if strings.TrimSpace(op.moveTo) != "" {
				if err := planner.planMove(op.path, op.moveTo); err != nil {
					return "", err
				}
			}
			changed++
		default:
			return "", fmt.Errorf("unknown patch op: %s", op.kind)
		}
	}
	if err := planner.commit(); err != nil {
		return "", err
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

// patchAction is a single filesystem mutation staged by patchPlanner.
type patchAction struct {
	kind     string // write|delete|move
	relPath  string
	path     validatedSandboxPath
	resolved string
	// content holds the full target content for kind == "write".
	content []byte
	// moveTo/movePath describe the destination for kind == "move".
	moveTo   string
	movePath validatedSandboxPath
}

// describe renders an action for inclusion in error messages.
func (a patchAction) describe() string {
	switch a.kind {
	case "delete":
		return fmt.Sprintf("delete %s", a.relPath)
	case "move":
		return fmt.Sprintf("move %s -> %s", a.relPath, a.moveTo)
	default:
		return fmt.Sprintf("write %s", a.relPath)
	}
}

// stagedFile is the in-memory state of a file touched by earlier patch ops.
type stagedFile struct {
	content []byte
	deleted bool
}

// patchPlanner validates and stages every patch operation in memory so the
// whole patch is known to be applicable before any of it reaches the disk.
type patchPlanner struct {
	s       *Sandbox
	actions []patchAction
	staged  map[string]*stagedFile
}

// newPatchPlanner creates an empty planner bound to a sandbox.
func newPatchPlanner(s *Sandbox) *patchPlanner {
	return &patchPlanner{s: s, staged: map[string]*stagedFile{}}
}

// planAdd stages creation of a new file, rejecting paths that already exist.
func (p *patchPlanner) planAdd(relPath string, lines []string) error {
	vp, err := p.s.resolveForAccess(relPath)
	if err != nil {
		return err
	}
	resolvedPath, err := p.s.revalidatePathForAccess(vp)
	if err != nil {
		return err
	}
	if staged, ok := p.staged[vp.resolved]; ok {
		if !staged.deleted {
			return fmt.Errorf("cannot add %s: file already exists in this patch; use '*** Update File: %s'", relPath, relPath)
		}
	} else if _, err := os.Lstat(resolvedPath); err == nil {
		return fmt.Errorf("cannot add %s: file already exists; use '*** Update File: %s'", relPath, relPath)
	} else if !os.IsNotExist(err) {
		return err
	}
	content := strings.Join(lines, "\n")
	if len(lines) > 0 {
		content += "\n"
	}
	p.stage(vp.resolved, &stagedFile{content: []byte(content)})
	p.actions = append(p.actions, patchAction{kind: "write", relPath: relPath, path: vp, resolved: resolvedPath, content: []byte(content)})
	return nil
}

// planDelete stages removal of an existing regular file.
func (p *patchPlanner) planDelete(relPath string) error {
	vp, err := p.s.resolveForAccess(relPath)
	if err != nil {
		return err
	}
	resolvedPath, err := p.s.revalidatePathForAccess(vp)
	if err != nil {
		return err
	}
	if staged, ok := p.staged[vp.resolved]; ok {
		if staged.deleted {
			return fmt.Errorf("cannot delete %s: already deleted in this patch", relPath)
		}
	} else {
		f, info, err := openFileNoFollow(resolvedPath)
		if err != nil {
			return err
		}
		_ = f.Close()
		if !info.Mode().IsRegular() {
			return fmt.Errorf("cannot delete %s: not a regular file", relPath)
		}
	}
	p.stage(vp.resolved, &stagedFile{deleted: true})
	p.actions = append(p.actions, patchAction{kind: "delete", relPath: relPath, path: vp, resolved: resolvedPath})
	return nil
}

// planMove stages a rename, validating the destination directory up front.
func (p *patchPlanner) planMove(fromRel, toRel string) error {
	fromVP, err := p.s.resolveForAccess(fromRel)
	if err != nil {
		return err
	}
	toVP, err := p.s.resolveForAccess(toRel)
	if err != nil {
		return err
	}
	fromPath, err := p.s.revalidatePathForAccess(fromVP)
	if err != nil {
		return err
	}
	toPath, err := p.s.revalidatePathForAccess(toVP)
	if err != nil {
		return err
	}
	toDir := filepath.Dir(toPath)
	if info, err := os.Lstat(toDir); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", toDir)}
	} else if err != nil && !os.IsNotExist(err) {
		return err
	}
	if staged, ok := p.staged[fromVP.resolved]; ok && !staged.deleted {
		p.stage(toVP.resolved, &stagedFile{content: staged.content})
		p.stage(fromVP.resolved, &stagedFile{deleted: true})
	}
	p.actions = append(p.actions, patchAction{kind: "move", relPath: fromRel, path: fromVP, resolved: fromPath, moveTo: toRel, movePath: toVP})
	return nil
}

// planUpdate stages the result of applying hunks to an existing file.
func (p *patchPlanner) planUpdate(relPath string, hunks []patchHunk) error {
	vp, err := p.s.resolveForAccess(relPath)
	if err != nil {
		return err
	}
	var (
		raw      []byte
		resolved string
	)
	if staged, ok := p.staged[vp.resolved]; ok {
		if staged.deleted {
			return fmt.Errorf("cannot update %s: file was deleted earlier in this patch", relPath)
		}
		raw = staged.content
		resolved, err = p.s.revalidatePathForAccess(vp)
		if err != nil {
			return err
		}
	} else {
		b, st, resolvedPath, err := p.s.readAllPathBounded(vp, maxEditFileBytes)
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
		raw = b
		resolved = resolvedPath
	}
	out, err := applyHunksToContent(relPath, string(raw), hunks)
	if err != nil {
		return err
	}
	p.stage(vp.resolved, &stagedFile{content: []byte(out)})
	p.actions = append(p.actions, patchAction{kind: "write", relPath: relPath, path: vp, resolved: resolved, content: []byte(out)})
	return nil
}

// stage records the pending state of a file so later ops in the same patch
// compose on top of earlier ones instead of reading stale bytes from disk.
func (p *patchPlanner) stage(key string, state *stagedFile) {
	if p.staged == nil {
		p.staged = map[string]*stagedFile{}
	}
	p.staged[key] = state
}

// commit writes every staged action to disk. If an action fails after others
// already landed, the error reports which files were applied and which were
// not so the caller can recover instead of blindly retrying the whole patch.
func (p *patchPlanner) commit() error {
	applied := []string{}
	for i, a := range p.actions {
		if err := p.commitAction(a); err != nil {
			if len(applied) == 0 {
				return err
			}
			pending := []string{}
			for _, rest := range p.actions[i:] {
				pending = append(pending, rest.describe())
			}
			return fmt.Errorf("apply_patch partially applied: %s failed (%v); already applied: %s; not applied: %s", a.describe(), err, strings.Join(applied, ", "), strings.Join(pending, ", "))
		}
		applied = append(applied, a.describe())
	}
	return nil
}

// commitAction performs a single staged action, revalidating the path first.
func (p *patchPlanner) commitAction(a patchAction) error {
	switch a.kind {
	case "delete":
		resolvedPath, err := p.s.revalidatePathForAccess(a.path)
		if err != nil {
			return err
		}
		return os.Remove(resolvedPath)
	case "move":
		fromPath, err := p.s.revalidatePathForAccess(a.path)
		if err != nil {
			return err
		}
		toPath, err := p.s.revalidatePathForAccess(a.movePath)
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
	default:
		currentPath, err := p.s.revalidatePathForAccess(a.path)
		if err != nil {
			return err
		}
		if !pathsEqual(currentPath, a.resolved) {
			return &SecurityError{Message: fmt.Sprintf("path changed during patch apply: %q (was %q, now %q)", a.relPath, a.resolved, currentPath)}
		}
		return writeFilePreserveMode(currentPath, a.content, 0o644)
	}
}

// applyAddFile creates a new file with the given content.
func applyAddFile(s *Sandbox, relPath string, lines []string) error {
	p := newPatchPlanner(s)
	if err := p.planAdd(relPath, lines); err != nil {
		return err
	}
	return p.commit()
}

// applyDeleteFile removes a file from the sandbox.
func applyDeleteFile(s *Sandbox, relPath string) error {
	p := newPatchPlanner(s)
	if err := p.planDelete(relPath); err != nil {
		return err
	}
	return p.commit()
}

// applyMoveFile renames a file within the sandbox.
func applyMoveFile(s *Sandbox, fromRel, toRel string) error {
	p := newPatchPlanner(s)
	if err := p.planMove(fromRel, toRel); err != nil {
		return err
	}
	return p.commit()
}

// applyUpdateFile applies hunks to an existing file.
func applyUpdateFile(s *Sandbox, relPath string, hunks []patchHunk) error {
	p := newPatchPlanner(s)
	if err := p.planUpdate(relPath, hunks); err != nil {
		return err
	}
	return p.commit()
}

// applyHunksToContent applies hunks to file content in memory and returns the
// resulting content without touching the filesystem.
func applyHunksToContent(relPath string, raw string, hunks []patchHunk) (string, error) {
	lineEnding := consistentLineEnding(raw)
	content := strings.ReplaceAll(raw, "\r\n", "\n")
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
			return "", fmt.Errorf("hunk %d failed to apply to %s (no context)", hunkNumber, relPath)
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
			if len(matches) == 0 {
				return "", fmt.Errorf("hunk %d failed to apply to %s (context not found)", hunkNumber, relPath)
			}
			if h.hasLineNumbers {
				// Anchor the fuzzy match near the declared @@ position so a
				// unique-but-distant match cannot silently patch the wrong place.
				expected := h.oldStart - 1 + offset
				if expected < 0 {
					expected = 0
				}
				best, dist, ambiguous := nearestHunkMatch(matches, expected)
				if ambiguous {
					return "", fmt.Errorf("hunk %d failed to apply to %s (context is ambiguous)", hunkNumber, relPath)
				}
				if dist > maxHunkLineDrift {
					return "", fmt.Errorf("hunk %d failed to apply to %s (closest context match is %d lines away from declared line %d - max %d; re-read the file and rebuild the hunk)", hunkNumber, relPath, dist, h.oldStart, maxHunkLineDrift)
				}
				matchIdx = best
			} else if len(matches) == 1 {
				matchIdx = matches[0]
			} else {
				return "", fmt.Errorf("hunk %d failed to apply to %s (context is ambiguous)", hunkNumber, relPath)
			}
		}
		lines = applyHunk(lines, matchIdx, oldLines, newLines)
		offset += len(newLines) - len(oldLines)
	}
	out := strings.Join(lines, "\n")
	if hasTrailingNewline {
		out += "\n"
	}
	if lineEnding == "\r\n" {
		out = strings.ReplaceAll(out, "\n", "\r\n")
	}
	return out, nil
}

func consistentLineEnding(content string) string {
	if !strings.Contains(content, "\r\n") {
		return "\n"
	}
	withoutCRLF := strings.ReplaceAll(content, "\r\n", "")
	if strings.ContainsAny(withoutCRLF, "\r\n") {
		return "\n"
	}
	return "\r\n"
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

// nearestHunkMatch picks the candidate match closest to expected. It reports
// the distance of that candidate and whether two candidates are equally close
// (which makes the target position ambiguous).
func nearestHunkMatch(matches []int, expected int) (int, int, bool) {
	best := -1
	bestDist := 0
	ambiguous := false
	for _, m := range matches {
		dist := m - expected
		if dist < 0 {
			dist = -dist
		}
		switch {
		case best == -1 || dist < bestDist:
			best = m
			bestDist = dist
			ambiguous = false
		case dist == bestDist && m != best:
			ambiguous = true
		}
	}
	return best, bestDist, ambiguous
}

// applyHunk applies a single hunk to lines at position start.
func applyHunk(lines []string, start int, oldLines []string, newLines []string) []string {
	out := make([]string, 0, len(lines)-len(oldLines)+len(newLines))
	out = append(out, lines[:start]...)
	out = append(out, newLines...)
	out = append(out, lines[start+len(oldLines):]...)
	return out
}
