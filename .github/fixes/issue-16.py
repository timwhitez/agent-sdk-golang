from pathlib import Path

path = Path("sdk/tools/sandbox/sandbox_patch.go")
text = path.read_text()

old = '''type patchPlanner struct {
\ts       *Sandbox
\tactions []patchAction
\tstaged  map[string]*stagedFile
}

// newPatchPlanner creates an empty planner bound to a sandbox.
func newPatchPlanner(s *Sandbox) *patchPlanner {
\treturn &patchPlanner{s: s, staged: map[string]*stagedFile{}}
}
'''
new = '''type patchPlanner struct {
\ts        *Sandbox
\tactions  []patchAction
\tstaged   map[string]*stagedFile
\tbaseline map[string]writeTargetSnapshot
}

// newPatchPlanner creates an empty planner bound to a sandbox.
func newPatchPlanner(s *Sandbox) *patchPlanner {
\treturn &patchPlanner{
\t\ts:        s,
\t\tstaged:   map[string]*stagedFile{},
\t\tbaseline: map[string]writeTargetSnapshot{},
\t}
}

// rememberBaseline stores the first on-disk state observed for a path. Later
// operations in the same patch compose through staged state and must not replace
// this baseline with their own planned bytes.
func (p *patchPlanner) rememberBaseline(key string, snapshot writeTargetSnapshot) {
\tif p.baseline == nil {
\t\tp.baseline = map[string]writeTargetSnapshot{}
\t}
\tif _, exists := p.baseline[key]; !exists {
\t\tp.baseline[key] = snapshot
\t}
}

func patchSnapshot(info os.FileInfo, content []byte) writeTargetSnapshot {
\treturn writeTargetSnapshot{
\t\tExists:      true,
\t\tInfo:        info,
\t\tContent:     string(content),
\t\tContentFull: true,
\t}
}

func snapshotPatchedFile(path string, content []byte) (writeTargetSnapshot, error) {
\tinfo, err := os.Lstat(path)
\tif err != nil {
\t\treturn writeTargetSnapshot{}, err
\t}
\treturn patchSnapshot(info, content), nil
}
'''
if text.count(old) != 1:
    raise SystemExit(f"planner anchor count={text.count(old)}")
text = text.replace(old, new)

old = '''\tcontent := strings.Join(lines, "\\n")
\tif len(lines) > 0 {
\t\tcontent += "\\n"
\t}
'''
new = '''\tp.rememberBaseline(vp.resolved, writeTargetSnapshot{Exists: false})
\tcontent := strings.Join(lines, "\\n")
\tif len(lines) > 0 {
\t\tcontent += "\\n"
\t}
'''
if text.count(old) != 1:
    raise SystemExit(f"planAdd anchor count={text.count(old)}")
text = text.replace(old, new)

old = '''\t} else {
\t\tf, info, err := openFileNoFollow(resolvedPath)
\t\tif err != nil {
\t\t\treturn err
\t\t}
\t\t_ = f.Close()
\t\tif !info.Mode().IsRegular() {
\t\t\treturn fmt.Errorf("cannot delete %s: not a regular file", relPath)
\t\t}
\t}
'''
new = '''\t} else {
\t\traw, info, _, err := p.s.readAllPathBounded(vp, maxEditFileBytes)
\t\tif err != nil {
\t\t\treturn err
\t\t}
\t\tif !info.Mode().IsRegular() {
\t\t\treturn fmt.Errorf("cannot delete %s: not a regular file", relPath)
\t\t}
\t\tp.rememberBaseline(vp.resolved, patchSnapshot(info, raw))
\t}
'''
if text.count(old) != 1:
    raise SystemExit(f"planDelete anchor count={text.count(old)}")
text = text.replace(old, new)

old = '''\tif staged, ok := p.staged[toVP.resolved]; ok {
\t\tif !staged.deleted {
\t\t\treturn fmt.Errorf("cannot move %s to %s: destination already exists in this patch", fromRel, toRel)
\t\t}
\t} else if _, err := os.Lstat(toPath); err == nil {
\t\treturn fmt.Errorf("cannot move %s to %s: destination already exists", fromRel, toRel)
\t} else if !os.IsNotExist(err) {
\t\treturn err
\t}
\tif staged, ok := p.staged[fromVP.resolved]; ok && !staged.deleted {
'''
new = '''\tif staged, ok := p.staged[toVP.resolved]; ok {
\t\tif !staged.deleted {
\t\t\treturn fmt.Errorf("cannot move %s to %s: destination already exists in this patch", fromRel, toRel)
\t\t}
\t} else if _, err := os.Lstat(toPath); err == nil {
\t\treturn fmt.Errorf("cannot move %s to %s: destination already exists", fromRel, toRel)
\t} else if !os.IsNotExist(err) {
\t\treturn err
\t} else {
\t\tp.rememberBaseline(toVP.resolved, writeTargetSnapshot{Exists: false})
\t}
\tif staged, ok := p.staged[fromVP.resolved]; ok {
\t\tif staged.deleted {
\t\t\treturn fmt.Errorf("cannot move %s: source was deleted earlier in this patch", fromRel)
\t\t}
'''
if text.count(old) != 1:
    raise SystemExit(f"planMove staged anchor count={text.count(old)}")
text = text.replace(old, new)
# The prior condition's body now belongs to the unconditional staged branch.
text = text.replace('''\t\tp.stage(toVP.resolved, &stagedFile{content: staged.content})
\t\tp.stage(fromVP.resolved, &stagedFile{deleted: true})
\t}
\tp.actions = append''', '''\t\tp.stage(toVP.resolved, &stagedFile{content: staged.content})
\t\tp.stage(fromVP.resolved, &stagedFile{deleted: true})
\t} else {
\t\traw, info, _, err := p.s.readAllPathBounded(fromVP, maxEditFileBytes)
\t\tif err != nil {
\t\t\treturn err
\t\t}
\t\tif !info.Mode().IsRegular() {
\t\t\treturn fmt.Errorf("cannot move %s: not a regular file", fromRel)
\t\t}
\t\tp.rememberBaseline(fromVP.resolved, patchSnapshot(info, raw))
\t}
\tp.actions = append''', 1)

old = '''\t\tif st.IsDir() {
\t\t\treturn fmt.Errorf("path is a directory: %s", relPath)
\t\t}
\t\traw = b
\t\tresolved = resolvedPath
'''
new = '''\t\tif st.IsDir() {
\t\t\treturn fmt.Errorf("path is a directory: %s", relPath)
\t\t}
\t\traw = b
\t\tresolved = resolvedPath
\t\tp.rememberBaseline(vp.resolved, patchSnapshot(st, b))
'''
if text.count(old) != 1:
    raise SystemExit(f"planUpdate snapshot anchor count={text.count(old)}")
text = text.replace(old, new)

start = text.index("func (p *patchPlanner) commit() error {")
end = text.index("\n// applyAddFile creates", start)
replacement = r'''func (p *patchPlanner) commit() error {
	expected := make(map[string]writeTargetSnapshot, len(p.baseline))
	for key, snapshot := range p.baseline {
		expected[key] = snapshot
	}
	applied := []string{}
	for i, a := range p.actions {
		if err := p.commitAction(a, expected); err != nil {
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

func verifyPatchTargetUnchanged(expected map[string]writeTargetSnapshot, key, path, displayPath string) error {
	snapshot, ok := expected[key]
	if !ok {
		return fmt.Errorf("%w: no planning snapshot for %s", errStaleWriteTarget, displayPath)
	}
	return verifyWriteTargetUnchanged(path, snapshot, displayPath)
}

func updatePatchExpected(expected map[string]writeTargetSnapshot, key, path string, content []byte) error {
	snapshot, err := snapshotPatchedFile(path, content)
	if err != nil {
		return err
	}
	expected[key] = snapshot
	return nil
}

// commitAction performs a single staged action, revalidating identity and
// content immediately before the mutation. expected is updated after each
// successful action so multiple operations in one patch do not reject their own
// earlier writes as external changes.
func (p *patchPlanner) commitAction(a patchAction, expected map[string]writeTargetSnapshot) error {
	switch a.kind {
	case "delete":
		resolvedPath, err := p.s.revalidatePathForAccess(a.path)
		if err != nil {
			return err
		}
		if err := verifyPatchTargetUnchanged(expected, a.path.resolved, resolvedPath, a.relPath); err != nil {
			return err
		}
		if err := os.Remove(resolvedPath); err != nil {
			return err
		}
		expected[a.path.resolved] = writeTargetSnapshot{Exists: false}
		return nil
	case "move":
		fromPath, err := p.s.revalidatePathForAccess(a.path)
		if err != nil {
			return err
		}
		toPath, err := p.s.revalidatePathForAccess(a.movePath)
		if err != nil {
			return err
		}
		if err := verifyPatchTargetUnchanged(expected, a.path.resolved, fromPath, a.relPath); err != nil {
			return err
		}
		toDir := filepath.Dir(toPath)
		if info, err := os.Lstat(toDir); err == nil && info.Mode()&os.ModeSymlink != 0 {
			return &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", toDir)}
		} else if err != nil && !os.IsNotExist(err) {
			return err
		}
		// Keep the explicit non-clobbering contract from issue #18.
		if _, err := os.Lstat(toPath); err == nil {
			return fmt.Errorf("cannot move %s to %s: destination already exists", a.relPath, a.moveTo)
		} else if !os.IsNotExist(err) {
			return err
		}
		if err := verifyPatchTargetUnchanged(expected, a.movePath.resolved, toPath, a.moveTo); err != nil {
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
		sourceContent := []byte(expected[a.path.resolved].Content)
		if err := os.Rename(fromPath, toPath); err != nil {
			return err
		}
		expected[a.path.resolved] = writeTargetSnapshot{Exists: false}
		return updatePatchExpected(expected, a.movePath.resolved, toPath, sourceContent)
	default:
		currentPath, err := p.s.revalidatePathForAccess(a.path)
		if err != nil {
			return err
		}
		if !pathsEqual(currentPath, a.resolved) {
			return &SecurityError{Message: fmt.Sprintf("path changed during patch apply: %q (was %q, now %q)", a.relPath, a.resolved, currentPath)}
		}
		if err := verifyPatchTargetUnchanged(expected, a.path.resolved, currentPath, a.relPath); err != nil {
			return err
		}
		if err := writeFilePreserveMode(currentPath, a.content, 0o644); err != nil {
			return err
		}
		return updatePatchExpected(expected, a.path.resolved, currentPath, a.content)
	}
}
'''
text = text[:start] + replacement + text[end:]
path.write_text(text)

Path("sdk/tools/sandbox/sandbox_patch_stale_target_test.go").write_text(r'''package sandbox

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func singleLinePatchHunk(old, replacement string) patchHunk {
	return patchHunk{
		lines:          []string{"-" + old, "+" + replacement},
		oldStart:       1,
		oldLen:         1,
		newStart:       1,
		newLen:         1,
		hasLineNumbers: true,
	}
}

func TestPatchPlannerRejectsUpdatedFileChangedAfterPlanning(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "target.txt")
	if err := os.WriteFile(target, []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s, _ := New(root)
	planner := newPatchPlanner(s)
	if err := planner.planUpdate("target.txt", []patchHunk{singleLinePatchHunk("old", "patched")}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("concurrent writer\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := planner.commit(); !errors.Is(err, errStaleWriteTarget) {
		t.Fatalf("commit() error = %v, want workspace_file_changed", err)
	}
	if got, _ := os.ReadFile(target); string(got) != "concurrent writer\n" {
		t.Fatalf("concurrent update was lost: %q", got)
	}
}

func TestPatchPlannerRejectsDeleteTargetReplacedAfterPlanning(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "target.txt")
	if err := os.WriteFile(target, []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s, _ := New(root)
	planner := newPatchPlanner(s)
	if err := planner.planDelete("target.txt"); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(target); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("replacement\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := planner.commit(); !errors.Is(err, errStaleWriteTarget) {
		t.Fatalf("commit() error = %v, want workspace_file_changed", err)
	}
	if got, _ := os.ReadFile(target); string(got) != "replacement\n" {
		t.Fatalf("replacement was deleted: %q", got)
	}
}

func TestPatchPlannerRejectsMoveSourceChangedAfterPlanning(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source.txt")
	destination := filepath.Join(root, "destination.txt")
	if err := os.WriteFile(source, []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s, _ := New(root)
	planner := newPatchPlanner(s)
	if err := planner.planMove("source.txt", "destination.txt"); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("concurrent\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := planner.commit(); !errors.Is(err, errStaleWriteTarget) {
		t.Fatalf("commit() error = %v, want workspace_file_changed", err)
	}
	if got, _ := os.ReadFile(source); string(got) != "concurrent\n" {
		t.Fatalf("source update was lost: %q", got)
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatalf("destination unexpectedly created: %v", err)
	}
}

func TestPatchPlannerMultipleUpdatesDoNotConflictWithOwnWrites(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "target.txt")
	if err := os.WriteFile(target, []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s, _ := New(root)
	planner := newPatchPlanner(s)
	if err := planner.planUpdate("target.txt", []patchHunk{singleLinePatchHunk("old", "first")}); err != nil {
		t.Fatal(err)
	}
	if err := planner.planUpdate("target.txt", []patchHunk{singleLinePatchHunk("first", "second")}); err != nil {
		t.Fatal(err)
	}
	if err := planner.commit(); err != nil {
		t.Fatalf("commit() error = %v", err)
	}
	if got, _ := os.ReadFile(target); string(got) != "second\n" {
		t.Fatalf("final content = %q", got)
	}
}
''')
