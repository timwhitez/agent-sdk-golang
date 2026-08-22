from pathlib import Path

path = Path("sdk/tools/sandbox/sandbox_patch.go")
text = path.read_text()
old_plan = '''\ttoDir := filepath.Dir(toPath)
\tif info, err := os.Lstat(toDir); err == nil && info.Mode()&os.ModeSymlink != 0 {
\t\treturn &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", toDir)}
\t} else if err != nil && !os.IsNotExist(err) {
\t\treturn err
\t}
\tif staged, ok := p.staged[fromVP.resolved]; ok && !staged.deleted {
'''
new_plan = '''\ttoDir := filepath.Dir(toPath)
\tif info, err := os.Lstat(toDir); err == nil && info.Mode()&os.ModeSymlink != 0 {
\t\treturn &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", toDir)}
\t} else if err != nil && !os.IsNotExist(err) {
\t\treturn err
\t}
\tif staged, ok := p.staged[toVP.resolved]; ok {
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
if text.count(old_plan) != 1:
    raise SystemExit(f"planMove anchor count={text.count(old_plan)}")
text = text.replace(old_plan, new_plan)
old_commit = '''\t\ttoDir := filepath.Dir(toPath)
\t\tif info, err := os.Lstat(toDir); err == nil && info.Mode()&os.ModeSymlink != 0 {
\t\t\treturn &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", toDir)}
\t\t} else if err != nil && !os.IsNotExist(err) {
\t\t\treturn err
\t\t}
\t\tif err := os.MkdirAll(toDir, 0o755); err != nil {
'''
new_commit = '''\t\ttoDir := filepath.Dir(toPath)
\t\tif info, err := os.Lstat(toDir); err == nil && info.Mode()&os.ModeSymlink != 0 {
\t\t\treturn &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", toDir)}
\t\t} else if err != nil && !os.IsNotExist(err) {
\t\t\treturn err
\t\t}
\t\t// os.Rename replaces an existing regular file on Unix. Re-check the
\t\t// destination immediately before commit so both pre-existing files and
\t\t// files created after planning fail closed on every platform.
\t\tif _, err := os.Lstat(toPath); err == nil {
\t\t\treturn fmt.Errorf("cannot move %s to %s: destination already exists", a.relPath, a.moveTo)
\t\t} else if !os.IsNotExist(err) {
\t\t\treturn err
\t\t}
\t\tif err := os.MkdirAll(toDir, 0o755); err != nil {
'''
if text.count(old_commit) != 1:
    raise SystemExit(f"commit move anchor count={text.count(old_commit)}")
path.write_text(text.replace(old_commit, new_commit))

Path("sdk/tools/sandbox/sandbox_patch_move_conflict_test.go").write_text(r'''package sandbox

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestApplyPatchMoveRejectsExistingDestination(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(root, "source.txt")
	destination := filepath.Join(root, "destination.txt")
	if err := os.WriteFile(source, []byte("source\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(destination, []byte("important\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	patch := `*** Begin Patch
*** Update File: source.txt
*** Move to: destination.txt
@@ -1,1 +1,1 @@
-source
+moved
*** End Patch`

	_, err = applyPatchToSandbox(s, patch)
	if err == nil || !strings.Contains(err.Error(), "destination already exists") {
		t.Fatalf("applyPatchToSandbox() error = %v, want destination conflict", err)
	}
	if got, err := os.ReadFile(destination); err != nil || string(got) != "important\n" {
		t.Fatalf("destination = %q, %v; want original content", got, err)
	}
	if got, err := os.ReadFile(source); err != nil || string(got) != "source\n" {
		t.Fatalf("source = %q, %v; want unchanged source", got, err)
	}
}

func TestPatchPlannerMoveRejectsDestinationCreatedAfterPlanning(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(root, "source.txt")
	destination := filepath.Join(root, "destination.txt")
	if err := os.WriteFile(source, []byte("source\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	planner := newPatchPlanner(s)
	if err := planner.planMove("source.txt", "destination.txt"); err != nil {
		t.Fatalf("planMove() error = %v", err)
	}
	if err := os.WriteFile(destination, []byte("concurrent\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	err = planner.commit()
	if err == nil || !strings.Contains(err.Error(), "destination already exists") {
		t.Fatalf("commit() error = %v, want destination conflict", err)
	}
	if got, err := os.ReadFile(destination); err != nil || string(got) != "concurrent\n" {
		t.Fatalf("destination = %q, %v; want concurrent content", got, err)
	}
	if got, err := os.ReadFile(source); err != nil || string(got) != "source\n" {
		t.Fatalf("source = %q, %v; want source retained", got, err)
	}
}
''')
