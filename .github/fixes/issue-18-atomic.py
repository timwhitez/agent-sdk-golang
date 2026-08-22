from pathlib import Path

path = Path("sdk/tools/sandbox/sandbox_patch.go")
text = path.read_text()
old = '''\t\tif err := os.Rename(fromPath, toPath); err != nil {
\t\t\treturn err
\t\t}
\t\treturn nil
'''
new = '''\t\tif err := moveRegularFileNoReplace(fromPath, toPath); err != nil {
\t\t\treturn err
\t\t}
\t\treturn nil
'''
if text.count(old) != 1:
    raise SystemExit(f"move commit anchor count={text.count(old)}")
text = text.replace(old, new)
anchor = '''// applyAddFile creates a new file and writes content.
func applyAddFile(s *Sandbox, relPath string, lines []string) error {
'''
insert = '''var beforePatchMoveNoReplace = func(string, string) {}

// moveRegularFileNoReplace atomically creates the destination name without
// replacing an existing path. Patch moves are limited to regular files, so a
// hard link plus source unlink provides portable no-clobber semantics on the
// filesystems where os.Rename would also be a same-filesystem operation.
func moveRegularFileNoReplace(fromPath, toPath string) error {
\tif beforePatchMoveNoReplace != nil {
\t\tbeforePatchMoveNoReplace(fromPath, toPath)
\t}
\tif err := os.Link(fromPath, toPath); err != nil {
\t\tif os.IsExist(err) {
\t\t\treturn fmt.Errorf("move destination already exists: %s", toPath)
\t\t}
\t\treturn fmt.Errorf("create non-clobbering move destination %s: %w", toPath, err)
\t}
\tif err := os.Remove(fromPath); err != nil {
\t\trollbackErr := os.Remove(toPath)
\t\tif rollbackErr != nil {
\t\t\treturn fmt.Errorf("remove move source %s: %w (destination rollback failed: %v)", fromPath, err, rollbackErr)
\t\t}
\t\treturn fmt.Errorf("remove move source %s: %w", fromPath, err)
\t}
\treturn nil
}

// applyAddFile creates a new file and writes content.
func applyAddFile(s *Sandbox, relPath string, lines []string) error {
'''
if text.count(anchor) != 1:
    raise SystemExit(f"move helper anchor count={text.count(anchor)}")
path.write_text(text.replace(anchor, insert))

test = Path("sdk/tools/sandbox/sandbox_patch_move_conflict_test.go")
existing = test.read_text()
existing += r'''

func TestPatchPlannerMoveRejectsDestinationCreatedAfterFinalCheck(t *testing.T) {
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
	originalHook := beforePatchMoveNoReplace
	beforePatchMoveNoReplace = func(_, target string) {
		if err := os.WriteFile(target, []byte("last-moment\n"), 0o644); err != nil {
			t.Errorf("create last-moment destination: %v", err)
		}
	}
	t.Cleanup(func() { beforePatchMoveNoReplace = originalHook })

	err = planner.commit()
	if err == nil || !strings.Contains(err.Error(), "destination already exists") {
		t.Fatalf("commit() error = %v, want atomic destination conflict", err)
	}
	if got, err := os.ReadFile(destination); err != nil || string(got) != "last-moment\n" {
		t.Fatalf("destination = %q, %v; want last-moment content", got, err)
	}
	if got, err := os.ReadFile(source); err != nil || string(got) != "source\n" {
		t.Fatalf("source = %q, %v; want source retained", got, err)
	}
}
'''
test.write_text(existing)
