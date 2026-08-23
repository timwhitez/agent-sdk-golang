package sandbox

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
		t.Fatalf("commit() error = %v", err)
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
		t.Fatalf("commit() error = %v", err)
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
		t.Fatalf("commit() error = %v", err)
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
