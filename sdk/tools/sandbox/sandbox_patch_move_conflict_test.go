package sandbox

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
