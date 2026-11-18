package sandbox

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// atomicDirImpls lists every anchored-directory implementation runnable here:
// the platform default (O_PATH dirfd on Linux) and the portable os.Root one
// that is the default on darwin, windows and other unix systems.
var atomicDirImpls = map[string]func(string, bool) (atomicDir, error){
	"platform": openAtomicDir,
	"root":     openRootAtomicDir,
}

func useAtomicDirImpl(t *testing.T, name string) {
	impl, ok := atomicDirImpls[name]
	if !ok {
		t.Fatalf("unknown atomicDir implementation %q", name)
	}
	orig := openAtomicDir
	openAtomicDir = impl
	t.Cleanup(func() { openAtomicDir = orig })
}

func forEachAtomicDirImpl(t *testing.T, f func(t *testing.T, impl string)) {
	for _, name := range []string{"platform", "root"} {
		t.Run(name, func(t *testing.T) {
			useAtomicDirImpl(t, name)
			f(t, name)
		})
	}
}

func runSandboxTool(t *testing.T, s *Sandbox, name, args string) error {
	t.Helper()
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowAllConfirmer{}, nil })
	for _, tool := range Tools() {
		if tool.Name == name {
			_, err := tool.Execute(context.Background(), args, deps)
			return err
		}
	}
	t.Fatalf("no tool %q", name)
	return nil
}

func mustWrite(t *testing.T, path, content string, mode os.FileMode) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), mode); err != nil {
		t.Fatal(err)
	}
}

func wantContent(t *testing.T, path, want string) {
	t.Helper()
	if got, err := os.ReadFile(path); err != nil || string(got) != want {
		t.Fatalf("%s = %q, %v; want %q", path, got, err, want)
	}
}

func wantAbsent(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Lstat(path); !os.IsNotExist(err) {
		t.Fatalf("%s exists (err=%v); want absent", path, err)
	}
}

func evalTempDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

// The anchored layer itself refuses symlink components even when handed a path
// that was never validated.
func TestAnchoredWriteRefusesSymlinkComponents(t *testing.T) {
	forEachAtomicDirImpl(t, func(t *testing.T, _ string) {
		root, outside := evalTempDir(t), evalTempDir(t)
		if err := os.Symlink(outside, filepath.Join(root, "dirlink")); err != nil {
			t.Fatal(err)
		}
		if err := writeFilePreserveMode(filepath.Join(root, "dirlink", "f.txt"), []byte("x"), 0o644); err == nil {
			t.Fatal("write through symlinked intermediate directory succeeded")
		}
		wantAbsent(t, filepath.Join(outside, "f.txt"))

		victim := filepath.Join(outside, "victim.txt")
		mustWrite(t, victim, "victim", 0o644)
		if err := os.Symlink(victim, filepath.Join(root, "leaf.txt")); err != nil {
			t.Fatal(err)
		}
		err := writeFilePreserveMode(filepath.Join(root, "leaf.txt"), []byte("x"), 0o644)
		if _, ok := err.(*SecurityError); !ok {
			t.Fatalf("write to final-component symlink: err = %v, want SecurityError", err)
		}
		wantContent(t, victim, "victim")

		if err := removeFileAnchored(filepath.Join(root, "dirlink", "victim.txt")); err == nil {
			t.Fatal("remove through symlinked intermediate directory succeeded")
		}
		wantContent(t, victim, "victim")
	})
}

// Tool-level contract: escapes are refused, in-root symlinks keep resolving to
// their in-root target (TestSandboxResolveAllowsSymlinkInsideRoot).
func TestFileToolsSymlinkContractAllImpls(t *testing.T) {
	forEachAtomicDirImpl(t, func(t *testing.T, _ string) {
		root, outside := evalTempDir(t), evalTempDir(t)
		s, err := New(root)
		if err != nil {
			t.Fatal(err)
		}
		mustWrite(t, filepath.Join(outside, "victim.txt"), "victim", 0o644)
		mustWrite(t, filepath.Join(root, "src.txt"), "src\n", 0o644)
		if err := os.Symlink(outside, filepath.Join(root, "out")); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join(outside, "victim.txt"), filepath.Join(root, "leaf.txt")); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Join(root, "real"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join(root, "real"), filepath.Join(root, "in")); err != nil {
			t.Fatal(err)
		}

		for _, tc := range []struct{ name, tool, args string }{
			{"intermediate symlink", "write", `{"file_path":"out/new.txt","content":"x"}`},
			{"final symlink", "write", `{"file_path":"leaf.txt","content":"x"}`},
			{"dotdot", "write", `{"file_path":"../escape.txt","content":"x"}`},
			{"move to symlinked dir", "apply_patch", `{"patch":"*** Begin Patch\n*** Update File: src.txt\n*** Move to: out/moved.txt\n@@\n-src\n+moved\n*** End Patch"}`},
			{"delete through symlink", "apply_patch", `{"patch":"*** Begin Patch\n*** Delete File: out/victim.txt\n*** End Patch"}`},
		} {
			if err := runSandboxTool(t, s, tc.tool, tc.args); err == nil {
				t.Errorf("%s: succeeded, want refusal", tc.name)
			}
		}
		wantContent(t, filepath.Join(outside, "victim.txt"), "victim")
		wantContent(t, filepath.Join(root, "src.txt"), "src\n")
		wantAbsent(t, filepath.Join(outside, "new.txt"))
		wantAbsent(t, filepath.Join(outside, "moved.txt"))
		wantAbsent(t, filepath.Join(filepath.Dir(root), "escape.txt"))

		if err := runSandboxTool(t, s, "write", `{"file_path":"in/f.txt","content":"inside"}`); err != nil {
			t.Fatalf("write through in-root absolute symlink: %v", err)
		}
		wantContent(t, filepath.Join(root, "real", "f.txt"), "inside")
		if info, err := os.Lstat(filepath.Join(root, "in")); err != nil || info.Mode()&os.ModeSymlink == 0 {
			t.Fatalf("in-root symlink replaced: %v %v", info, err)
		}
	})
}

// Issue #240: a destination directory swapped for a symlink after the move was
// validated must not redirect the hard link outside the sandbox.
func TestApplyPatchMoveFailsClosedWhenDestinationDirSwapped(t *testing.T) {
	forEachAtomicDirImpl(t, func(t *testing.T, _ string) {
		root, outside := evalTempDir(t), evalTempDir(t)
		mustWrite(t, filepath.Join(root, "src.txt"), "src\n", 0o644)
		if err := os.MkdirAll(filepath.Join(root, "d", "e"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Join(outside, "e"), 0o755); err != nil {
			t.Fatal(err)
		}
		s, err := New(root)
		if err != nil {
			t.Fatal(err)
		}
		orig := beforePatchMoveNoReplace
		beforePatchMoveNoReplace = func(string, string) {
			beforePatchMoveNoReplace = orig
			if err := os.Rename(filepath.Join(root, "d"), filepath.Join(root, "d-real")); err != nil {
				t.Error(err)
			}
			if err := os.Symlink(outside, filepath.Join(root, "d")); err != nil {
				t.Error(err)
			}
		}
		t.Cleanup(func() { beforePatchMoveNoReplace = orig })
		err = runSandboxTool(t, s, "apply_patch", `{"patch":"*** Begin Patch\n*** Update File: src.txt\n*** Move to: d/e/dst.txt\n@@\n-src\n+moved\n*** End Patch"}`)
		if err == nil {
			t.Fatal("move succeeded after the destination directory became a symlink")
		}
		if entries, _ := os.ReadDir(filepath.Join(outside, "e")); len(entries) != 0 {
			t.Fatalf("move escaped the sandbox: %v", entries)
		}
		// The in-place update lands before the move (patch actions are ordered);
		// the source itself must stay in the sandbox.
		wantContent(t, filepath.Join(root, "src.txt"), "moved\n")
	})
}

// Issue #240: a parent directory swapped for a symlink after the delete was
// validated must not unlink a file outside the sandbox.
func TestApplyPatchDeleteFailsClosedWhenParentSwapped(t *testing.T) {
	forEachAtomicDirImpl(t, func(t *testing.T, _ string) {
		root, outside := evalTempDir(t), evalTempDir(t)
		mustWrite(t, filepath.Join(root, "a", "f.txt"), "inside\n", 0o644)
		mustWrite(t, filepath.Join(outside, "f.txt"), "victim\n", 0o644)
		s, err := New(root)
		if err != nil {
			t.Fatal(err)
		}
		beforeWriteAnchor = func(string) {
			beforeWriteAnchor = func(string) {}
			if err := os.Rename(filepath.Join(root, "a"), filepath.Join(root, "a-real")); err != nil {
				t.Error(err)
			}
			if err := os.Symlink(outside, filepath.Join(root, "a")); err != nil {
				t.Error(err)
			}
		}
		t.Cleanup(func() { beforeWriteAnchor = func(string) {} })
		if err := runSandboxTool(t, s, "apply_patch", `{"patch":"*** Begin Patch\n*** Delete File: a/f.txt\n*** End Patch"}`); err == nil {
			t.Fatal("delete succeeded after the parent directory became a symlink")
		}
		wantContent(t, filepath.Join(outside, "f.txt"), "victim\n")
		wantContent(t, filepath.Join(root, "a-real", "f.txt"), "inside\n")
	})
}

// New directories inherit the parent's mode despite the umask; a directory that
// already exists (here: created concurrently right before mkdir) keeps its mode.
// An overwrite keeps the file mode and replaces the file atomically.
func TestAnchoredWriteModes(t *testing.T) {
	forEachAtomicDirImpl(t, func(t *testing.T, _ string) {
		root := evalTempDir(t)
		if err := os.Chmod(root, 0o777); err != nil {
			t.Fatal(err)
		}
		if err := writeFilePreserveMode(filepath.Join(root, "n1", "n2", "f.txt"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		for _, d := range []string{"n1", filepath.Join("n1", "n2")} {
			if info, err := os.Stat(filepath.Join(root, d)); err != nil || info.Mode().Perm() != 0o777 {
				t.Fatalf("%s mode = %v, %v; want inherited 0777", d, info.Mode().Perm(), err)
			}
		}

		raced := filepath.Join(root, "raced")
		beforeAnchorMkdir = func() {
			beforeAnchorMkdir = func() {}
			if err := os.Mkdir(raced, 0o700); err != nil {
				t.Error(err)
			}
			if err := os.Chmod(raced, 0o700); err != nil {
				t.Error(err)
			}
		}
		t.Cleanup(func() { beforeAnchorMkdir = func() {} })
		if err := writeFilePreserveMode(filepath.Join(raced, "f.txt"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		if info, err := os.Stat(raced); err != nil || info.Mode().Perm() != 0o700 {
			t.Fatalf("existing dir mode = %v, %v; want untouched 0700", info.Mode().Perm(), err)
		}

		f := filepath.Join(root, "keep.txt")
		mustWrite(t, f, "old", 0o600)
		if err := os.Chmod(f, 0o640); err != nil {
			t.Fatal(err)
		}
		before, err := os.Stat(f)
		if err != nil {
			t.Fatal(err)
		}
		if err := writeFilePreserveMode(f, []byte("new"), 0o644); err != nil {
			t.Fatal(err)
		}
		after, err := os.Stat(f)
		if err != nil {
			t.Fatal(err)
		}
		if after.Mode().Perm() != 0o640 {
			t.Fatalf("mode after overwrite = %v; want 0640", after.Mode().Perm())
		}
		if os.SameFile(before, after) {
			t.Fatal("overwrite reused the inode; want temp file + rename")
		}
		wantContent(t, f, "new")
		entries, _ := os.ReadDir(root)
		for _, e := range entries {
			if strings.HasPrefix(e.Name(), ".tmp-") {
				t.Fatalf("temp file left behind: %s", e.Name())
			}
		}
	})
}

// The os.Root link runs in a root at the common ancestor and is verified against
// the pinned directories: a destination swapped after pinning is rolled back.
func TestRootAtomicDirLinkRollsBackSwappedDestination(t *testing.T) {
	root, outside := evalTempDir(t), evalTempDir(t)
	mustWrite(t, filepath.Join(root, "s", "src.txt"), "src", 0o644)
	for _, d := range []string{filepath.Join(root, "d"), filepath.Join(root, "other")} {
		if err := os.Mkdir(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, target := range []string{filepath.Join(root, "other"), outside} {
		from, err := openRootAtomicDir(filepath.Join(root, "s"), false)
		if err != nil {
			t.Fatal(err)
		}
		to, err := openRootAtomicDir(filepath.Join(root, "d"), false)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(filepath.Join(root, "d"), filepath.Join(root, "d-pinned")); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(target, filepath.Join(root, "d")); err != nil {
			t.Fatal(err)
		}
		if err := from.link("src.txt", to, "dst.txt"); err == nil {
			t.Fatalf("link through swapped destination (%s) succeeded", target)
		}
		from.close()
		to.close()
		wantAbsent(t, filepath.Join(target, "dst.txt"))
		wantAbsent(t, filepath.Join(root, "d-pinned", "dst.txt"))
		wantContent(t, filepath.Join(root, "s", "src.txt"), "src")
		if err := os.Remove(filepath.Join(root, "d")); err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(filepath.Join(root, "d-pinned"), filepath.Join(root, "d")); err != nil {
			t.Fatal(err)
		}
	}
}
