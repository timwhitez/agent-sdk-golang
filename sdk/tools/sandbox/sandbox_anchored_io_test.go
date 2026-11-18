package sandbox

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// atomicDirImpls lists every anchored-directory implementation runnable here:
// the platform default (O_PATH dirfd on Linux) and the portable os.Root one
// that is the default on darwin, windows and other unix systems.
var atomicDirImpls = map[string]func(string, string, bool) (atomicDir, error){
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
		s, err := New(root)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, filepath.Join(root, "dirlink")); err != nil {
			t.Fatal(err)
		}
		if err := s.writeFilePreserveMode(filepath.Join(root, "dirlink", "f.txt"), []byte("x"), 0o644); err == nil {
			t.Fatal("write through symlinked intermediate directory succeeded")
		}
		wantAbsent(t, filepath.Join(outside, "f.txt"))

		victim := filepath.Join(outside, "victim.txt")
		mustWrite(t, victim, "victim", 0o644)
		if err := os.Symlink(victim, filepath.Join(root, "leaf.txt")); err != nil {
			t.Fatal(err)
		}
		err = s.writeFilePreserveMode(filepath.Join(root, "leaf.txt"), []byte("x"), 0o644)
		if _, ok := err.(*SecurityError); !ok {
			t.Fatalf("write to final-component symlink: err = %v, want SecurityError", err)
		}
		wantContent(t, victim, "victim")

		if err := s.removeFileAnchored(filepath.Join(root, "dirlink", "victim.txt")); err == nil {
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
		s, err := New(root)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.writeFilePreserveMode(filepath.Join(root, "n1", "n2", "f.txt"), []byte("x"), 0o644); err != nil {
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
		if err := s.writeFilePreserveMode(filepath.Join(raced, "f.txt"), []byte("x"), 0o644); err != nil {
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
		if err := s.writeFilePreserveMode(f, []byte("new"), 0o644); err != nil {
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

// Review probe for #272: after the destination directory of a cross-root move
// is pinned, the attacker moves it aside, symlinks its name to a directory
// outside every allowed root and plants a hard link of the source under the
// destination name in the pinned directory. Nothing may land outside, and the
// source must stay.
func TestApplyPatchCrossRootMoveHardlinkDecoy(t *testing.T) {
	forEachAtomicDirImpl(t, func(t *testing.T, _ string) {
		home := evalTempDir(t)
		ws, ext, ssh := filepath.Join(home, "ws"), filepath.Join(home, "ext"), filepath.Join(home, ".ssh")
		for _, d := range []string{ws, filepath.Join(ext, "d"), ssh} {
			if err := os.MkdirAll(d, 0o755); err != nil {
				t.Fatal(err)
			}
		}
		mustWrite(t, filepath.Join(ws, "src.txt"), "ssh-ed25519 ATTACKER\n", 0o644)
		s, err := New(ws)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := s.AllowExternalDirectory(ext); err != nil {
			t.Fatal(err)
		}
		impl := openAtomicDir
		fired := false
		openAtomicDir = func(anchor, dir string, create bool) (atomicDir, error) {
			ad, err := impl(anchor, dir, create)
			if !fired && err == nil && dir == filepath.Join(ext, "d") {
				fired = true
				for _, step := range []error{
					os.Rename(filepath.Join(ext, "d"), filepath.Join(ext, "d-real")),
					os.Symlink("../.ssh", filepath.Join(ext, "d")),
					os.Link(filepath.Join(ws, "src.txt"), filepath.Join(ext, "d-real", "authorized_keys")),
				} {
					if step != nil {
						t.Error(step)
					}
				}
			}
			return ad, err
		}
		patch := fmt.Sprintf(`{"patch":"*** Begin Patch\n*** Update File: src.txt\n*** Move to: %s\n@@\n-ssh-ed25519 ATTACKER\n+ssh-ed25519 ATTACKER\n+x\n*** End Patch"}`, filepath.Join(ext, "d", "authorized_keys"))
		toolErr := runSandboxTool(t, s, "apply_patch", patch)
		if !fired {
			t.Fatal("attacker hook did not run")
		}
		if toolErr == nil {
			t.Error("cross-root move onto a planted decoy succeeded")
		}
		wantAbsent(t, filepath.Join(ssh, "authorized_keys"))
		if _, err := os.Stat(filepath.Join(ws, "src.txt")); err != nil {
			t.Fatalf("source removed: %v", err)
		}
	})
}

func dirHasTemp(t *testing.T, dir string) bool {
	t.Helper()
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".tmp-") {
			return true
		}
	}
	return false
}

// The move primitive never replaces, keeps content and mode across roots,
// leaves no temp file, and on a failed copy keeps the source.
func TestMoveRegularFileNoReplace(t *testing.T) {
	forEachAtomicDirImpl(t, func(t *testing.T, impl string) {
		home := evalTempDir(t)
		ws, ext := filepath.Join(home, "ws"), filepath.Join(home, "ext")
		mustWrite(t, filepath.Join(ws, "a", "src"), "src", 0o644)
		mustWrite(t, filepath.Join(ws, "b", "dst"), "dst", 0o644)
		if err := os.MkdirAll(ext, 0o755); err != nil {
			t.Fatal(err)
		}
		s, err := New(ws)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := s.AllowExternalDirectory(ext); err != nil {
			t.Fatal(err)
		}

		if err := s.moveRegularFileNoReplace(filepath.Join(ws, "a", "src"), filepath.Join(ws, "b", "dst")); err == nil {
			t.Error("move onto an existing destination succeeded")
		}
		wantContent(t, filepath.Join(ws, "b", "dst"), "dst")
		wantContent(t, filepath.Join(ws, "a", "src"), "src")

		if err := s.moveRegularFileNoReplace(filepath.Join(ws, "a", "src"), filepath.Join(ws, "a", "src2")); err != nil {
			t.Fatalf("same-directory move: %v", err)
		}
		wantContent(t, filepath.Join(ws, "a", "src2"), "src")
		wantAbsent(t, filepath.Join(ws, "a", "src"))

		if impl == "root" {
			orig := writeFileBytes
			writeFileBytes = func(*os.File, []byte) (int, error) { return 0, os.ErrInvalid }
			err := s.moveRegularFileNoReplace(filepath.Join(ws, "a", "src2"), filepath.Join(ext, "n", "moved"))
			writeFileBytes = orig
			if err == nil {
				t.Error("cross-root move with a failing copy succeeded")
			}
			wantContent(t, filepath.Join(ws, "a", "src2"), "src")
			wantAbsent(t, filepath.Join(ext, "n", "moved"))
			if dirHasTemp(t, filepath.Join(ext, "n")) {
				t.Error("temp file left after a failed copy")
			}
		}

		if err := os.Chmod(filepath.Join(ws, "a", "src2"), 0o640); err != nil {
			t.Fatal(err)
		}
		if err := s.moveRegularFileNoReplace(filepath.Join(ws, "a", "src2"), filepath.Join(ext, "n", "moved")); err != nil {
			t.Fatalf("cross-root move: %v", err)
		}
		wantContent(t, filepath.Join(ext, "n", "moved"), "src")
		wantAbsent(t, filepath.Join(ws, "a", "src2"))
		if info, err := os.Stat(filepath.Join(ext, "n", "moved")); err != nil || info.Mode().Perm() != 0o640 {
			t.Fatalf("moved mode = %v, %v; want 0640", info.Mode().Perm(), err)
		}
		if dirHasTemp(t, filepath.Join(ext, "n")) {
			t.Error("temp file left after a cross-root move")
		}

		// A symlink source: the dirfd hard link keeps it a symlink (never
		// followed); os.Root cannot copy a symlink and refuses it.
		if err := os.Symlink("/etc/passwd", filepath.Join(ws, "a", "lnk")); err != nil {
			t.Fatal(err)
		}
		err = s.moveRegularFileNoReplace(filepath.Join(ws, "a", "lnk"), filepath.Join(ws, "c", "lnk2"))
		info, lerr := os.Lstat(filepath.Join(ws, "c", "lnk2"))
		switch {
		case impl == "root" && (err == nil || lerr == nil):
			t.Errorf("os.Root moved a symlink source: err=%v", err)
		case lerr == nil && info.Mode()&os.ModeSymlink == 0:
			t.Error("symlink source was followed")
		}
	})
}

// Both implementations refuse to unlink a directory.
func TestRemoveFileAnchoredRefusesDirectory(t *testing.T) {
	forEachAtomicDirImpl(t, func(t *testing.T, _ string) {
		root := evalTempDir(t)
		s, err := New(root)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(filepath.Join(root, "empty"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := s.removeFileAnchored(filepath.Join(root, "empty")); err == nil {
			t.Error("removed a directory")
		}
		if info, err := os.Stat(filepath.Join(root, "empty")); err != nil || !info.IsDir() {
			t.Fatalf("directory gone: %v", err)
		}
	})
}
