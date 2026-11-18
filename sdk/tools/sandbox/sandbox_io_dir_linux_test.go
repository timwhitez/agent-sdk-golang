package sandbox

import (
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// runAsNobody re-runs test (selected by name) as uid/gid 65534 with impl and
// arg in the environment, since root bypasses permission bits. Without root it
// runs child in-process.
func runAsNobody(t *testing.T, test, impl, base, arg string, child func(t *testing.T, impl, arg string)) {
	t.Helper()
	if os.Geteuid() != 0 {
		child(t, impl, arg)
		return
	}
	exe, err := os.Executable()
	if err != nil {
		t.Skip(err)
	}
	data, err := os.ReadFile(exe)
	if err != nil {
		t.Skip(err)
	}
	bin := filepath.Join(base, "sandbox.test")
	if err := os.WriteFile(bin, data, 0o755); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bin, "-test.run=^"+test+"$", "-test.v")
	cmd.Env = append(os.Environ(), "SANDBOX_NOBODY_ARG="+arg, "SANDBOX_NOBODY_IMPL="+impl)
	cmd.SysProcAttr = &syscall.SysProcAttr{Credential: &syscall.Credential{Uid: 65534, Gid: 65534}}
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("unprivileged child failed: %v\n%s", err, out)
	}
}

// nobodyChild runs child when this process is the re-executed unprivileged copy.
func nobodyChild(t *testing.T, child func(t *testing.T, impl, arg string)) bool {
	arg := os.Getenv("SANDBOX_NOBODY_ARG")
	if arg == "" {
		return false
	}
	impl := os.Getenv("SANDBOX_NOBODY_IMPL")
	useAtomicDirImpl(t, impl)
	child(t, impl, arg)
	return true
}

// TestWriteWithExecuteOnlyAncestor: an ancestor ABOVE the sandbox root that is
// only searchable (x) must not break writes. The dirfd walk opens components
// with O_PATH; the os.Root walk starts at the sandbox root.
func TestWriteWithExecuteOnlyAncestor(t *testing.T) {
	child := func(t *testing.T, _, root string) {
		s, err := New(root)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.writeFilePreserveMode(filepath.Join(root, "sub", "f.txt"), []byte("ok"), 0o644); err != nil {
			t.Fatalf("write under execute-only ancestor: %v", err)
		}
	}
	if nobodyChild(t, child) {
		return
	}
	forEachAtomicDirImpl(t, func(t *testing.T, impl string) {
		base, err := os.MkdirTemp("", "execonly")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Chmod(filepath.Join(base, "x"), 0o755); _ = os.RemoveAll(base) })
		root := filepath.Join(base, "x", "root")
		if err := os.MkdirAll(root, 0o777); err != nil {
			t.Fatal(err)
		}
		mode := os.FileMode(0o711) // root re-runs as nobody: others get x only
		if os.Geteuid() != 0 {
			mode = 0o311
		}
		for dir, m := range map[string]os.FileMode{base: 0o755, root: 0o777, filepath.Join(base, "x"): mode} {
			if err := os.Chmod(dir, m); err != nil {
				t.Fatal(err)
			}
		}
		runAsNobody(t, "TestWriteWithExecuteOnlyAncestor", impl, base, root, child)
		if got, _ := os.ReadFile(filepath.Join(root, "sub", "f.txt")); string(got) != "ok" {
			t.Fatalf("file not written: %q", got)
		}
	})
}

// TestAnchoredWalkBelowUnreadableInRootDir: the workspace actor (same uid) can
// make a directory INSIDE the root write+search only and then swap it for a
// symlink. No implementation may fall back to a path-based step there.
func TestAnchoredWalkBelowUnreadableInRootDir(t *testing.T) {
	child := func(t *testing.T, impl, base string) {
		ws, outside := filepath.Join(base, "ws"), filepath.Join(base, "outside")
		for _, d := range []string{filepath.Join(ws, "a"), filepath.Join(ws, "u"), filepath.Join(outside, "b")} {
			if err := os.MkdirAll(d, 0o755); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.Symlink(outside, filepath.Join(ws, "u", "link")); err != nil {
			t.Fatal(err)
		}
		s, err := New(ws)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range []string{"a", "u"} {
			if err := os.Chmod(filepath.Join(ws, d), 0o311); err != nil {
				t.Fatal(err)
			}
		}
		// Direct write into the unreadable directory: the O_PATH walk only needs
		// search permission; os.Root cannot pin it and fails closed.
		errDirect := runSandboxTool(t, s, "write", `{"file_path":"a/direct.txt","content":"x"}`)
		if impl == "root" && errDirect == nil {
			t.Error("os.Root: write into a write+search-only directory succeeded; want fail closed")
		}
		if impl != "root" && errDirect != nil {
			t.Errorf("dirfd: write into a write+search-only directory: %v", errDirect)
		}

		// Hook-free: a symlink below the unreadable directory is refused by the
		// layer itself (the resolver is bypassed).
		if err := s.writeFilePreserveMode(filepath.Join(ws, "u", "link", "b", "x.txt"), []byte("x"), 0o644); err == nil {
			t.Error("write through a symlink below an unreadable in-root directory succeeded")
		}

		// Swap ws/a for a symlink after validation, while the walk is below it.
		beforeAnchorMkdir = func() {
			beforeAnchorMkdir = func() {}
			if err := os.Rename(filepath.Join(ws, "a"), filepath.Join(ws, "a-real")); err != nil {
				t.Error(err)
			}
			if err := os.Symlink(outside, filepath.Join(ws, "a")); err != nil {
				t.Error(err)
			}
		}
		t.Cleanup(func() { beforeAnchorMkdir = func() {} })
		errSwap := runSandboxTool(t, s, "write", `{"file_path":"a/b/c.txt","content":"escaped"}`)
		for _, p := range []string{filepath.Join(outside, "b", "c.txt"), filepath.Join(outside, "b", "x.txt")} {
			if _, err := os.Lstat(p); err == nil {
				t.Errorf("ESCAPE: %s written (swap tool err=%v)", p, errSwap)
			}
		}
	}
	if nobodyChild(t, child) {
		return
	}
	forEachAtomicDirImpl(t, func(t *testing.T, impl string) {
		base, err := os.MkdirTemp("", "unreadable")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(base, 0o777); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			_ = exec.Command("chmod", "-R", "u+rwx", base).Run()
			_ = os.RemoveAll(base)
		})
		runAsNobody(t, "TestAnchoredWalkBelowUnreadableInRootDir", impl, base, base, child)
	})
}

// A FIFO swapped in for the source between the regular-file lstat and the open
// of an os.Root cross-directory copy must not block the move.
func TestRootCopyRefusesFIFOSwappedInForSource(t *testing.T) {
	useAtomicDirImpl(t, "root")
	home := evalTempDir(t)
	ws, ext := filepath.Join(home, "ws"), filepath.Join(home, "ext")
	mustWrite(t, filepath.Join(ws, "src"), "src", 0o644)
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
	beforeRootCopyOpen = func(string) {
		beforeRootCopyOpen = func(string) {}
		if err := os.Rename(filepath.Join(ws, "src"), filepath.Join(ws, "src-real")); err != nil {
			t.Error(err)
		}
		if err := syscall.Mkfifo(filepath.Join(ws, "src"), 0o644); err != nil {
			t.Error(err)
		}
	}
	t.Cleanup(func() { beforeRootCopyOpen = func(string) {} })
	done := make(chan error, 1)
	go func() { done <- s.moveRegularFileNoReplace(filepath.Join(ws, "src"), filepath.Join(ext, "dst")) }()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("move of a swapped-in FIFO succeeded")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("move blocked on a FIFO source")
	}
	if info, err := os.Lstat(filepath.Join(ws, "src")); err != nil || info.Mode()&os.ModeNamedPipe == 0 {
		t.Fatalf("source name changed: %v %v", info, err)
	}
	wantAbsent(t, filepath.Join(ext, "dst"))
	wantContent(t, filepath.Join(ws, "src-real"), "src")
}
