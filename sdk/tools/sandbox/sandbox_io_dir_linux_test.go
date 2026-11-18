package sandbox

import (
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
)

// TestWriteWithExecuteOnlyAncestor guards against the dirfd walk requiring read
// permission on ancestors: a path-based open only needs search (x).
// The os.Root implementation needs read access to open a directory, so it
// walks path-based below a non-readable ancestor.
func TestWriteWithExecuteOnlyAncestor(t *testing.T) {
	if os.Getenv("SANDBOX_EXEC_ONLY_CHILD") != "" {
		useAtomicDirImpl(t, os.Getenv("SANDBOX_EXEC_ONLY_IMPL"))
		root := os.Getenv("SANDBOX_EXEC_ONLY_CHILD")
		if err := writeFilePreserveMode(filepath.Join(root, "sub", "f.txt"), []byte("ok"), 0o644); err != nil {
			t.Fatalf("write under execute-only ancestor: %v", err)
		}
		return
	}
	forEachAtomicDirImpl(t, testWriteWithExecuteOnlyAncestor)
}

func testWriteWithExecuteOnlyAncestor(t *testing.T, impl string) {
	base, err := os.MkdirTemp("", "execonly")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(filepath.Join(base, "x"), 0o755); _ = os.RemoveAll(base) })
	root := filepath.Join(base, "x", "root")
	if err := os.MkdirAll(root, 0o777); err != nil {
		t.Fatal(err)
	}
	for dir, mode := range map[string]os.FileMode{base: 0o755, root: 0o777, filepath.Join(base, "x"): 0o711} {
		if err := os.Chmod(dir, mode); err != nil {
			t.Fatal(err)
		}
	}
	if os.Geteuid() != 0 {
		if err := os.Chmod(filepath.Join(base, "x"), 0o311); err != nil {
			t.Fatal(err)
		}
		if err := writeFilePreserveMode(filepath.Join(root, "sub", "f.txt"), []byte("ok"), 0o644); err != nil {
			t.Fatalf("write under execute-only ancestor: %v", err)
		}
		return
	}
	// Root bypasses permission bits, so re-run this test as an unprivileged user.
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
	cmd := exec.Command(bin, "-test.run=^TestWriteWithExecuteOnlyAncestor$")
	cmd.Env = append(os.Environ(), "SANDBOX_EXEC_ONLY_CHILD="+root, "SANDBOX_EXEC_ONLY_IMPL="+impl)
	cmd.SysProcAttr = &syscall.SysProcAttr{Credential: &syscall.Credential{Uid: 65534, Gid: 65534}}
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("unprivileged write failed: %v\n%s", err, out)
	}
	if got, _ := os.ReadFile(filepath.Join(root, "sub", "f.txt")); string(got) != "ok" {
		t.Fatalf("file not written: %q", got)
	}
}
