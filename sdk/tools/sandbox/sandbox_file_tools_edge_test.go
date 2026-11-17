package sandbox

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func fakeGitRepo(t *testing.T) *tools.Container {
	root := t.TempDir()
	for i := 0; i < 260; i++ {
		dir := filepath.Join(root, ".git", "objects", fmt.Sprintf("%02x", i%256))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("obj%04d", i)), []byte("needle blob"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(filepath.Join(root, "src"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "src", "main.go"), []byte("package main\n// needle here\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	return deps
}

func TestGrepGlobSkipVCSDirsAndKeepScanBudgetSeparate(t *testing.T) {
	deps := fakeGitRepo(t)
	for name, tc := range map[string]struct {
		tool tools.Tool
		args string
	}{
		"grep":      {grepTool(), `{"pattern":"needle"}`},
		"grep-glob": {grepTool(), `{"pattern":"needle","glob":"**/*.go"}`},
		"glob":      {globTool(), `{"pattern":"**/*.go"}`},
	} {
		t.Run(name, func(t *testing.T) {
			res, err := tc.tool.Execute(context.Background(), tc.args, deps)
			if err != nil {
				t.Fatal(err)
			}
			if out := res.PlainText(); !strings.Contains(out, "src/main.go") || strings.Contains(out, "scan stopped") {
				t.Fatalf("expected src/main.go without scan cap, got %q", out)
			}
		})
	}
	// An explicit .git path is still searchable.
	res, err := grepTool().Execute(context.Background(), `{"pattern":"needle","path":".git","max_results":3}`, deps)
	if err != nil || !strings.Contains(res.PlainText(), ".git/objects") {
		t.Fatalf("expected .git content, got %q err=%v", res.PlainText(), err)
	}
	// An empty result names the skipped directories.
	res, err = grepTool().Execute(context.Background(), `{"pattern":"absent-token"}`, deps)
	if err != nil || !strings.Contains(res.PlainText(), "did not search") {
		t.Fatalf("expected skipped-dir warning, got %q err=%v", res.PlainText(), err)
	}
}

func TestGrepTraversalBudgetStopsOnNonMatchingGlob(t *testing.T) {
	defer func(old int) { grepTraversalBudget = old }(grepTraversalBudget)
	grepTraversalBudget = 10
	deps := fakeGitRepo(t)
	s, _ := tools.Get(deps, context.Background(), Key)
	for i := 0; i < 30; i++ {
		_ = os.WriteFile(filepath.Join(s.RootDir, "src", fmt.Sprintf("z%02d.txt", i)), []byte("x"), 0o644)
	}
	res, err := grepTool().Execute(context.Background(), `{"pattern":"needle","glob":"**/*.rs"}`, deps)
	if err != nil || !strings.Contains(res.PlainText(), "[WARN] grep scan stopped after") {
		t.Fatalf("expected traversal-budget warning, got %q err=%v", res.PlainText(), err)
	}
}

func TestEditRejectsEmptyOrIdenticalOldString(t *testing.T) {
	deps := fakeGitRepo(t)
	// An allow-all confirmer ensures only the new validation can reject the call.
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowAllConfirmer{}, nil })
	s, _ := tools.Get(deps, context.Background(), Key)
	path := filepath.Join(s.RootDir, "abc.txt")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	for args, want := range map[string]string{
		`{"file_path":"abc.txt","old_string":"","new_string":"X","replace_all":true}`: "empty old_string",
		`{"file_path":"abc.txt","old_string":"b","new_string":"b"}`:                   "old_string equals new_string",
	} {
		if _, err := editTool().Execute(context.Background(), args, deps); err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("expected %q for %s, got %v", want, args, err)
		}
		if got, _ := os.ReadFile(path); string(got) != "abc" {
			t.Fatalf("file changed: %q", got)
		}
	}
	// A normal replacement still works with the same wiring.
	if _, err := editTool().Execute(context.Background(), `{"file_path":"abc.txt","old_string":"b","new_string":"B"}`, deps); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); string(got) != "aBc" {
		t.Fatalf("edit did not apply: %q", got)
	}
}

func TestGrepFilesWithMatchesEmptyResultWarnsAboutSkippedDirs(t *testing.T) {
	deps := fakeGitRepo(t)
	res, err := grepTool().Execute(context.Background(), `{"pattern":"absent-token","output_mode":"files_with_matches"}`, deps)
	if err != nil || !strings.Contains(res.PlainText(), "did not search") {
		t.Fatalf("expected skipped-dir warning, got %q err=%v", res.PlainText(), err)
	}
}

func TestTraversalBudgetCountsDirectories(t *testing.T) {
	defer func(g, l int) { grepTraversalBudget, maxGlobScanFiles = g, l }(grepTraversalBudget, maxGlobScanFiles)
	grepTraversalBudget, maxGlobScanFiles = 10, 10
	deps := fakeGitRepo(t)
	s, _ := tools.Get(deps, context.Background(), Key)
	for i := 0; i < 30; i++ {
		if err := os.MkdirAll(filepath.Join(s.RootDir, "empty", fmt.Sprintf("d%02d", i)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	res, err := grepTool().Execute(context.Background(), `{"pattern":"needle","path":"empty"}`, deps)
	if err != nil || !strings.Contains(res.PlainText(), "stopped after 10 visited entries at cap 10") {
		t.Fatalf("grep: expected traversal warning, got %q err=%v", res.PlainText(), err)
	}
	res, err = globTool().Execute(context.Background(), `{"pattern":"**/*.go","path":"empty"}`, deps)
	if err != nil || !strings.Contains(res.PlainText(), "[WARN] glob scan stopped after") {
		t.Fatalf("glob: expected traversal warning, got %q err=%v", res.PlainText(), err)
	}
}

// TestWriteWithExecuteOnlyAncestor guards against the dirfd walk requiring read
// permission on ancestors: a path-based open only needs search (x).
func TestWriteWithExecuteOnlyAncestor(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("dirfd-anchored writes are Linux-only")
	}
	if os.Getenv("SANDBOX_EXEC_ONLY_CHILD") != "" {
		root := os.Getenv("SANDBOX_EXEC_ONLY_CHILD")
		if err := writeFilePreserveMode(filepath.Join(root, "sub", "f.txt"), []byte("ok"), 0o644); err != nil {
			t.Fatalf("write under execute-only ancestor: %v", err)
		}
		return
	}
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
	cmd.Env = append(os.Environ(), "SANDBOX_EXEC_ONLY_CHILD="+root)
	cmd.SysProcAttr = &syscall.SysProcAttr{Credential: &syscall.Credential{Uid: 65534, Gid: 65534}}
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("unprivileged write failed: %v\n%s", err, out)
	}
	if got, _ := os.ReadFile(filepath.Join(root, "sub", "f.txt")); string(got) != "ok" {
		t.Fatalf("file not written: %q", got)
	}
}

func TestWriteToolsFailClosedWhenIntermediateDirBecomesSymlink(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("dirfd-anchored writes are Linux-only")
	}
	for _, tc := range []struct{ name, args string }{
		{"write", `{"file_path":"a/b/c.txt","content":"after\n"}`},
		{"edit", `{"file_path":"a/b/c.txt","old_string":"before","new_string":"after"}`},
		{"multiedit", `{"file_path":"a/b/c.txt","edits":[{"old_string":"before","new_string":"after"}]}`},
		{"apply_patch", `{"patch":"*** Begin Patch\n*** Update File: a/b/c.txt\n@@\n-before\n+after\n*** End Patch"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root, outside := t.TempDir(), t.TempDir()
			if err := os.MkdirAll(filepath.Join(root, "a", "b"), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(filepath.Join(outside, "b"), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, "a", "b", "c.txt"), []byte("before\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			s, err := New(root)
			if err != nil {
				t.Fatal(err)
			}
			deps := tools.NewContainer()
			tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
			tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowAllConfirmer{}, nil })

			// Swap root/a for a symlink after validation, before the write opens its directory.
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

			var execErr error
			for _, tool := range Tools() {
				if tool.Name == tc.name {
					_, execErr = tool.Execute(context.Background(), tc.args, deps)
				}
			}
			if execErr == nil {
				t.Fatal("expected the write to fail closed")
			}
			if entries, _ := os.ReadDir(filepath.Join(outside, "b")); len(entries) != 0 {
				t.Fatalf("write escaped the sandbox: %v", entries)
			}
			if got, _ := os.ReadFile(filepath.Join(root, "a-real", "b", "c.txt")); string(got) != "before\n" {
				t.Fatalf("original changed: %q", got)
			}
		})
	}
}

type allowAllConfirmer struct{}

func (allowAllConfirmer) Confirm(context.Context, string, string) (bool, error) { return true, nil }
