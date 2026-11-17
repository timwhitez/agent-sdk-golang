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
	s, _ := tools.Get(deps, context.Background(), Key)
	path := filepath.Join(s.RootDir, "abc.txt")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, args := range []string{
		`{"file_path":"abc.txt","old_string":"","new_string":"X","replace_all":true}`,
		`{"file_path":"abc.txt","old_string":"b","new_string":"b"}`,
	} {
		if _, err := editTool().Execute(context.Background(), args, deps); err == nil {
			t.Fatalf("expected error for %s", args)
		}
		if got, _ := os.ReadFile(path); string(got) != "abc" {
			t.Fatalf("file changed: %q", got)
		}
	}
}
