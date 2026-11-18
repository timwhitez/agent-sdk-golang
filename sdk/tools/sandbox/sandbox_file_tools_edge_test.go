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

func TestWriteToolsFailClosedWhenIntermediateDirBecomesSymlink(t *testing.T) {
	forEachAtomicDirImpl(t, testWriteToolsFailClosedWhenIntermediateDirBecomesSymlink)
}

func testWriteToolsFailClosedWhenIntermediateDirBecomesSymlink(t *testing.T, _ string) {
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

// Issue #268: glob/grep scan-cap metadata keeps scanned_* as file counts and
// reports directories separately as visited_entries.
func TestScanCapMetadataSeparatesFilesFromVisitedEntries(t *testing.T) {
	defer func(g, l int) { grepTraversalBudget, maxGlobScanFiles = g, l }(grepTraversalBudget, maxGlobScanFiles)
	grepTraversalBudget, maxGlobScanFiles = 10, 10
	deps := fakeGitRepo(t)
	s, _ := tools.Get(deps, context.Background(), Key)
	for i := 0; i < 4; i++ { // d0 d1 d2 d3, each with a file => 8 entries, then dirs
		dir := filepath.Join(s.RootDir, "tree", fmt.Sprintf("d%d", i))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for i := 4; i < 12; i++ {
		if err := os.MkdirAll(filepath.Join(s.RootDir, "tree", fmt.Sprintf("d%d", i)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for name, tc := range map[string]struct {
		tool tools.Tool
		args string
	}{
		"glob": {globTool(), `{"pattern":"**/*.nomatch","path":"tree"}`},
		"grep": {grepTool(), `{"pattern":"absent","path":"tree"}`},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := tools.WithToolResultMetadata(context.Background())
			res, err := tc.tool.Execute(ctx, tc.args, deps)
			if err != nil {
				t.Fatal(err)
			}
			if out := res.PlainText(); !strings.Contains(out, "stopped after 10 visited entries at cap 10") || strings.Contains(out, "file candidate") {
				t.Fatalf("unexpected warning: %q", out)
			}
			meta := tools.ToolResultMetadataSnapshot(ctx)
			if meta["visited_entries"] != 10 {
				t.Fatalf("visited_entries = %#v", meta["visited_entries"])
			}
			files := meta["scanned_files"]
			if files == 10 || files == nil || files != meta["scanned_candidates"] {
				t.Fatalf("scanned_files/candidates must count files only, got %#v / %#v", files, meta["scanned_candidates"])
			}
		})
	}
}

// Issue #268: the skipped-directory list is configurable per sandbox.
func TestSkippedSearchDirsConfigurable(t *testing.T) {
	deps := fakeGitRepo(t)
	s, _ := tools.Get(deps, context.Background(), Key)
	if err := os.MkdirAll(filepath.Join(s.RootDir, "vendor"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(s.RootDir, "vendor", "v.go"), []byte("package v\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name         string
		set          func()
		wantGit      bool
		wantVendor   bool
		wantSkipMeta string
	}{
		{"default", func() { s.SetSkippedSearchDirs(nil) }, false, true, ".git"},
		{"override", func() { s.SetSkippedSearchDirs([]string{"vendor"}) }, true, false, "vendor"},
		{"empty skips nothing", func() { s.SetSkippedSearchDirs([]string{}) }, true, true, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.set()
			ctx := tools.WithToolResultMetadata(context.Background())
			run := func(tool tools.Tool, args, want string) bool {
				res, err := tool.Execute(ctx, args, deps)
				if err != nil {
					t.Fatal(err)
				}
				return strings.Contains(res.PlainText(), want)
			}
			if got := run(globTool(), `{"pattern":"**/obj0001"}`, ".git/objects/01/obj0001"); got != tc.wantGit {
				t.Errorf("glob .git searched = %v, want %v", got, tc.wantGit)
			}
			if got := run(globTool(), `{"pattern":"**/v.go"}`, "vendor/v.go"); got != tc.wantVendor {
				t.Errorf("glob vendor searched = %v, want %v", got, tc.wantVendor)
			}
			if got := run(grepTool(), `{"pattern":"needle blob","glob":"**/obj0001","output_mode":"files_with_matches"}`, "obj0001"); got != tc.wantGit {
				t.Errorf("grep .git searched = %v, want %v", got, tc.wantGit)
			}
			if got := run(grepTool(), `{"pattern":"package v","output_mode":"files_with_matches"}`, "vendor/v.go"); got != tc.wantVendor {
				t.Errorf("grep vendor searched = %v, want %v", got, tc.wantVendor)
			}
			res, _ := globTool().Execute(ctx, `{"pattern":"**/*.nomatch"}`, deps)
			meta := tools.ToolResultMetadataSnapshot(ctx)
			samples, _ := meta["skipped_dir_samples"].([]string)
			if tc.wantSkipMeta == "" && len(samples) != 0 || tc.wantSkipMeta != "" && (len(samples) != 1 || samples[0] != tc.wantSkipMeta) {
				t.Errorf("skipped_dir_samples = %v, want [%s]; %s", samples, tc.wantSkipMeta, res.PlainText())
			}
		})
	}
	// Entries with a separator or glob character can never match a base name and are ignored.
	s.SetSkippedSearchDirs([]string{".git/objects", "*", "{src}"})
	if len(s.searchSkipDirs()) != 0 {
		t.Errorf("unmatchable entries kept: %v", s.searchSkipDirs())
	}
	// An explicit path inside a configured-skipped directory is still searched.
	s.SetSkippedSearchDirs([]string{"src"})
	res, err := globTool().Execute(context.Background(), `{"pattern":"**/*.go","path":"src"}`, deps)
	if err != nil || !strings.Contains(res.PlainText(), "src/main.go") {
		t.Fatalf("explicit path in skipped dir not searched: %q err=%v", res.PlainText(), err)
	}
}
