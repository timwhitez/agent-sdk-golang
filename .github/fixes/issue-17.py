from pathlib import Path

path = Path("sdk/tools/sandbox/sandbox_glob.go")
text = path.read_text()
anchor = '''// globSymlinkWarning creates a warning for paths skipped due to being symbolic links.
func globSymlinkWarning(diag globSkippedDiagnostics) string {
'''
insert = '''// globWalkWarning creates a warning for paths skipped because directory traversal failed.
func globWalkWarning(diag globSkippedDiagnostics) string {
\tif diag.count == 0 {
\t\treturn ""
\t}
\tsummary := ""
\tif len(diag.samples) == 0 {
\t\tsummary = fmt.Sprintf("glob skipped %d path(s) because directory traversal failed", diag.count)
\t} else {
\t\tsummary = fmt.Sprintf(
\t\t\t"glob skipped %d path(s) because directory traversal failed: %s",
\t\t\tdiag.count,
\t\t\tstrings.Join(diag.samples, ", "),
\t\t)
\t}
\treturn formatWarningDiagnostic(
\t\tsummary,
\t\t"Review traversal errors and directory permissions, then rerun glob for complete results.",
\t)
}

// globSymlinkWarning creates a warning for paths skipped due to being symbolic links.
func globSymlinkWarning(diag globSkippedDiagnostics) string {
'''
if text.count(anchor) != 1:
    raise SystemExit(f"warning anchor count={text.count(anchor)}")
text = text.replace(anchor, insert)
old_scan = '''\t\tskippedStat := globSkippedDiagnostics{}
\t\tskippedSymlink := globSkippedDiagnostics{}
\t\twalkErr := globWalkDir(base, func(path string, d os.DirEntry, walkErr error) error {
\t\t\tif walkErr != nil {
\t\t\t\tif d != nil && d.IsDir() {
\t\t\t\t\treturn filepath.SkipDir
\t\t\t\t}
\t\t\t\treturn nil
\t\t\t}
'''
new_scan = '''\t\tskippedStat := globSkippedDiagnostics{}
\t\tskippedSymlink := globSkippedDiagnostics{}
\t\tskippedWalk := globSkippedDiagnostics{}
\t\twalkErr := globWalkDir(base, func(path string, d os.DirEntry, walkErr error) error {
\t\t\tif walkErr != nil {
\t\t\t\tdisplayPath := resultPathForDisplay(s, path)
\t\t\t\tif pathsEqual(filepath.Clean(path), filepath.Clean(base)) {
\t\t\t\t\treturn fmt.Errorf("cannot inspect glob root %s: %w", displayPath, walkErr)
\t\t\t\t}
\t\t\t\tskippedWalk.add(displayPath)
\t\t\t\tif d != nil && d.IsDir() {
\t\t\t\t\treturn filepath.SkipDir
\t\t\t\t}
\t\t\t\treturn nil
\t\t\t}
'''
if text.count(old_scan) != 1:
    raise SystemExit(f"scan anchor count={text.count(old_scan)}")
text = text.replace(old_scan, new_scan)
old_diag = '''\t\tstatWarning := globSkippedWarning(skippedStat)
\t\tsymlinkWarning := globSymlinkWarning(skippedSymlink)
\t\tscanWarning := ""
'''
new_diag = '''\t\tstatWarning := globSkippedWarning(skippedStat)
\t\tsymlinkWarning := globSymlinkWarning(skippedSymlink)
\t\twalkWarning := globWalkWarning(skippedWalk)
\t\tscanWarning := ""
'''
if text.count(old_diag) != 1:
    raise SystemExit(f"diagnostic anchor count={text.count(old_diag)}")
text = text.replace(old_diag, new_diag)
old_reason = '''\t\tskippedCount := skippedStat.count + skippedSymlink.count
\t\tskippedReason := ""
\t\tswitch {
\t\tcase skippedStat.count > 0 && skippedSymlink.count > 0:
\t\t\tskippedReason = "multiple"
\t\tcase skippedStat.count > 0:
\t\t\tskippedReason = "stat_error"
\t\tcase skippedSymlink.count > 0:
\t\t\tskippedReason = "symlink_target"
\t\t}
'''
new_reason = '''\t\tskippedCount := skippedStat.count + skippedSymlink.count + skippedWalk.count
\t\tskippedReason := ""
\t\treasonKinds := 0
\t\tif skippedStat.count > 0 {
\t\t\treasonKinds++
\t\t\tskippedReason = "stat_error"
\t\t}
\t\tif skippedSymlink.count > 0 {
\t\t\treasonKinds++
\t\t\tskippedReason = "symlink_target"
\t\t}
\t\tif skippedWalk.count > 0 {
\t\t\treasonKinds++
\t\t\tskippedReason = "walk_error"
\t\t}
\t\tif reasonKinds > 1 {
\t\t\tskippedReason = "multiple"
\t\t}
'''
if text.count(old_reason) != 1:
    raise SystemExit(f"reason anchor count={text.count(old_reason)}")
text = text.replace(old_reason, new_reason)
old_meta = '''\t\tif skippedSymlink.count > 0 {
\t\t\tmeta["skipped_symlink_paths"] = skippedSymlink.count
\t\t\tmeta["skipped_symlink_samples"] = append([]string(nil), skippedSymlink.samples...)
\t\t}
\t\tif scanTruncated {
'''
new_meta = '''\t\tif skippedSymlink.count > 0 {
\t\t\tmeta["skipped_symlink_paths"] = skippedSymlink.count
\t\t\tmeta["skipped_symlink_samples"] = append([]string(nil), skippedSymlink.samples...)
\t\t}
\t\tif skippedWalk.count > 0 {
\t\t\tmeta["skipped_walk_paths"] = skippedWalk.count
\t\t\tmeta["skipped_walk_samples"] = append([]string(nil), skippedWalk.samples...)
\t\t}
\t\tif scanTruncated {
'''
if text.count(old_meta) != 1:
    raise SystemExit(f"metadata anchor count={text.count(old_meta)}")
text = text.replace(old_meta, new_meta)
text = text.replace('appendGlobWarning("No files match pattern: "+a.Pattern, statWarning, symlinkWarning, scanWarning)', 'appendGlobWarning("No files match pattern: "+a.Pattern, statWarning, symlinkWarning, walkWarning, scanWarning)')
text = text.replace('appendGlobWarning(fmt.Sprintf("%s\\n%s", header, strings.Join(files, "\\n")), statWarning, symlinkWarning, scanWarning)', 'appendGlobWarning(fmt.Sprintf("%s\\n%s", header, strings.Join(files, "\\n")), statWarning, symlinkWarning, walkWarning, scanWarning)')
path.write_text(text)

Path("sdk/tools/sandbox/sandbox_glob_walk_error_test.go").write_text(r'''package sandbox

import (
	"context"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func globWalkTestDeps(t *testing.T) (*Sandbox, *tools.Container) {
	t.Helper()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	return s, deps
}

func TestGlobReportsDescendantWalkFailure(t *testing.T) {
	s, deps := globWalkTestDeps(t)
	original := globWalkDir
	t.Cleanup(func() { globWalkDir = original })
	globWalkDir = func(root string, fn fs.WalkDirFunc) error {
		return fn(filepath.Join(root, "unreadable"), nil, fs.ErrPermission)
	}

	ctx := tools.WithToolResultMetadata(context.Background())
	out, err := globTool().Execute(ctx, `{"pattern":"**/*.go"}`, deps)
	if err != nil {
		t.Fatalf("glob Execute() error = %v", err)
	}
	if !strings.Contains(out.PlainText(), "[WARN]") || !strings.Contains(out.PlainText(), "directory traversal failed") {
		t.Fatalf("glob output did not disclose incomplete traversal: %q", out.PlainText())
	}
	meta := tools.ToolResultMetadataSnapshot(ctx)
	if got := meta["warning_kind"]; got != partialScanWarningKind {
		t.Fatalf("warning_kind = %v, want %s", got, partialScanWarningKind)
	}
	if got := meta["skipped_count"]; got != 1 {
		t.Fatalf("skipped_count = %v, want 1", got)
	}
	if got := meta["skipped_reason"]; got != "walk_error" {
		t.Fatalf("skipped_reason = %v, want walk_error", got)
	}
	if got := meta["skipped_walk_paths"]; got != 1 {
		t.Fatalf("skipped_walk_paths = %v, want 1", got)
	}
	_ = s
}

func TestGlobFailsWhenRequestedRootCannotBeInspected(t *testing.T) {
	_, deps := globWalkTestDeps(t)
	original := globWalkDir
	t.Cleanup(func() { globWalkDir = original })
	globWalkDir = func(root string, fn fs.WalkDirFunc) error {
		return fn(root, nil, fs.ErrPermission)
	}

	_, err := globTool().Execute(context.Background(), `{"pattern":"**/*.go"}`, deps)
	if err == nil || !strings.Contains(err.Error(), "cannot inspect glob root") {
		t.Fatalf("glob Execute() error = %v, want root inspection failure", err)
	}
}
''')
