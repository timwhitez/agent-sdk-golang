package sandbox

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
