package sandbox

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestAuditReadOnlyToolsHonorEndedContext(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte("found\n"), 0600); err != nil {
		t.Fatal(err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	// Resolve before cancellation so the failure cannot be assigned to a custom provider.
	if _, err := tools.Get(deps, t.Context(), Key); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		tool tools.Tool
		args string
	}{
		{readTool(), `{"file_path":"a.txt"}`},
		{lsTool(), `{}`},
		{grepTool(), `{"pattern":"found"}`},
		{globTool(), `{"pattern":"*.txt"}`},
	} {
		t.Run(tc.tool.Name, func(t *testing.T) {
			for _, ended := range []struct {
				name       string
				newContext func() (context.Context, context.CancelFunc)
				want       error
			}{
				{"canceled", func() (context.Context, context.CancelFunc) {
					ctx, cancel := context.WithCancel(t.Context())
					cancel()
					return ctx, cancel
				}, context.Canceled},
				{"expired", func() (context.Context, context.CancelFunc) {
					return context.WithDeadline(t.Context(), time.Unix(1, 0))
				}, context.DeadlineExceeded},
			} {
				t.Run(ended.name, func(t *testing.T) {
					ctx, cancel := ended.newContext()
					defer cancel()
					out, err := tc.tool.Execute(ctx, tc.args, deps)
					if !errors.Is(err, ended.want) {
						t.Fatalf("ended context: result=%q err=%v; want %v", out.PlainText(), err, ended.want)
					}
				})
			}
		})
	}
}

func TestReadOnlyToolsHonorCancellationDuringDependencyResolution(t *testing.T) {
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		tool tools.Tool
		args string
	}{
		{readTool(), `{"file_path":"missing.txt"}`},
		{lsTool(), `{}`},
		{grepTool(), `{"pattern":"found"}`},
		{globTool(), `{"pattern":"*.txt"}`},
	} {
		t.Run(tc.tool.Name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			deps := tools.NewContainer()
			tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) {
				cancel()
				return s, nil
			})
			out, err := tc.tool.Execute(ctx, tc.args, deps)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled during provider: result=%q err=%v", out.PlainText(), err)
			}
		})
	}
}

func TestAuditGrepCancellationStopsFurtherFiles(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"a.txt", "b.txt", "c.txt"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte("found\n"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	original := grepReadSample
	t.Cleanup(func() { grepReadSample = original })
	opened := 0
	grepReadSample = func(file *os.File, limit int) ([]byte, error) {
		opened++
		result, err := original(file, limit)
		cancel()
		return result, err
	}
	out, err := grepTool().Execute(ctx, `{"pattern":"found","output_mode":"count"}`, deps)
	if !errors.Is(err, context.Canceled) || opened != 1 {
		t.Fatalf("canceled after first file: opened=%d result=%q err=%v", opened, out.PlainText(), err)
	}
}

func TestGlobCancellationStopsTraversalAndPreventsSuccess(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"a.txt", "b.txt", "c.txt"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte("found\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	original := globWalkDir
	t.Cleanup(func() { globWalkDir = original })
	for _, stage := range []string{"first_file", "after_walk"} {
		t.Run(stage, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			visited := 0
			globWalkDir = func(root string, fn fs.WalkDirFunc) error {
				err := original(root, func(path string, d os.DirEntry, err error) error {
					if d != nil && !d.IsDir() {
						visited++
						if stage == "first_file" {
							cancel()
						}
					}
					return fn(path, d, err)
				})
				if stage == "after_walk" {
					cancel()
				}
				return err
			}
			out, err := globTool().Execute(ctx, `{"pattern":"*.txt"}`, deps)
			if !errors.Is(err, context.Canceled) || (stage == "first_file" && visited != 1) {
				t.Fatalf("canceled at %s: visited=%d result=%q err=%v", stage, visited, out.PlainText(), err)
			}
		})
	}
}
