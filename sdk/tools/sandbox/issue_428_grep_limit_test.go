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

func TestIssue428GrepContentLimitReportsIncompleteScan(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte("needle one\nneedle two\nneedle three\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "b.txt"), []byte("needle four\n"), 0600); err != nil {
		t.Fatal(err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	for _, mode := range []string{"content", "files_with_matches", "count"} {
		t.Run(mode, func(t *testing.T) {
			ctx := tools.WithToolResultMetadata(context.Background())
			result, err := grepTool().Execute(ctx, fmt.Sprintf(`{"pattern":"needle","output_mode":%q,"max_results":1}`, mode), deps)
			if err != nil {
				t.Fatal(err)
			}
			meta := tools.ToolResultMetadataSnapshot(ctx)
			t.Logf("output=%q metadata=%#v", result.PlainText(), meta)
			if meta["scan_truncated"] != true || !strings.Contains(result.PlainText(), "[WARN]") {
				t.Fatalf("early result cap silently omitted remaining matches: output=%q metadata=%#v", result.PlainText(), meta)
			}
		})
	}
}

func TestIssue428ContentBudgetContextsAndUntruncatedSearch(t *testing.T) {
	for _, tc := range []struct {
		name, text, extra string
		limit             int
		partial           bool
	}{
		{"default", "needle\nneedle\n", "", 1, true},
		{"before", "before\nneedle\nafter\n", `,"before":1`, 1, true},
		{"after", "needle\nafter\ntail\n", `,"after":1`, 2, true},
		{"context", "before\nneedle\nafter\n", `,"context":1`, 2, true},
		{"below-limit", "needle\nother\n", "", 5, false},
		{"no-match", "other\n", "", 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte(tc.text), 0600); err != nil {
				t.Fatal(err)
			}
			s, err := New(root)
			if err != nil {
				t.Fatal(err)
			}
			deps := tools.NewContainer()
			tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
			ctx := tools.WithToolResultMetadata(context.Background())
			result, err := grepTool().Execute(ctx, fmt.Sprintf(`{"pattern":"needle","max_results":%d%s}`, tc.limit, tc.extra), deps)
			if err != nil {
				t.Fatal(err)
			}
			meta := tools.ToolResultMetadataSnapshot(ctx)
			if got := meta["scan_truncated"] == true; got != tc.partial {
				t.Fatalf("metadata=%#v", meta)
			}
			if tc.partial {
				if meta["output_truncated"] != true || meta["output_line_limit"] != tc.limit || !strings.Contains(result.PlainText(), "max_results=") {
					t.Fatalf("missing output budget metadata: %#v %q", meta, result.PlainText())
				}
				if _, ok := meta["scan_cap"]; ok {
					t.Fatalf("output limit confused with file scan limit: %#v", meta)
				}
			} else if strings.Contains(result.PlainText(), "[WARN]") {
				t.Fatalf("unexpected warning: %q", result.PlainText())
			}
		})
	}
}
