package sandbox

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestAuditGlobRejectsMalformedPatterns(t *testing.T) {
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	for _, args := range []string{`{"pattern":"["}`, `{"pattern":"**/[abc"}`} {
		out, err := globTool().Execute(t.Context(), args, deps)
		if err == nil || !strings.Contains(strings.ToLower(err.Error()), "pattern") {
			t.Errorf("malformed glob %s: result=%q err=%v; want explicit pattern error", args, out.PlainText(), err)
		}
	}
}

func TestGlobValidatedPatternsKeepMatchesAndNoMatches(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "nested"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "nested", "a.txt"), []byte("data"), 0o600); err != nil {
		t.Fatal(err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	for _, tc := range []struct {
		pattern string
		want    string
	}{
		{" **/*.txt ", "nested/a.txt"},
		{"**/*.missing", "No files match pattern:"},
	} {
		t.Run(tc.pattern, func(t *testing.T) {
			args, err := json.Marshal(globArgs{Pattern: tc.pattern})
			if err != nil {
				t.Fatal(err)
			}
			out, err := globTool().Execute(t.Context(), string(args), deps)
			if err != nil || !strings.Contains(out.PlainText(), tc.want) {
				t.Fatalf("glob %q: result=%q err=%v; want %q", tc.pattern, out.PlainText(), err, tc.want)
			}
		})
	}
}
