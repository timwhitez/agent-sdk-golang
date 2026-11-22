package sandbox

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestGrepCountLimitsFileEntriesNotMatchingLines(t *testing.T) {
	root := t.TempDir()
	for name, count := range map[string]int{"a.txt": 51, "b.txt": 2} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(strings.Repeat("needle\n", count)), 0600); err != nil {
			t.Fatal(err)
		}
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	ctx := tools.WithToolResultMetadata(context.Background())
	out, err := grepTool().Execute(ctx, `{"pattern":"needle","output_mode":"count","context":2}`, deps)
	if err != nil || out.PlainText() != "a.txt: 51\nb.txt: 2" || len(tools.ToolResultMetadataSnapshot(ctx)) != 0 {
		t.Fatalf("output=%q err=%v meta=%v", out.PlainText(), err, tools.ToolResultMetadataSnapshot(ctx))
	}
	ctx = tools.WithToolResultMetadata(context.Background())
	out, err = grepTool().Execute(ctx, `{"pattern":"needle","output_mode":"count","max_results":1}`, deps)
	if err != nil || !strings.HasPrefix(out.PlainText(), "a.txt: 51\n") || strings.Contains(out.PlainText(), "b.txt: 2") || tools.ToolResultMetadataSnapshot(ctx)["scan_truncated"] != true {
		t.Fatalf("capped output=%q err=%v meta=%v", out.PlainText(), err, tools.ToolResultMetadataSnapshot(ctx))
	}
}
