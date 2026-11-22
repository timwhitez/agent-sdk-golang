package sandbox

import (
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestGrepDecodesUnicodeBOMBeforeMatching(t *testing.T) {
	text := "needle\n世界😀\n"
	fixtures := map[string][]byte{
		"utf8":    append([]byte{0xef, 0xbb, 0xbf}, []byte(text)...),
		"utf16le": encodeUTF16Test(binary.LittleEndian, text),
		"utf16be": encodeUTF16Test(binary.BigEndian, text),
		"utf32le": encodeUTF32Test(binary.LittleEndian, text),
		"utf32be": encodeUTF32Test(binary.BigEndian, text),
	}
	for name, raw := range fixtures {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			if err := os.WriteFile(filepath.Join(root, "text.txt"), raw, 0600); err != nil {
				t.Fatal(err)
			}
			s, err := New(root)
			if err != nil {
				t.Fatal(err)
			}
			deps := tools.NewContainer()
			tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
			out, err := grepTool().Execute(context.Background(), `{"path":"text.txt","pattern":"^needle$"}`, deps)
			if err != nil || out.PlainText() != "text.txt:1: needle" {
				t.Fatalf("output=%q err=%v", out.PlainText(), err)
			}
			out, err = grepTool().Execute(context.Background(), `{"path":"text.txt","pattern":"世界😀"}`, deps)
			if err != nil || out.PlainText() != "text.txt:2: 世界😀" {
				t.Fatalf("unicode output=%q err=%v", out.PlainText(), err)
			}
		})
	}
}

func TestGrepMalformedBOMTextReportsIncompleteScan(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "broken.txt"), []byte{0xff, 0xfe, 'n'}, 0600); err != nil {
		t.Fatal(err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
	ctx := tools.WithToolResultMetadata(context.Background())
	out, err := grepTool().Execute(ctx, `{"pattern":"needle"}`, deps)
	if err != nil || !strings.Contains(out.PlainText(), "malformed UTF-16LE") || tools.ToolResultMetadataSnapshot(ctx)["warning_kind"] != partialScanWarningKind {
		t.Fatalf("output=%q err=%v meta=%v", out.PlainText(), err, tools.ToolResultMetadataSnapshot(ctx))
	}
}
