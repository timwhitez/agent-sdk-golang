package execrunner_test

import (
	"context"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/execrunner"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestOutputTruncatedAfterUTF8PreviewExpansion(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX cat fixture")
	}

	for _, tc := range []struct {
		name      string
		raw       []byte
		limit     int
		truncated bool
		suffix    string
	}{
		{"replacement-expands-past-budget", []byte{0xff, 'a', 'b', 'c', 'd', 'e', 'f', 'Z'}, 8, true, ""},
		{"replacement-fits", []byte{0xff, 'f', 'Z'}, 8, false, "fZ"},
		{"valid-exact-budget", []byte("abcdefgh"), 8, false, "gh"},
		{"split-valid-rune", []byte("界界"), 5, true, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			p := filepath.Join(dir, "short.dat")
			if err := os.WriteFile(p, tc.raw, 0600); err != nil {
				t.Fatal(err)
			}
			res, err := execrunner.Run(context.Background(), execrunner.Options{Program: "cat", Args: []string{p}, MaxOutputBytes: tc.limit, ArtifactDir: dir})
			if err != nil {
				t.Fatal(err)
			}
			if res.OutputTruncated != tc.truncated || !strings.HasSuffix(res.Output, tc.suffix) || len(res.Output) > tc.limit || !utf8.ValidString(res.Output) || res.OutputBytes != int64(len(tc.raw)) {
				t.Fatalf("output=%q total=%d captured=%d truncated=%t; want truncated=%t suffix=%q", res.Output, res.OutputBytes, res.CapturedBytes, res.OutputTruncated, tc.truncated, tc.suffix)
			}
		})
	}
}
