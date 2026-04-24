package execrunner

import (
	"context"
	"runtime"
	"testing"
)

func TestRun_OnOutputChunkReportsStreamProgress(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fixture")
	}

	var chunks []OutputChunk
	res, err := Run(context.Background(), Options{
		Program: "sh",
		Args:    []string{"-c", "printf 'hello-progress'"},
		OnOutputChunk: func(chunk OutputChunk) {
			chunks = append(chunks, chunk)
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Output != "hello-progress" {
		t.Fatalf("output = %q, want %q", res.Output, "hello-progress")
	}
	if len(chunks) == 0 {
		t.Fatal("expected output chunk callbacks")
	}
	last := chunks[len(chunks)-1]
	if last.TotalBytes < int64(len("hello-progress")) {
		t.Fatalf("total bytes = %d, want >= %d", last.TotalBytes, len("hello-progress"))
	}
}
