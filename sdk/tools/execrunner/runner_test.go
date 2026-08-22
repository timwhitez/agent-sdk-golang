package execrunner

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"io"
	"os"
	"runtime"
	"strings"
	"testing"
	"unicode/utf8"
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

func TestOutputCollectorKeepsBoundedUTF8PreviewAndCompleteArtifact(t *testing.T) {
	payload := []byte(strings.Repeat("界🙂", 16) + "TAIL_SENTINEL")
	collector := newOutputCollector(17, t.TempDir(), "collector-*.log", nil)
	if n, err := collector.Write(payload); err != nil || n != len(payload) {
		t.Fatalf("write = %d, %v", n, err)
	}
	collector.Close()
	snapshot := collector.snapshot()

	if len(snapshot.preview) > 17 {
		t.Fatalf("preview bytes = %d, want <= 17", len(snapshot.preview))
	}
	if !utf8.ValidString(snapshot.preview) {
		t.Fatalf("preview is not valid UTF-8: %x", []byte(snapshot.preview))
	}
	artifact, err := os.ReadFile(snapshot.outputPath)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	if !bytes.Equal(artifact, payload) {
		t.Fatalf("artifact differs: got=%d bytes want=%d", len(artifact), len(payload))
	}
	if !bytes.Contains(artifact, []byte("TAIL_SENTINEL")) {
		t.Fatal("artifact lost tail sentinel")
	}
	gotHash := sha256.Sum256(artifact)
	wantHash := sha256.Sum256(payload)
	if gotHash != wantHash {
		t.Fatalf("artifact sha256 = %x, want %x", gotHash, wantHash)
	}
	if snapshot.totalBytes != int64(len(payload)) || snapshot.artifactBytes != int64(len(payload)) {
		t.Fatalf("accounting total=%d artifact=%d want=%d", snapshot.totalBytes, snapshot.artifactBytes, len(payload))
	}
}

func TestOutputCollectorReportsArtifactCreateFailure(t *testing.T) {
	badDir := t.TempDir() + "/not-a-directory"
	if err := os.WriteFile(badDir, []byte("fixture"), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	collector := newOutputCollector(4, badDir, "collector-*.log", nil)
	if n, err := collector.Write([]byte("preview-and-overflow")); err != nil || n != len("preview-and-overflow") {
		t.Fatalf("write = %d, %v", n, err)
	}
	snapshot := collector.snapshot()
	if strings.TrimSpace(snapshot.artifactErr) == "" {
		t.Fatal("expected visible artifact creation failure")
	}
	if snapshot.outputPath != "" || snapshot.artifactBytes != 0 {
		t.Fatalf("failed artifact claimed path/bytes: %q %d", snapshot.outputPath, snapshot.artifactBytes)
	}
}

type shortArtifactWriter struct{}

func (shortArtifactWriter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	return len(p) - 1, nil
}

func TestWriteArtifactChunkTurnsShortWriteIntoError(t *testing.T) {
	written, err := writeArtifactChunk(shortArtifactWriter{}, []byte("complete-me"))
	if written != len("complete-me")-1 || !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("writeArtifactChunk = %d, %v", written, err)
	}
}

func TestRunAccountsCombinedStdoutAndStderr(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fixture")
	}
	const stdout = "stdout-fixture"
	const stderr = "stderr-fixture"
	res, err := Run(context.Background(), Options{
		Program:        "sh",
		Args:           []string{"-c", "printf 'stdout-fixture'; printf 'stderr-fixture' >&2"},
		MaxOutputBytes: 4,
		ArtifactDir:    t.TempDir(),
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	wantBytes := int64(len(stdout) + len(stderr))
	if res.OutputBytes != wantBytes || res.ArtifactBytes != wantBytes {
		t.Fatalf("accounting output=%d artifact=%d want=%d", res.OutputBytes, res.ArtifactBytes, wantBytes)
	}
	artifact, err := os.ReadFile(res.OutputPath)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	if len(artifact) != int(wantBytes) || !bytes.Contains(artifact, []byte(stdout)) || !bytes.Contains(artifact, []byte(stderr)) {
		t.Fatalf("combined artifact = %q", artifact)
	}
}
