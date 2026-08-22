package execrunner

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"runtime"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
)

type recordingStreamSink struct {
	mu        sync.Mutex
	next      int
	objects   map[string][]byte
	manifests map[string]artifact.Manifest
	short     bool
}

func (s *recordingStreamSink) Begin(_ context.Context, request artifact.StreamPutRequest) (artifact.StreamObjectWriter, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.next++
	return &recordingStreamWriter{
		sink:    s,
		request: request,
		ref:     fmt.Sprintf("obj:v1:raw-stream-%03d", s.next),
		short:   s.short,
	}, nil
}

type recordingStreamWriter struct {
	sink    *recordingStreamSink
	request artifact.StreamPutRequest
	ref     string
	content bytes.Buffer
	short   bool
	aborted bool
}

func (w *recordingStreamWriter) Write(p []byte) (int, error) {
	if w.short && len(p) > 0 {
		_, _ = w.content.Write(p[:len(p)-1])
		return len(p) - 1, nil
	}
	return w.content.Write(p)
}

func (w *recordingStreamWriter) Commit(_ context.Context) (artifact.Manifest, error) {
	if w.aborted {
		return artifact.Manifest{}, fmt.Errorf("writer aborted")
	}
	content := append([]byte(nil), w.content.Bytes()...)
	bytesCount := int64(len(content))
	linesCount := int64(0)
	if len(content) > 0 {
		linesCount = int64(bytes.Count(content, []byte("\n")) + 1)
	}
	complete := true
	manifest := artifact.Manifest{
		SchemaVersion: artifact.SchemaVersion,
		ObjectRef:     w.ref,
		ObjectKind:    w.request.ObjectKind,
		Owner:         w.request.Owner,
		Complete:      true,
		Recoverable:   true,
		ObjectMeasurement: artifact.Measurement{
			Bytes:             &bytesCount,
			Lines:             &linesCount,
			SHA256:            artifact.DigestSHA256(content),
			MeasurementSource: "test_stream_sink",
			Complete:          &complete,
		},
		Preview:     artifact.Preview{Kind: artifact.PreviewKindNone},
		Retention:   w.request.Retention,
		ContentType: w.request.ContentType,
		Encoding:    w.request.Encoding,
		Recovery:    w.request.Recovery,
	}
	if err := manifest.Validate(); err != nil {
		return artifact.Manifest{}, err
	}
	w.sink.mu.Lock()
	defer w.sink.mu.Unlock()
	if w.sink.objects == nil {
		w.sink.objects = make(map[string][]byte)
		w.sink.manifests = make(map[string]artifact.Manifest)
	}
	w.sink.objects[w.ref] = content
	w.sink.manifests[w.ref] = manifest.Clone()
	return manifest, nil
}

func (w *recordingStreamWriter) Abort(_ context.Context) error {
	w.aborted = true
	return nil
}

func canonicalRunnerOwner() artifact.Owner {
	return artifact.Owner{
		WorkspaceID: "workspace-execrunner",
		SubjectKind: artifact.SubjectKindRun,
		SubjectID:   "run-execrunner",
		ToolCallID:  "call-shell",
		ToolName:    "shell",
	}
}

func canonicalRunnerCapability() artifact.ResolverCapability {
	return artifact.ResolverCapability{
		Registered: true,
		Recovery: artifact.Recovery{
			Capability:        "goode.artifact.resolve.v1",
			Tool:              "artifact_read",
			AllowedRangeUnits: []artifact.RangeUnit{artifact.RangeUnitBytes},
			Instruction:       "Call artifact_read with object_ref and a byte range.",
		},
	}
}

func TestRunCanonicalStreamsPersistSeparateCompleteStdoutAndStderr(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fixture")
	}

	const stdoutBytes = 150 * 1024
	const stderrText = "stderr-canonical-sentinel"
	sink := &recordingStreamSink{}
	var chunks []OutputChunk
	res, err := Run(context.Background(), Options{
		Program:                    "sh",
		Args:                       []string{"-c", "head -c 153587 /dev/zero | tr '\\000' x; printf 'TAIL_SENTINEL'; printf 'stderr-canonical-sentinel' >&2"},
		MaxOutputBytes:             4_096,
		ArtifactOwner:              canonicalRunnerOwner(),
		ArtifactStreamSink:         sink,
		ArtifactResolverCapability: canonicalRunnerCapability(),
		OnOutputChunk: func(chunk OutputChunk) {
			chunks = append(chunks, chunk)
		},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(res.Output) > 4_096 || !res.OutputTruncated {
		t.Fatalf("combined preview accounting: bytes=%d truncated=%v", len(res.Output), res.OutputTruncated)
	}
	if res.OutputPath != "" {
		t.Fatalf("canonical mode duplicated output into legacy path %q", res.OutputPath)
	}
	if len(res.OutputArtifacts) != 2 {
		t.Fatalf("canonical stream manifests = %d, want 2: %#v", len(res.OutputArtifacts), res.OutputArtifacts)
	}
	byStream := map[string]artifact.Manifest{}
	for _, manifest := range res.OutputArtifacts {
		if err := manifest.Validate(); err != nil {
			t.Fatalf("invalid output manifest: %v", err)
		}
		byStream[manifest.Owner.Stream] = manifest
	}
	stdoutManifest, ok := byStream["stdout"]
	if !ok {
		t.Fatalf("missing stdout manifest: %#v", byStream)
	}
	stderrManifest, ok := byStream["stderr"]
	if !ok {
		t.Fatalf("missing stderr manifest: %#v", byStream)
	}
	stdoutObject := sink.objects[stdoutManifest.ObjectRef]
	stderrObject := sink.objects[stderrManifest.ObjectRef]
	if len(stdoutObject) != stdoutBytes || !bytes.HasSuffix(stdoutObject, []byte("TAIL_SENTINEL")) {
		t.Fatalf("stdout object bytes=%d tail=%q", len(stdoutObject), stdoutObject[maxInt(0, len(stdoutObject)-32):])
	}
	if string(stderrObject) != stderrText {
		t.Fatalf("stderr object = %q", stderrObject)
	}
	if stdoutManifest.ObjectMeasurement.Bytes == nil || *stdoutManifest.ObjectMeasurement.Bytes != int64(len(stdoutObject)) ||
		stdoutManifest.ObjectMeasurement.SHA256 != artifact.DigestSHA256(stdoutObject) {
		t.Fatalf("stdout manifest does not describe stored bytes: %#v", stdoutManifest.ObjectMeasurement)
	}
	if stderrManifest.ObjectMeasurement.Bytes == nil || *stderrManifest.ObjectMeasurement.Bytes != int64(len(stderrObject)) ||
		stderrManifest.ObjectMeasurement.SHA256 != artifact.DigestSHA256(stderrObject) {
		t.Fatalf("stderr manifest does not describe stored bytes: %#v", stderrManifest.ObjectMeasurement)
	}
	if res.OutputBytes != int64(len(stdoutObject)+len(stderrObject)) || res.ArtifactBytes != res.OutputBytes {
		t.Fatalf("combined accounting output=%d artifact=%d objects=%d", res.OutputBytes, res.ArtifactBytes, len(stdoutObject)+len(stderrObject))
	}
	seenStreams := map[string]bool{}
	for _, chunk := range chunks {
		seenStreams[chunk.Stream] = true
	}
	if !seenStreams["stdout"] || !seenStreams["stderr"] {
		t.Fatalf("chunk callbacks lost stream identity: %#v", seenStreams)
	}
}

func TestRunCanonicalStreamShortWriteFailsClosedWithBoundedPreview(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fixture")
	}

	sink := &recordingStreamSink{short: true}
	res, err := Run(context.Background(), Options{
		Program:                    "sh",
		Args:                       []string{"-c", "head -c 20000 /dev/zero | tr '\\000' z"},
		MaxOutputBytes:             1_024,
		ArtifactOwner:              canonicalRunnerOwner(),
		ArtifactStreamSink:         sink,
		ArtifactResolverCapability: canonicalRunnerCapability(),
	})
	if err != nil {
		t.Fatalf("Run should preserve process result after artifact failure: %v", err)
	}
	if len(res.Output) > 1_024 || !res.OutputTruncated {
		t.Fatalf("preview bytes=%d truncated=%v", len(res.Output), res.OutputTruncated)
	}
	if len(res.OutputArtifacts) != 0 {
		t.Fatalf("short write produced complete canonical manifest: %#v", res.OutputArtifacts)
	}
	if res.OutputPath != "" {
		t.Fatalf("canonical failure silently fell back to legacy path: %q", res.OutputPath)
	}
	if !strings.Contains(res.OutputArtifactErr, "stage=artifact_stream_write") ||
		!strings.Contains(res.OutputArtifactErr, "action=") ||
		!strings.Contains(res.OutputArtifactErr, io.ErrShortWrite.Error()) {
		t.Fatalf("short-write diagnostic is not actionable: %q", res.OutputArtifactErr)
	}
	if len(res.OutputArtifactDiagnostics) == 0 || res.OutputArtifactDiagnostics[0].Stage != "artifact_stream_write" {
		t.Fatalf("missing structured stream diagnostic: %#v", res.OutputArtifactDiagnostics)
	}
}

func TestRunCanonicalStreamPreviewIsUTF8SafeForSingleLineCJK(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fixture")
	}

	sink := &recordingStreamSink{}
	res, err := Run(context.Background(), Options{
		Program:                    "sh",
		Args:                       []string{"-c", "i=0; while [ $i -lt 3000 ]; do printf '界🙂'; i=$((i+1)); done"},
		MaxOutputBytes:             1_003,
		ArtifactOwner:              canonicalRunnerOwner(),
		ArtifactStreamSink:         sink,
		ArtifactResolverCapability: canonicalRunnerCapability(),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(res.Output) > 1_003 || !strings.HasPrefix(res.Output, "界🙂") {
		t.Fatalf("unexpected bounded preview: bytes=%d prefix=%q", len(res.Output), res.Output[:minInt(len(res.Output), 16)])
	}
	if !utf8.ValidString(res.Output) {
		t.Fatalf("preview is not valid UTF-8: %x", []byte(res.Output))
	}
	if !strings.Contains(string(sink.objects[res.OutputArtifacts[0].ObjectRef]), "界🙂界🙂") {
		t.Fatal("canonical object lost CJK/emoji stream bytes")
	}
}

func maxInt(left, right int) int {
	if left > right {
		return left
	}
	return right
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}

var _ artifact.StreamSink = (*recordingStreamSink)(nil)
var _ artifact.StreamObjectWriter = (*recordingStreamWriter)(nil)
