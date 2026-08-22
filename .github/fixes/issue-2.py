from pathlib import Path

runner = Path("sdk/tools/execrunner/runner.go")
text = runner.read_text()
old = '''\tcanonical.finish(context.WithoutCancel(ctx))
\tsnap := collector.snapshot()
'''
new = '''\tif killWaitTimedOut {
\t\tcanonical.abortIncomplete(context.WithoutCancel(ctx), fmt.Errorf("process wait exceeded %s after tree kill; output streams have not reached EOF", opts.KillWaitGrace))
\t} else {
\t\tcanonical.finish(context.WithoutCancel(ctx))
\t}
\tsnap := collector.snapshot()
'''
if text.count(old) != 1:
    raise SystemExit(f"runner canonical finish anchor count={text.count(old)}")
runner.write_text(text.replace(old, new))

canonical = Path("sdk/tools/execrunner/canonical_stream.go")
text = canonical.read_text()
anchor = '''func (s *canonicalProcessStreams) manifests() []artifact.Manifest {
'''
insert = '''func (s *canonicalProcessStreams) abortIncomplete(ctx context.Context, cause error) {
\tif s == nil {
\t\treturn
\t}
\tif s.stdout != nil {
\t\ts.stdout.abortIncomplete(ctx, cause)
\t}
\tif s.stderr != nil {
\t\ts.stderr.abortIncomplete(ctx, cause)
\t}
}

func (s *canonicalProcessStreams) manifests() []artifact.Manifest {
'''
if text.count(anchor) != 1:
    raise SystemExit(f"stream collection anchor count={text.count(anchor)}")
text = text.replace(anchor, insert)
anchor = '''func (c *canonicalStreamCapture) failLocked(stage, action string, err error) {
'''
insert = '''func (c *canonicalStreamCapture) abortIncomplete(ctx context.Context, cause error) {
\tc.mu.Lock()
\tdefer c.mu.Unlock()
\tif c.finished {
\t\treturn
\t}
\tc.finished = true
\tc.failed = true
\tc.diagnostics = append(c.diagnostics, streamDiagnostic(
\t\t"artifact_stream_incomplete",
\t\t"wait_for_stream_eof_then_retry",
\t\tcause,
\t))
\tif c.writer != nil {
\t\tif ctx == nil {
\t\t\tctx = c.ctx
\t\t}
\t\tif abortErr := c.writer.Abort(ctx); abortErr != nil {
\t\t\tc.diagnostics = append(c.diagnostics, streamDiagnostic(
\t\t\t\t"artifact_stream_abort",
\t\t\t\t"inspect_stream_store_cleanup",
\t\t\t\tabortErr,
\t\t\t))
\t\t}
\t\tc.writer = nil
\t}
}

func (c *canonicalStreamCapture) failLocked(stage, action string, err error) {
'''
if text.count(anchor) != 1:
    raise SystemExit(f"capture fail anchor count={text.count(anchor)}")
canonical.write_text(text.replace(anchor, insert))

Path("sdk/tools/execrunner/runner_incomplete_stream_unix_test.go").write_text(r'''//go:build unix

package execrunner

import (
	"context"
	"errors"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestRunDoesNotCommitCanonicalStreamsBeforeEOF(t *testing.T) {
	if _, err := exec.LookPath("setsid"); err != nil {
		t.Skip("setsid is required for escaped-process fixture")
	}
	sink := &recordingStreamSink{}
	res, err := Run(context.Background(), Options{
		Program: "sh",
		Args: []string{"-c", "printf early; setsid sh -c 'sleep 1; printf late' & sleep 30"},
		Timeout:                    100 * time.Millisecond,
		KillGrace:                  10 * time.Millisecond,
		KillWaitGrace:              100 * time.Millisecond,
		ArtifactOwner:              canonicalRunnerOwner(),
		ArtifactStreamSink:         sink,
		ArtifactResolverCapability: canonicalRunnerCapability(),
	})
	if !errors.Is(err, ErrProcessKillTimeout) {
		t.Fatalf("Run() error = %v, want ErrProcessKillTimeout", err)
	}
	if !res.KillProcessTimedOut {
		t.Fatalf("KillProcessTimedOut = false")
	}
	if len(res.OutputArtifacts) != 0 {
		t.Fatalf("open stream was committed as complete: %#v", res.OutputArtifacts)
	}
	foundIncomplete := false
	for _, diagnostic := range res.OutputArtifactDiagnostics {
		if diagnostic.Stage == "artifact_stream_incomplete" {
			foundIncomplete = true
			if diagnostic.Action != "wait_for_stream_eof_then_retry" {
				t.Fatalf("incomplete action = %q", diagnostic.Action)
			}
		}
	}
	if !foundIncomplete || !strings.Contains(res.OutputArtifactErr, "stage=artifact_stream_incomplete") {
		t.Fatalf("missing actionable incomplete-stream diagnostic: %#v / %q", res.OutputArtifactDiagnostics, res.OutputArtifactErr)
	}
	// Let the escaped writer deliver its suffix. It must not create a manifest
	// after Run has returned and reported the stream incomplete.
	time.Sleep(1200 * time.Millisecond)
	if len(sink.manifests) != 0 {
		t.Fatalf("aborted stream was committed later: %#v", sink.manifests)
	}
}
''')
