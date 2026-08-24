//go:build unix

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
		Program:                    "sh",
		Args:                       []string{"-c", "printf early; setsid sh -c 'sleep 1; printf late' & sleep 30"},
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
