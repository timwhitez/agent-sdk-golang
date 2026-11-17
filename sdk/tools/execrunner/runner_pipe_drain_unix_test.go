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

// A foreground shell that exits while a backgrounded child still holds stdout
// is a finished command, not a timeout (issue #248).
func TestRunBackgroundChildHoldingPipeIsNotATimeout(t *testing.T) {
	sink := &recordingStreamSink{}
	start := time.Now()
	res, err := Run(context.Background(), Options{
		Program:                    "sh",
		Args:                       []string{"-c", "sleep 5 & echo started"},
		Timeout:                    3 * time.Second,
		PipeDrainGrace:             100 * time.Millisecond,
		ArtifactOwner:              canonicalRunnerOwner(),
		ArtifactStreamSink:         sink,
		ArtifactResolverCapability: canonicalRunnerCapability(),
	})
	if err != nil || res.TimedOut || res.ExitCode != 0 {
		t.Fatalf("Run() err=%v timed_out=%v exit=%d, want clean exit", err, res.TimedOut, res.ExitCode)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("Run waited %s, want about PipeDrainGrace", elapsed)
	}
	if !res.PipeDrainTimedOut || !strings.Contains(res.Output, "started") || !strings.Contains(res.Output, "background process still holds stdout") {
		t.Fatalf("unexpected result: drain=%v output=%q", res.PipeDrainTimedOut, res.Output)
	}
	if len(res.OutputArtifacts) != 0 || len(sink.manifests) != 0 {
		t.Fatalf("stream still held by a descendant was committed as complete: %#v", res.OutputArtifacts)
	}
}

func TestRunRealTimeoutStillKillsProcessGroupWithPipeDrainGrace(t *testing.T) {
	res, err := Run(context.Background(), Options{
		Program:        "sh",
		Args:           []string{"-c", "sleep 10"},
		Timeout:        200 * time.Millisecond,
		PipeDrainGrace: 100 * time.Millisecond,
	})
	if !res.TimedOut || err == nil || res.PipeDrainTimedOut {
		t.Fatalf("Run() err=%v timed_out=%v drain=%v, want a plain timeout", err, res.TimedOut, res.PipeDrainTimedOut)
	}
}

// A failed command whose descendant still holds the pipes must report the
// drain timeout (exec.Cmd.Wait hides ErrWaitDelay behind ExitError) and must not
// commit the stream as complete.
func TestRunNonZeroExitWithBackgroundChildHoldingPipeIsIncomplete(t *testing.T) {
	sink := &recordingStreamSink{}
	res, err := Run(context.Background(), Options{
		Program:                    "sh",
		Args:                       []string{"-c", "echo a; sleep 5 & exit 3"},
		Timeout:                    3 * time.Second,
		PipeDrainGrace:             100 * time.Millisecond,
		ArtifactOwner:              canonicalRunnerOwner(),
		ArtifactStreamSink:         sink,
		ArtifactResolverCapability: canonicalRunnerCapability(),
	})
	if res.TimedOut || res.ExitCode != 3 || err == nil || !res.PipeDrainTimedOut {
		t.Fatalf("Run() err=%v timed_out=%v exit=%d drain=%v, want exit 3 with drain timeout", err, res.TimedOut, res.ExitCode, res.PipeDrainTimedOut)
	}
	if len(res.OutputArtifacts) != 0 || len(sink.manifests) != 0 {
		t.Fatalf("stream held by a descendant was committed as complete: %#v", res.OutputArtifacts)
	}
}

// With default graces, a timeout whose escaped child holds the pipes must still
// end as KillProcessTimedOut with no committed stream (not at PipeDrainGrace).
func TestRunTimeoutWithEscapedChildHoldingPipeKeepsKillWaitContract(t *testing.T) {
	if _, err := exec.LookPath("setsid"); err != nil {
		t.Skip("setsid is required for escaped-process fixture")
	}
	sink := &recordingStreamSink{}
	res, err := Run(context.Background(), Options{
		Program:                    "sh",
		Args:                       []string{"-c", "printf early; setsid sh -c 'sleep 8; printf late' & sleep 30"},
		Timeout:                    100 * time.Millisecond,
		KillGrace:                  10 * time.Millisecond,
		ArtifactOwner:              canonicalRunnerOwner(),
		ArtifactStreamSink:         sink,
		ArtifactResolverCapability: canonicalRunnerCapability(),
	})
	if !errors.Is(err, ErrProcessKillTimeout) || !res.KillProcessTimedOut || !res.TimedOut {
		t.Fatalf("Run() err=%v kill_timeout=%v timed_out=%v, want ErrProcessKillTimeout", err, res.KillProcessTimedOut, res.TimedOut)
	}
	if len(res.OutputArtifacts) != 0 || len(sink.manifests) != 0 {
		t.Fatalf("open stream was committed as complete: %#v", res.OutputArtifacts)
	}
}
