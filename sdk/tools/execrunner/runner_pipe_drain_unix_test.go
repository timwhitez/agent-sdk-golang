//go:build unix

package execrunner

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// A foreground shell that exits while a backgrounded child still holds stdout
// is a finished command, not a timeout (issue #248).
func TestRunBackgroundChildHoldingPipeIsNotATimeout(t *testing.T) {
	sink := &recordingStreamSink{}
	res, err := Run(context.Background(), Options{
		Program:                    "sh",
		Args:                       []string{"-c", "sleep 5 & echo started"},
		Timeout:                    30 * time.Second, // only a safety net on loaded machines
		PipeDrainGrace:             100 * time.Millisecond,
		ArtifactOwner:              canonicalRunnerOwner(),
		ArtifactStreamSink:         sink,
		ArtifactResolverCapability: canonicalRunnerCapability(),
	})
	if err != nil || res.TimedOut || res.ExitCode != 0 {
		t.Fatalf("Run() err=%v timed_out=%v exit=%d, want clean exit", err, res.TimedOut, res.ExitCode)
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
		Timeout:                    30 * time.Second, // only a safety net on loaded machines
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

// With the drain grace shorter than the kill wait grace (the defaults), a timeout whose escaped child holds the pipes must still
// end as KillProcessTimedOut with no committed stream (not at PipeDrainGrace).
func TestRunTimeoutWithEscapedChildHoldingPipeKeepsKillWaitContract(t *testing.T) {
	if _, err := exec.LookPath("setsid"); err != nil {
		t.Skip("setsid is required for escaped-process fixture")
	}
	sink := &recordingStreamSink{}
	// The escaped child signals readiness before the command waits, so the
	// timeout cannot fire before setsid has run, even on a loaded machine.
	ready := filepath.Join(t.TempDir(), "ready")
	res, err := Run(context.Background(), Options{
		Program:                    "sh",
		Args:                       []string{"-c", `printf early; setsid sh -c 'touch "$0"; sleep 8; printf late' "$0" & while [ ! -e "$0" ]; do sleep 0.01; done; sleep 30`, ready},
		Timeout:                    3 * time.Second,
		KillGrace:                  10 * time.Millisecond,
		KillWaitGrace:              time.Second,
		PipeDrainGrace:             200 * time.Millisecond, // shorter than KillWaitGrace, as with default options
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

// Output the process already wrote must not be dropped because the consumer is
// slower than PipeDrainGrace.
func TestRunSlowConsumerDoesNotLoseOutputToPipeDrain(t *testing.T) {
	res, err := Run(context.Background(), Options{
		Program:        "sh",
		Args:           []string{"-c", "echo a; sleep 0.2; echo b"},
		Timeout:        10 * time.Second,
		PipeDrainGrace: 300 * time.Millisecond,
		OnOutputChunk:  func(OutputChunk) { time.Sleep(time.Second) },
	})
	if err != nil || res.PipeDrainTimedOut || !strings.Contains(res.Output, "a") || !strings.Contains(res.Output, "b") || strings.Contains(res.Output, "background process") {
		t.Fatalf("Run() err=%v drain=%v output=%q, want both lines and no drain timeout", err, res.PipeDrainTimedOut, res.Output)
	}
}

func TestRunClosesPipesWhenContextEndsBeforeStart(t *testing.T) {
	count := func() int {
		entries, err := os.ReadDir("/proc/self/fd")
		if err != nil {
			t.Skip("/proc/self/fd is unavailable")
		}
		return len(entries)
	}
	before := count()
	for i := 0; i < 5; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		_, err := run(ctx, Options{Program: "true"}, context.WithTimeout, func(context.Context) { cancel() }, (*exec.Cmd).Start)
		if err == nil {
			t.Fatal("expected cancellation error")
		}
	}
	if after := count(); after > before {
		t.Fatalf("leaked %d fds across cancelled starts", after-before)
	}
}

// startYesScript backgrounds a continuous writer and exits at once. The
// background PID is recorded so the test can kill the stray process.
func startYesScript(t *testing.T) (args []string, stampFile string) {
	t.Helper()
	pidFile := filepath.Join(t.TempDir(), "pid")
	t.Cleanup(func() {
		if b, err := os.ReadFile(pidFile); err == nil {
			if pid, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil {
				_ = syscall.Kill(pid, syscall.SIGKILL)
			}
		}
	})
	// The stamp is taken just before the shell exits, so tests can measure the
	// post-exit wait without fork/exec latency on a loaded machine.
	stampFile = filepath.Join(filepath.Dir(pidFile), "stamp")
	return []string{"-c", `yes & echo $! > "$0"; date +%s%N > "$1"; exit 0`, pidFile, stampFile}, stampFile
}

// A background writer must not extend the post-exit drain beyond the hard
// ceiling PipeDrainGrace+KillWaitGrace, even though time inside a consumer Write
// is excluded from the PipeDrainGrace budget.
func TestRunContinuousBackgroundWriterIsBoundedByPipeDrainGrace(t *testing.T) {
	args, stampFile := startYesScript(t)
	res, err := Run(context.Background(), Options{Program: "sh", Args: args, PipeDrainGrace: 300 * time.Millisecond, KillWaitGrace: 300 * time.Millisecond})
	end := time.Now()
	if err != nil || res.ExitCode != 0 || res.TimedOut || !res.PipeDrainTimedOut {
		t.Fatalf("Run() err=%v exit=%d timed_out=%v drain=%v", err, res.ExitCode, res.TimedOut, res.PipeDrainTimedOut)
	}
	b, rerr := os.ReadFile(stampFile)
	exitNanos, perr := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	if rerr != nil || perr != nil {
		t.Fatalf("exit stamp unreadable: %v %v", rerr, perr)
	}
	if after := end.Sub(time.Unix(0, exitNanos)); after > 3*time.Second {
		t.Fatalf("Run returned %s after the shell exited (ceiling is 600ms) with a continuous background writer", after)
	}
}

// Timeout and caller cancellation still end a drain wait with the usual
// timeout/cancel semantics.
func TestRunTimeoutAndCancelEndDrainWithChattyBackgroundWriter(t *testing.T) {
	start := time.Now()
	res, err := Run(context.Background(), Options{Program: "sh", Args: yesArgs(t), Timeout: time.Second, PipeDrainGrace: time.Minute})
	if err == nil || !res.TimedOut || time.Since(start) > 15*time.Second {
		t.Fatalf("timeout: err=%v timed_out=%v after %s", err, res.TimedOut, time.Since(start))
	}
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(200*time.Millisecond, cancel)
	start = time.Now()
	res, err = Run(ctx, Options{Program: "sh", Args: yesArgs(t), PipeDrainGrace: time.Minute})
	if err == nil || res.TimedOut || time.Since(start) > 15*time.Second {
		t.Fatalf("cancel: err=%v timed_out=%v after %s", err, res.TimedOut, time.Since(start))
	}
}

// A forced drain cut-off that races with the timeout must never be committed as a
// complete stream: sweep the timeout across the drain deadline.
func TestRunDrainCutoffRacingTimeoutNeverCommitsIncompleteStream(t *testing.T) {
	if _, err := exec.LookPath("setsid"); err != nil {
		t.Skip("setsid is required for escaped-process fixture")
	}
	for ms := 190; ms <= 214; ms += 2 {
		sink := &recordingStreamSink{}
		pidFile := filepath.Join(t.TempDir(), "pid")
		res, _ := Run(context.Background(), Options{
			Program:                    "sh",
			Args:                       []string{"-c", `echo x; setsid sleep 2 & echo $! > "$0"; exit 0`, pidFile},
			Timeout:                    time.Duration(ms) * time.Millisecond,
			PipeDrainGrace:             200 * time.Millisecond,
			KillGrace:                  50 * time.Millisecond,
			KillWaitGrace:              100 * time.Millisecond,
			ArtifactOwner:              canonicalRunnerOwner(),
			ArtifactStreamSink:         sink,
			ArtifactResolverCapability: canonicalRunnerCapability(),
		})
		// If the group kill reached the child before setsid ran, the stream really
		// ended; only a still-live holder proves the pipe was force-closed.
		if holderAlive(pidFile) && (len(res.OutputArtifacts) != 0 || len(sink.manifests) != 0) {
			t.Fatalf("timeout=%dms: stream still held by an escaped child was committed (drain=%v killwait=%v)", ms, res.PipeDrainTimedOut, res.KillProcessTimedOut)
		}
	}
}

func holderAlive(pidFile string) bool {
	b, err := os.ReadFile(pidFile)
	if err != nil {
		return false
	}
	stat, err := os.ReadFile("/proc/" + strings.TrimSpace(string(b)) + "/stat")
	if err != nil {
		return false
	}
	_, rest, _ := strings.Cut(string(stat), ") ")
	return !strings.HasPrefix(rest, "Z")
}

func TestRunNormalPathReleasesPipeDescriptors(t *testing.T) {
	entries := func() int {
		e, err := os.ReadDir("/proc/self/fd")
		if err != nil {
			t.Skip("/proc/self/fd is unavailable")
		}
		return len(e)
	}
	defer debug.SetGCPercent(debug.SetGCPercent(-1)) // descriptors must not rely on finalizers
	before := entries()
	for i := 0; i < 20; i++ {
		if res, err := Run(context.Background(), Options{Program: "sh", Args: []string{"-c", "echo hi"}}); err != nil || !strings.Contains(res.Output, "hi") {
			t.Fatalf("Run() err=%v output=%q", err, res.Output)
		}
	}
	if after := entries(); after > before {
		t.Fatalf("leaked %d fds over 20 normal runs", after-before)
	}
}

func yesArgs(t *testing.T) []string {
	args, _ := startYesScript(t)
	return args
}

// Issue #248/#266: a finished foreground command must not kill a background
// child that still holds stdout; the child outlives Run.
func TestRunDoesNotKillBackgroundChildHoldingPipe(t *testing.T) {
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "pid")
	t.Cleanup(func() {
		if b, err := os.ReadFile(pidFile); err == nil {
			if pid, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil && pid > 1 {
				_ = syscall.Kill(pid, syscall.SIGKILL) // exactly the PID this test started
			}
		}
	})
	res, err := Run(context.Background(), Options{
		Program:        "sh",
		Args:           []string{"-c", `sleep 30 & echo $! > "$0"; exit 0`, pidFile},
		Timeout:        30 * time.Second, // safety net only
		PipeDrainGrace: 100 * time.Millisecond,
	})
	if err != nil || res.TimedOut || res.ExitCode != 0 || !res.PipeDrainTimedOut {
		t.Fatalf("Run() err=%v timed_out=%v exit=%d drain=%v", err, res.TimedOut, res.ExitCode, res.PipeDrainTimedOut)
	}
	b, err := os.ReadFile(pidFile) // written before the shell exited, so present once Run returns
	if err != nil {
		t.Fatalf("pid file: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil {
		t.Fatalf("pid %q: %v", b, err)
	}
	if err := syscall.Kill(pid, 0); err != nil {
		t.Fatalf("background child %d was killed or exited after Run returned: %v", pid, err)
	}
}

// Issue #266: a continuous writer behind a slow consumer must still be cut off
// well before the hard ceiling once more than a pipe capacity is read after the
// exit (Write time is excluded only for output that predates it).
func TestRunContinuousWriterWithSlowConsumerIsCutOffBeforeCeiling(t *testing.T) {
	args, stampFile := startYesScript(t)
	res, err := Run(context.Background(), Options{
		Program:        "sh",
		Args:           args,
		PipeDrainGrace: 300 * time.Millisecond,
		KillWaitGrace:  30 * time.Second, // ceiling = 30.3s
		OnOutputChunk:  func(OutputChunk) { time.Sleep(100 * time.Millisecond) },
	})
	end := time.Now()
	if err != nil || res.TimedOut || !res.PipeDrainTimedOut {
		t.Fatalf("Run() err=%v timed_out=%v drain=%v", err, res.TimedOut, res.PipeDrainTimedOut)
	}
	b, _ := os.ReadFile(stampFile)
	exitNanos, perr := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	if perr != nil {
		t.Fatalf("exit stamp unreadable: %v", perr)
	}
	if after := end.Sub(time.Unix(0, exitNanos)); after > 10*time.Second {
		t.Fatalf("Run returned %s after exit; the slow consumer kept extending the drain", after)
	}
}
