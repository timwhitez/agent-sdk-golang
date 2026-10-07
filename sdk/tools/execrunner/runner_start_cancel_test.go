package execrunner

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"testing"
	"time"
)

type cancelOnDeadlineContext struct {
	context.Context
	cancel context.CancelFunc
}

func (c cancelOnDeadlineContext) Deadline() (time.Time, bool) {
	c.cancel()
	return c.Context.Deadline()
}

func TestRunRejectsCancellationBeforeStart(t *testing.T) {
	for _, stage := range []string{"already_canceled", "expired_deadline", "timeout_context_canceled", "preparation_canceled", "preparation_timeout"} {
		for _, canonical := range []bool{false, true} {
			t.Run(stage+map[bool]string{false: "/legacy", true: "/canonical"}[canonical], func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				wantErr := error(context.Canceled)
				if stage == "already_canceled" {
					cancel()
				}
				if stage == "expired_deadline" {
					var deadlineCancel context.CancelFunc
					ctx, deadlineCancel = context.WithDeadline(ctx, time.Now().Add(-time.Hour))
					defer deadlineCancel()
					wantErr = context.DeadlineExceeded
				}
				starts, preparations, callbacks := 0, 0, 0
				sink := &recordingStreamSink{}
				dir := t.TempDir()
				opts := Options{
					Program: "unused-test-command", ArtifactDir: dir, MaxOutputBytes: 1,
					OnOutputChunk: func(OutputChunk) { callbacks++ },
				}
				if canonical {
					opts.ArtifactOwner = canonicalRunnerOwner()
					opts.ArtifactStreamSink = sink
					opts.ArtifactResolverCapability = canonicalRunnerCapability()
				}
				if stage == "preparation_timeout" {
					opts.Timeout = time.Millisecond
					wantErr = context.DeadlineExceeded
				}
				if stage == "timeout_context_canceled" {
					// WithTimeout consults Deadline after the entry check. Cancel
					// there to exercise the check before output resources are made.
					ctx = cancelOnDeadlineContext{Context: ctx, cancel: cancel}
					opts.Timeout = time.Hour
				}
				res, err := run(ctx, opts, func(runCtx context.Context) {
					preparations++
					switch stage {
					case "preparation_canceled":
						cancel()
					case "preparation_timeout":
						// Wait on the actual timeout, never guess scheduling with sleep.
						<-runCtx.Done()
					}
				}, func(*exec.Cmd) error {
					starts++
					return errors.New("Start must not be called")
				})
				if starts != 0 || !errors.Is(err, wantErr) {
					t.Errorf("Start=%d, error=%v; want Start=0, %v", starts, err, wantErr)
				}
				if stage == "already_canceled" || stage == "expired_deadline" || stage == "timeout_context_canceled" {
					if preparations != 0 {
						t.Errorf("prepared canceled command %d times", preparations)
					}
				} else if preparations != 1 {
					t.Errorf("preparations=%d, want 1", preparations)
				}
				want := Result{ExitCode: -1, TimedOut: errors.Is(wantErr, context.DeadlineExceeded)}
				if !reflect.DeepEqual(res, want) {
					t.Errorf("result=%+v, want %+v (no output or artifacts)", res, want)
				}
				if sink.next != 0 || len(sink.objects) != 0 || len(sink.manifests) != 0 || callbacks != 0 {
					t.Errorf("output side effects: Begin=%d objects=%d Commit=%d callbacks=%d", sink.next, len(sink.objects), len(sink.manifests), callbacks)
				}
				entries, readErr := os.ReadDir(dir)
				if readErr != nil || len(entries) != 0 {
					t.Errorf("legacy artifact directory=%v, error=%v; want empty", entries, readErr)
				}
				if stage == "preparation_timeout" && ctx.Err() != nil {
					t.Errorf("internal timeout canceled parent: %v", ctx.Err())
				}
			})
		}
	}
}

func TestRunStartFailureHasNoOutputSideEffects(t *testing.T) {
	sink := &recordingStreamSink{}
	dir := t.TempDir()
	res, err := Run(context.Background(), Options{
		Program: dir + "/missing-program", ArtifactDir: dir,
		ArtifactOwner: canonicalRunnerOwner(), ArtifactStreamSink: sink,
		ArtifactResolverCapability: canonicalRunnerCapability(),
		OnOutputChunk:              func(OutputChunk) { t.Error("callback on Start failure") },
	})
	if err == nil || !reflect.DeepEqual(res, Result{ExitCode: -1}) || sink.next != 0 {
		t.Fatalf("Run=%+v, %v, Begin=%d; want Start failure with empty output", res, err, sink.next)
	}
}

func TestRunPreservesStartedCommandOutcomes(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fixture")
	}
	for _, tc := range []struct {
		name, command string
		cancel        bool
		exitCode      int
	}{
		{"success", "printf started", false, 0},
		{"nonzero", "printf started; exit 7", false, 7},
		{"canceled", "printf started; exec sleep 30", true, -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			res, err := Run(ctx, Options{
				Program: "sh", Args: []string{"-c", tc.command},
				OnOutputChunk: func(OutputChunk) {
					if tc.cancel {
						cancel() // Output proves Start succeeded before cancellation.
					}
				},
			})
			if res.Output != "started" || res.ExitCode != tc.exitCode || res.TimedOut {
				t.Fatalf("Run=%+v, %v", res, err)
			}
			if tc.cancel {
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("error=%v, want canceled", err)
				}
			} else if tc.exitCode == 0 {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				var exitErr *exec.ExitError
				if !errors.As(err, &exitErr) {
					t.Fatalf("error=%v, want ExitError", err)
				}
			}
		})
	}
}

func TestRunMissingProgramPrecedesCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	res, err := Run(ctx, Options{})
	if err == nil || err.Error() != "missing program" || res.ExitCode != -1 {
		t.Fatalf("Run=%+v, %v; want missing program", res, err)
	}
}
