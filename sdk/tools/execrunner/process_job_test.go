package execrunner

import (
	"errors"
	"reflect"
	"testing"
)

// fakeJobOps records calls; failAt makes the named call fail.
type fakeJobOps struct {
	calls  []string
	failAt string
}

func (f *fakeJobOps) rec(name string) error {
	f.calls = append(f.calls, name)
	if f.failAt == name {
		return errors.New(name + " failed")
	}
	return nil
}
func (f *fakeJobOps) createJob() (uintptr, error) { return 1, f.rec("create") }
func (f *fakeJobOps) setKillOnClose(_ uintptr, on bool) error {
	if on {
		return f.rec("killOnClose=true")
	}
	return f.rec("killOnClose=false")
}
func (f *fakeJobOps) openProcess(int) (uintptr, error) { return 2, f.rec("open") }
func (f *fakeJobOps) assign(_, _ uintptr) error        { return f.rec("assign") }
func (f *fakeJobOps) resume(uintptr) error             { return f.rec("resume") }
func (f *fakeJobOps) terminate(uintptr) error          { return f.rec("terminate") }
func (f *fakeJobOps) closeHandle(h uintptr) error {
	if h == 1 {
		return f.rec("closeJob")
	}
	return f.rec("closeProc")
}

func TestAttachJobOrderAndCleanup(t *testing.T) {
	// The process must be inside a kill-on-close job before it first runs, and
	// every handle opened before a failure must be closed.
	cases := []struct {
		failAt string
		want   []string
	}{
		{"", []string{"create", "killOnClose=true", "open", "assign", "resume", "closeProc"}},
		{"create", []string{"create"}},
		{"killOnClose=true", []string{"create", "killOnClose=true", "closeJob"}},
		{"open", []string{"create", "killOnClose=true", "open", "closeJob"}},
		{"assign", []string{"create", "killOnClose=true", "open", "assign", "closeProc", "closeJob"}},
		{"resume", []string{"create", "killOnClose=true", "open", "assign", "resume", "closeProc", "closeJob"}},
	}
	for _, tc := range cases {
		t.Run("fail_"+tc.failAt, func(t *testing.T) {
			f := &fakeJobOps{failAt: tc.failAt}
			c, err := attachJob(f, 42)
			if (err == nil) != (tc.failAt == "") || (c == nil) != (tc.failAt != "") {
				t.Fatalf("attachJob() = %v, %v", c, err)
			}
			if !reflect.DeepEqual(f.calls, tc.want) {
				t.Fatalf("calls = %v, want %v", f.calls, tc.want)
			}
		})
	}
}

// close must clear kill-on-close before dropping the last job handle, otherwise
// a background child that outlived the foreground command is killed at Run
// return (unix leaves it alive, issue #248).
func TestJobCloseReleasesWithoutKilling(t *testing.T) {
	f := &fakeJobOps{}
	c, err := attachJob(f, 42)
	if err != nil {
		t.Fatal(err)
	}
	f.calls = nil
	if err := c.close(); err != nil {
		t.Fatal(err)
	}
	if want := []string{"killOnClose=false", "closeJob"}; !reflect.DeepEqual(f.calls, want) {
		t.Fatalf("close calls = %v, want %v", f.calls, want)
	}
	f.calls = nil
	if err := c.close(); err != nil || c.terminate() != nil || len(f.calls) != 0 {
		t.Fatalf("second close / terminate after close must be no-ops, calls=%v", f.calls)
	}
}

func TestJobCloseStillClosesHandleWhenReleaseFails(t *testing.T) {
	f := &fakeJobOps{}
	c, _ := attachJob(f, 42)
	f.calls, f.failAt = nil, "killOnClose=false"
	if err := c.close(); err == nil {
		t.Fatal("close() hid the release failure")
	}
	if want := []string{"killOnClose=false", "closeJob"}; !reflect.DeepEqual(f.calls, want) {
		t.Fatalf("close calls = %v, want %v (handle must not leak)", f.calls, want)
	}
}

func TestJobTerminateAndKillReachTheJob(t *testing.T) {
	f := &fakeJobOps{}
	c, _ := attachJob(f, 42)
	f.calls, f.failAt = nil, "terminate"
	if c.terminate() == nil || c.kill() == nil {
		t.Fatal("terminate error was dropped")
	}
	if want := []string{"terminate", "terminate"}; !reflect.DeepEqual(f.calls, want) {
		t.Fatalf("calls = %v", f.calls)
	}
	var nilCtl *jobController
	if nilCtl.terminate() != nil || nilCtl.kill() != nil || nilCtl.close() != nil {
		t.Fatal("nil controller must be a no-op")
	}
}

// After terminate/kill (timeout, cancel) the runner discards their errors, so
// close must leave kill-on-close set as the backstop.
func TestJobCloseAfterTerminateOrKillKeepsKillOnClose(t *testing.T) {
	for _, name := range []string{"terminate", "kill"} {
		t.Run(name, func(t *testing.T) {
			f := &fakeJobOps{}
			c, _ := attachJob(f, 42)
			f.calls, f.failAt = nil, "terminate" // the kill itself fails
			if name == "terminate" {
				_ = c.terminate()
			} else {
				_ = c.kill()
			}
			f.calls, f.failAt = nil, ""
			if err := c.close(); err != nil {
				t.Fatal(err)
			}
			if want := []string{"closeJob"}; !reflect.DeepEqual(f.calls, want) {
				t.Fatalf("close calls = %v, want %v (no release after a kill request)", f.calls, want)
			}
		})
	}
}
