package execrunner

import (
	"fmt"
	"sync"
)

// jobOps is the thin syscall surface of a Windows Job Object. The ordering and
// cleanup decisions live in attachJob/jobController so they run (against fakes)
// on every platform; only the Windows file talks to the OS.
type jobOps interface {
	createJob() (uintptr, error)
	// setKillOnClose sets or clears JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
	setKillOnClose(job uintptr, on bool) error
	openProcess(pid int) (uintptr, error)
	assign(job, proc uintptr) error
	// resume releases a process created with CREATE_SUSPENDED.
	resume(proc uintptr) error
	terminate(job uintptr) error
	closeHandle(h uintptr) error
}

// attachJob places the suspended process pid into a new kill-on-close job, then
// resumes it, so it cannot run (or spawn a descendant) outside the job. Any
// failure closes every handle it opened; closing a job that holds the process
// kills it, so the caller never inherits a half-supervised process.
func attachJob(ops jobOps, pid int) (*jobController, error) {
	job, err := ops.createJob()
	if err != nil {
		return nil, err
	}
	attached := false
	defer func() {
		if !attached {
			_ = ops.closeHandle(job)
		}
	}()
	// The limit is set before assignment so there is no moment at which the
	// process is in the job without kill-on-close.
	if err := ops.setKillOnClose(job, true); err != nil {
		return nil, err
	}
	proc, err := ops.openProcess(pid)
	if err != nil {
		return nil, fmt.Errorf("OpenProcess(%d): %w", pid, err)
	}
	defer ops.closeHandle(proc)
	if err := ops.assign(job, proc); err != nil {
		return nil, err
	}
	if err := ops.resume(proc); err != nil {
		return nil, err
	}
	attached = true
	return &jobController{ops: ops, job: job}, nil
}

type jobController struct {
	ops    jobOps
	mu     sync.Mutex
	job    uintptr
	closed bool
}

func (c *jobController) terminate() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	return c.ops.terminate(c.job)
}

func (c *jobController) kill() error { return c.terminate() }

// close releases the job handle WITHOUT killing what is still in it. Closing the
// last handle of a job with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE terminates every
// process in it (Win32 JOBOBJECT_BASIC_LIMIT_INFORMATION), which would kill a
// background child that outlived the foreground command, unlike unix (#248).
// Kill-on-close stays set until here so a crashed host still cleans up; timeouts
// and cancellation have already called kill() explicitly.
func (c *jobController) close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	releaseErr := c.ops.setKillOnClose(c.job, false)
	closeErr := c.ops.closeHandle(c.job)
	if releaseErr != nil {
		return releaseErr
	}
	return closeErr
}
