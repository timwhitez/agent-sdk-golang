//go:build unix

package execrunner

import (
	"errors"
	"os"
	"os/exec"
	"syscall"
)

func configureProcessGroup(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

func signalProcessGroupTerminate(proc *os.Process) error {
	if proc == nil || proc.Pid <= 0 {
		return nil
	}
	err := syscall.Kill(-proc.Pid, syscall.SIGTERM)
	if err == nil || errors.Is(err, syscall.ESRCH) {
		return nil
	}
	return err
}

func signalProcessGroupKill(proc *os.Process) error {
	if proc == nil || proc.Pid <= 0 {
		return nil
	}
	err := syscall.Kill(-proc.Pid, syscall.SIGKILL)
	if err == nil || errors.Is(err, syscall.ESRCH) {
		return nil
	}
	return err
}

type processGroupController struct {
	process *os.Process
}

func attachProcessGroup(proc *os.Process) (*processGroupController, error) {
	return &processGroupController{process: proc}, nil
}

func (c *processGroupController) terminate() error {
	if c == nil {
		return nil
	}
	return signalProcessGroupTerminate(c.process)
}

func (c *processGroupController) kill() error {
	if c == nil {
		return nil
	}
	return signalProcessGroupKill(c.process)
}

func (c *processGroupController) close() error { return nil }
