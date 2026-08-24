//go:build !unix && !windows

package execrunner

import (
	"os"
	"os/exec"
)

func configureProcessGroup(cmd *exec.Cmd) {
	_ = cmd
}

func signalProcessGroupTerminate(proc *os.Process) error {
	if proc == nil {
		return nil
	}
	return proc.Kill()
}

func signalProcessGroupKill(proc *os.Process) error {
	if proc == nil {
		return nil
	}
	return proc.Kill()
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
