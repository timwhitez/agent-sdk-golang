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
