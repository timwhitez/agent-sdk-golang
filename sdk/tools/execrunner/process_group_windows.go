//go:build windows

package execrunner

import (
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
	cmd.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP
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
