//go:build windows

package execrunner

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

const windowsHelperModeEnv = "GOODE_EXECRUNNER_WINDOWS_HELPER"
const windowsHelperPIDFileEnv = "GOODE_EXECRUNNER_WINDOWS_PID_FILE"

func TestWindowsProcessTreeHelper(t *testing.T) {
	mode := os.Getenv(windowsHelperModeEnv)
	if mode == "" {
		return
	}
	if mode == "child" {
		time.Sleep(10 * time.Minute)
		return
	}
	if mode != "parent" {
		os.Exit(2)
	}
	pidFile := os.Getenv(windowsHelperPIDFileEnv)
	child := exec.Command(os.Args[0], "-test.run=^TestWindowsProcessTreeHelper$")
	child.Env = replaceEnv(os.Environ(), windowsHelperModeEnv, "child")
	if err := child.Start(); err != nil {
		os.Exit(3)
	}
	content := fmt.Sprintf("%d\n%d\n", os.Getpid(), child.Process.Pid)
	if err := os.WriteFile(pidFile, []byte(content), 0o600); err != nil {
		os.Exit(4)
	}
	time.Sleep(10 * time.Minute)
}

func TestRunTimeoutTerminatesWindowsProcessTree(t *testing.T) {
	pidFile := t.TempDir() + `\pids.txt`
	env := replaceEnv(os.Environ(), windowsHelperModeEnv, "parent")
	env = replaceEnv(env, windowsHelperPIDFileEnv, pidFile)
	res, err := Run(context.Background(), Options{
		Program:       os.Args[0],
		Args:          []string{"-test.run=^TestWindowsProcessTreeHelper$"},
		Env:           env,
		Timeout:       5 * time.Second,
		KillGrace:     100 * time.Millisecond,
		KillWaitGrace: 5 * time.Second,
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Run() error = %v, want deadline exceeded", err)
	}
	if !res.TimedOut {
		t.Fatalf("Run() TimedOut = false")
	}
	data, readErr := os.ReadFile(pidFile)
	if readErr != nil {
		t.Fatalf("read helper pid file: %v", readErr)
	}
	fields := strings.Fields(string(data))
	if len(fields) != 2 {
		t.Fatalf("pid file = %q, want parent and child pid", data)
	}
	for _, field := range fields {
		pid, parseErr := strconv.Atoi(field)
		if parseErr != nil {
			t.Fatalf("parse pid %q: %v", field, parseErr)
		}
		deadline := time.Now().Add(5 * time.Second)
		for windowsProcessAlive(pid) && time.Now().Before(deadline) {
			time.Sleep(25 * time.Millisecond)
		}
		if windowsProcessAlive(pid) {
			t.Fatalf("process %d remains alive after command timeout", pid)
		}
	}
}

func replaceEnv(env []string, key, value string) []string {
	prefix := strings.ToUpper(key) + "="
	out := make([]string, 0, len(env)+1)
	for _, entry := range env {
		if strings.HasPrefix(strings.ToUpper(entry), prefix) {
			continue
		}
		out = append(out, entry)
	}
	return append(out, key+"="+value)
}

func windowsProcessAlive(pid int) bool {
	const synchronize = 0x00100000
	handle, err := syscall.OpenProcess(synchronize, false, uint32(pid))
	if err != nil {
		return false
	}
	defer syscall.CloseHandle(handle)
	status, err := syscall.WaitForSingleObject(handle, 0)
	if err != nil {
		return true
	}
	return status == syscall.WAIT_TIMEOUT
}
