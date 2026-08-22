from pathlib import Path

runner = Path("sdk/tools/execrunner/runner.go")
text = runner.read_text()
old = '''\tif err := cmd.Start(); err != nil {
\t\tcollector.Close()
\t\treturn res, err
\t}

\twaitCh := make(chan error, 1)
'''
new = '''\tif err := cmd.Start(); err != nil {
\t\tcollector.Close()
\t\treturn res, err
\t}
\tprocessGroup, err := attachProcessGroup(cmd.Process)
\tif err != nil {
\t\t// Fail closed: a command that could not be placed inside the platform
\t\t// process-tree boundary must not continue running unsupervised.
\t\t_ = cmd.Process.Kill()
\t\t_ = cmd.Wait()
\t\tcollector.Close()
\t\tcanonical.finish(context.WithoutCancel(ctx))
\t\treturn res, fmt.Errorf("establish process-tree boundary: %w", err)
\t}
\tdefer processGroup.close()

\twaitCh := make(chan error, 1)
'''
if text.count(old) != 1:
    raise SystemExit(f"runner start anchor count={text.count(old)}")
text = text.replace(old, new)
text = text.replace('_ = signalProcessGroupTerminate(cmd.Process)', '_ = processGroup.terminate()')
text = text.replace('_ = signalProcessGroupKill(cmd.Process)', '_ = processGroup.kill()')
runner.write_text(text)

Path("sdk/tools/execrunner/process_group_windows.go").write_text(r'''//go:build windows

package execrunner

import (
	"fmt"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"unsafe"
)

const (
	jobObjectExtendedLimitInformation = 9
	jobObjectLimitKillOnJobClose       = 0x00002000
	processSetQuota                    = 0x0100
	processTerminate                   = 0x0001
)

var (
	kernel32DLL                  = syscall.NewLazyDLL("kernel32.dll")
	procCreateJobObjectW         = kernel32DLL.NewProc("CreateJobObjectW")
	procSetInformationJobObject  = kernel32DLL.NewProc("SetInformationJobObject")
	procAssignProcessToJobObject = kernel32DLL.NewProc("AssignProcessToJobObject")
	procTerminateJobObject       = kernel32DLL.NewProc("TerminateJobObject")
)

type jobObjectBasicLimitInformation struct {
	PerProcessUserTimeLimit int64
	PerJobUserTimeLimit     int64
	LimitFlags              uint32
	MinimumWorkingSetSize   uintptr
	MaximumWorkingSetSize   uintptr
	ActiveProcessLimit      uint32
	Affinity                uintptr
	PriorityClass           uint32
	SchedulingClass         uint32
}

type ioCounters struct {
	ReadOperationCount  uint64
	WriteOperationCount uint64
	OtherOperationCount uint64
	ReadTransferCount   uint64
	WriteTransferCount  uint64
	OtherTransferCount  uint64
}

type jobObjectExtendedLimitInfo struct {
	BasicLimitInformation jobObjectBasicLimitInformation
	IOInfo                ioCounters
	ProcessMemoryLimit    uintptr
	JobMemoryLimit        uintptr
	PeakProcessMemoryUsed uintptr
	PeakJobMemoryUsed     uintptr
}

type processGroupController struct {
	mu      sync.Mutex
	process *os.Process
	job     syscall.Handle
	closed  bool
}

func configureProcessGroup(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP
}

func attachProcessGroup(proc *os.Process) (*processGroupController, error) {
	if proc == nil {
		return nil, fmt.Errorf("missing process")
	}
	jobRaw, _, callErr := procCreateJobObjectW.Call(0, 0)
	if jobRaw == 0 {
		return nil, windowsCallError("CreateJobObjectW", callErr)
	}
	job := syscall.Handle(jobRaw)
	closeJob := true
	defer func() {
		if closeJob {
			_ = syscall.CloseHandle(job)
		}
	}()

	info := jobObjectExtendedLimitInfo{}
	info.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose
	ok, _, callErr := procSetInformationJobObject.Call(
		uintptr(job),
		jobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		unsafe.Sizeof(info),
	)
	if ok == 0 {
		return nil, windowsCallError("SetInformationJobObject", callErr)
	}

	processHandle, err := syscall.OpenProcess(processSetQuota|processTerminate, false, uint32(proc.Pid))
	if err != nil {
		return nil, fmt.Errorf("OpenProcess(%d): %w", proc.Pid, err)
	}
	defer syscall.CloseHandle(processHandle)
	ok, _, callErr = procAssignProcessToJobObject.Call(uintptr(job), uintptr(processHandle))
	if ok == 0 {
		return nil, windowsCallError("AssignProcessToJobObject", callErr)
	}

	closeJob = false
	return &processGroupController{process: proc, job: job}, nil
}

func (c *processGroupController) terminate() error {
	return c.terminateJob()
}

func (c *processGroupController) kill() error {
	return c.terminateJob()
}

func (c *processGroupController) terminateJob() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.job == 0 {
		return nil
	}
	ok, _, callErr := procTerminateJobObject.Call(uintptr(c.job), 1)
	if ok == 0 {
		return windowsCallError("TerminateJobObject", callErr)
	}
	return nil
}

func (c *processGroupController) close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.job == 0 {
		return nil
	}
	c.closed = true
	err := syscall.CloseHandle(c.job)
	c.job = 0
	return err
}

func windowsCallError(name string, callErr error) error {
	if callErr == nil || callErr == syscall.Errno(0) {
		callErr = syscall.GetLastError()
	}
	return fmt.Errorf("%s: %w", name, callErr)
}
''')

unix = Path("sdk/tools/execrunner/process_group_unix.go")
unix_text = unix.read_text()
unix_text += r'''

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
'''
unix.write_text(unix_text)

other = Path("sdk/tools/execrunner/process_group_other.go")
other_text = other.read_text()
other_text += r'''

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
'''
other.write_text(other_text)

Path("sdk/tools/execrunner/process_tree_windows_test.go").write_text(r'''//go:build windows

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
		Program: os.Args[0],
		Args: []string{"-test.run=^TestWindowsProcessTreeHelper$"},
		Env: env,
		Timeout: 5 * time.Second,
		KillGrace: 100 * time.Millisecond,
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
''')
