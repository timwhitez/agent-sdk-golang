//go:build windows

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
	jobObjectLimitKillOnJobClose      = 0x00002000
	processSetQuota                   = 0x0100
	processTerminate                  = 0x0001
	processSuspendResume              = 0x0800
	createSuspended                   = 0x00000004
)

var (
	kernel32DLL                  = syscall.NewLazyDLL("kernel32.dll")
	procCreateJobObjectW         = kernel32DLL.NewProc("CreateJobObjectW")
	procSetInformationJobObject  = kernel32DLL.NewProc("SetInformationJobObject")
	procAssignProcessToJobObject = kernel32DLL.NewProc("AssignProcessToJobObject")
	procTerminateJobObject       = kernel32DLL.NewProc("TerminateJobObject")
	ntdllDLL                     = syscall.NewLazyDLL("ntdll.dll")
	procNtResumeProcess          = ntdllDLL.NewProc("NtResumeProcess")
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
	// The process must not execute user code before it is assigned to the Job
	// Object, otherwise it can spawn a descendant during the Start-to-Assign gap.
	cmd.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP | createSuspended
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

	processHandle, err := syscall.OpenProcess(processSetQuota|processTerminate|processSuspendResume, false, uint32(proc.Pid))
	if err != nil {
		return nil, fmt.Errorf("OpenProcess(%d): %w", proc.Pid, err)
	}
	defer syscall.CloseHandle(processHandle)
	ok, _, callErr = procAssignProcessToJobObject.Call(uintptr(job), uintptr(processHandle))
	if ok == 0 {
		return nil, windowsCallError("AssignProcessToJobObject", callErr)
	}
	status, _, _ := procNtResumeProcess.Call(uintptr(processHandle))
	if status != 0 {
		return nil, fmt.Errorf("NtResumeProcess: NTSTATUS 0x%08X", uint32(status))
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
