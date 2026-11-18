//go:build windows

package execrunner

import (
	"fmt"
	"os"
	"os/exec"
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

// processGroupController is the Job Object controller; the logic is in process_job.go.
type processGroupController = jobController

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
	return attachJob(winJobOps{}, proc.Pid)
}

// winJobOps is the Win32 glue behind jobOps.
type winJobOps struct{}

func (winJobOps) createJob() (uintptr, error) {
	job, _, callErr := procCreateJobObjectW.Call(0, 0)
	if job == 0 {
		return 0, windowsCallError("CreateJobObjectW", callErr)
	}
	return job, nil
}

func (winJobOps) setKillOnClose(job uintptr, on bool) error {
	info := jobObjectExtendedLimitInfo{}
	if on {
		info.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose
	}
	ok, _, callErr := procSetInformationJobObject.Call(
		job,
		jobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		unsafe.Sizeof(info),
	)
	if ok == 0 {
		return windowsCallError("SetInformationJobObject", callErr)
	}
	return nil
}

func (winJobOps) openProcess(pid int) (uintptr, error) {
	h, err := syscall.OpenProcess(processSetQuota|processTerminate|processSuspendResume, false, uint32(pid))
	return uintptr(h), err
}

func (winJobOps) assign(job, proc uintptr) error {
	ok, _, callErr := procAssignProcessToJobObject.Call(job, proc)
	if ok == 0 {
		return windowsCallError("AssignProcessToJobObject", callErr)
	}
	return nil
}

func (winJobOps) resume(proc uintptr) error {
	status, _, _ := procNtResumeProcess.Call(proc)
	if status != 0 {
		return fmt.Errorf("NtResumeProcess: NTSTATUS 0x%08X", uint32(status))
	}
	return nil
}

func (winJobOps) terminate(job uintptr) error {
	ok, _, callErr := procTerminateJobObject.Call(job, 1)
	if ok == 0 {
		return windowsCallError("TerminateJobObject", callErr)
	}
	return nil
}

func (winJobOps) closeHandle(h uintptr) error { return syscall.CloseHandle(syscall.Handle(h)) }

func windowsCallError(name string, callErr error) error {
	if callErr == nil || callErr == syscall.Errno(0) {
		callErr = syscall.GetLastError()
	}
	return fmt.Errorf("%s: %w", name, callErr)
}
