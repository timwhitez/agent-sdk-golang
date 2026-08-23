from pathlib import Path

path = Path("sdk/tools/execrunner/process_group_windows.go")
text = path.read_text(encoding="utf-8")
text = text.replace(
'''\tprocessSetQuota                    = 0x0100
\tprocessTerminate                   = 0x0001
''',
'''\tprocessSetQuota                    = 0x0100
\tprocessTerminate                   = 0x0001
\tprocessSuspendResume               = 0x0800
\tcreateSuspended                    = 0x00000004
''',
1,
)
text = text.replace(
'''\tprocTerminateJobObject       = kernel32DLL.NewProc("TerminateJobObject")
''',
'''\tprocTerminateJobObject       = kernel32DLL.NewProc("TerminateJobObject")
\tntdllDLL                     = syscall.NewLazyDLL("ntdll.dll")
\tprocNtResumeProcess          = ntdllDLL.NewProc("NtResumeProcess")
''',
1,
)
text = text.replace(
'''\tcmd.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP
''',
'''\t// The process must not execute user code before it is assigned to the Job
\t// Object, otherwise it can spawn a descendant during the Start-to-Assign gap.
\tcmd.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP | createSuspended
''',
1,
)
text = text.replace(
'''\tprocessHandle, err := syscall.OpenProcess(processSetQuota|processTerminate, false, uint32(proc.Pid))
''',
'''\tprocessHandle, err := syscall.OpenProcess(processSetQuota|processTerminate|processSuspendResume, false, uint32(proc.Pid))
''',
1,
)
old = '''\tok, _, callErr = procAssignProcessToJobObject.Call(uintptr(job), uintptr(processHandle))
\tif ok == 0 {
\t\treturn nil, windowsCallError("AssignProcessToJobObject", callErr)
\t}

\tcloseJob = false
'''
new = '''\tok, _, callErr = procAssignProcessToJobObject.Call(uintptr(job), uintptr(processHandle))
\tif ok == 0 {
\t\treturn nil, windowsCallError("AssignProcessToJobObject", callErr)
\t}
\tstatus, _, _ := procNtResumeProcess.Call(uintptr(processHandle))
\tif status != 0 {
\t\treturn nil, fmt.Errorf("NtResumeProcess: NTSTATUS 0x%08X", uint32(status))
\t}

\tcloseJob = false
'''
if text.count(old) != 1:
    raise SystemExit(f"assign/resume anchor count={text.count(old)}")
path.write_text(text.replace(old, new), encoding="utf-8")
