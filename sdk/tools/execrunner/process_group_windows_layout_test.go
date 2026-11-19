//go:build windows

package execrunner

import (
	"testing"
	"unsafe"
)

const (
	winPtrSize = unsafe.Sizeof(uintptr(0))

	// Win32 ABI layout for JOBOBJECT_EXTENDED_LIMIT_INFORMATION (winnt.h):
	//   - amd64/arm64 (winPtrSize=8): basic 64, IOInfo @ 64, ProcessMemoryLimit @ 112, total 144
	//   - 386         (winPtrSize=4): basic 48, IOInfo @ 48, ProcessMemoryLimit @ 96,  total 112
	expectedBasicLimitSize     = 32 + 4*winPtrSize
	expectedIOInfoOffset       = 32 + 4*winPtrSize
	expectedProcessMemOffset   = 80 + 4*winPtrSize
	expectedExtendedLimitSize  = 80 + 8*winPtrSize
	expectedMinWorkingSetOff   = 16 + winPtrSize
	expectedMaxWorkingSetOff   = expectedMinWorkingSetOff + winPtrSize
	expectedActiveProcLimitOff = expectedMaxWorkingSetOff + winPtrSize
	expectedAffinityOff        = expectedActiveProcLimitOff + winPtrSize
	expectedPriorityClassOff   = expectedAffinityOff + winPtrSize
	expectedSchedulingClassOff = expectedPriorityClassOff + 4
	expectedJobMemOffset       = expectedProcessMemOffset + winPtrSize
	expectedPeakProcMemOffset  = expectedJobMemOffset + winPtrSize
	expectedPeakJobMemOffset   = expectedPeakProcMemOffset + winPtrSize
)

// Compile-time layout assertions checked on Linux via
// `GOOS=windows GOARCH={386,amd64,arm64} go vet ./...` without needing a
// Windows host to execute the test binary.
var (
	// Basic limit header field offsets and size.
	_ [0 - unsafe.Offsetof(jobObjectBasicLimitInformation{}.PerProcessUserTimeLimit)]byte
	_ [8 - unsafe.Offsetof(jobObjectBasicLimitInformation{}.PerJobUserTimeLimit)]byte
	_ [unsafe.Offsetof(jobObjectBasicLimitInformation{}.PerJobUserTimeLimit) - 8]byte
	_ [16 - unsafe.Offsetof(jobObjectBasicLimitInformation{}.LimitFlags)]byte
	_ [unsafe.Offsetof(jobObjectBasicLimitInformation{}.LimitFlags) - 16]byte
	_ [expectedMinWorkingSetOff - unsafe.Offsetof(jobObjectBasicLimitInformation{}.MinimumWorkingSetSize)]byte
	_ [unsafe.Offsetof(jobObjectBasicLimitInformation{}.MinimumWorkingSetSize) - expectedMinWorkingSetOff]byte
	_ [expectedMaxWorkingSetOff - unsafe.Offsetof(jobObjectBasicLimitInformation{}.MaximumWorkingSetSize)]byte
	_ [unsafe.Offsetof(jobObjectBasicLimitInformation{}.MaximumWorkingSetSize) - expectedMaxWorkingSetOff]byte
	_ [expectedActiveProcLimitOff - unsafe.Offsetof(jobObjectBasicLimitInformation{}.ActiveProcessLimit)]byte
	_ [unsafe.Offsetof(jobObjectBasicLimitInformation{}.ActiveProcessLimit) - expectedActiveProcLimitOff]byte
	_ [expectedAffinityOff - unsafe.Offsetof(jobObjectBasicLimitInformation{}.Affinity)]byte
	_ [unsafe.Offsetof(jobObjectBasicLimitInformation{}.Affinity) - expectedAffinityOff]byte
	_ [expectedPriorityClassOff - unsafe.Offsetof(jobObjectBasicLimitInformation{}.PriorityClass)]byte
	_ [unsafe.Offsetof(jobObjectBasicLimitInformation{}.PriorityClass) - expectedPriorityClassOff]byte
	_ [expectedSchedulingClassOff - unsafe.Offsetof(jobObjectBasicLimitInformation{}.SchedulingClass)]byte
	_ [unsafe.Offsetof(jobObjectBasicLimitInformation{}.SchedulingClass) - expectedSchedulingClassOff]byte
	_ [expectedBasicLimitSize - unsafe.Sizeof(jobObjectBasicLimitInformation{})]byte
	_ [unsafe.Sizeof(jobObjectBasicLimitInformation{}) - expectedBasicLimitSize]byte

	// IO_COUNTERS size.
	_ [48 - unsafe.Sizeof(ioCounters{})]byte
	_ [unsafe.Sizeof(ioCounters{}) - 48]byte

	// Extended limit info offsets and total size.
	_ [0 - unsafe.Offsetof(jobObjectExtendedLimitInfo{}.BasicLimitInformation)]byte
	_ [expectedIOInfoOffset - unsafe.Offsetof(jobObjectExtendedLimitInfo{}.IOInfo)]byte
	_ [unsafe.Offsetof(jobObjectExtendedLimitInfo{}.IOInfo) - expectedIOInfoOffset]byte
	_ [expectedProcessMemOffset - unsafe.Offsetof(jobObjectExtendedLimitInfo{}.ProcessMemoryLimit)]byte
	_ [unsafe.Offsetof(jobObjectExtendedLimitInfo{}.ProcessMemoryLimit) - expectedProcessMemOffset]byte
	_ [expectedJobMemOffset - unsafe.Offsetof(jobObjectExtendedLimitInfo{}.JobMemoryLimit)]byte
	_ [unsafe.Offsetof(jobObjectExtendedLimitInfo{}.JobMemoryLimit) - expectedJobMemOffset]byte
	_ [expectedPeakProcMemOffset - unsafe.Offsetof(jobObjectExtendedLimitInfo{}.PeakProcessMemoryUsed)]byte
	_ [unsafe.Offsetof(jobObjectExtendedLimitInfo{}.PeakProcessMemoryUsed) - expectedPeakProcMemOffset]byte
	_ [expectedPeakJobMemOffset - unsafe.Offsetof(jobObjectExtendedLimitInfo{}.PeakJobMemoryUsed)]byte
	_ [unsafe.Offsetof(jobObjectExtendedLimitInfo{}.PeakJobMemoryUsed) - expectedPeakJobMemOffset]byte
	_ [expectedExtendedLimitSize - unsafe.Sizeof(jobObjectExtendedLimitInfo{})]byte
	_ [unsafe.Sizeof(jobObjectExtendedLimitInfo{}) - expectedExtendedLimitSize]byte
)

func TestJobObjectExtendedLimitInfoLayout(t *testing.T) {
	if got := unsafe.Sizeof(jobObjectBasicLimitInformation{}); got != expectedBasicLimitSize {
		t.Fatalf("sizeof(jobObjectBasicLimitInformation) = %d, want %d", got, expectedBasicLimitSize)
	}
	if got := unsafe.Offsetof(jobObjectExtendedLimitInfo{}.IOInfo); got != expectedIOInfoOffset {
		t.Fatalf("offsetof(IOInfo) = %d, want %d", got, expectedIOInfoOffset)
	}
	if got := unsafe.Offsetof(jobObjectExtendedLimitInfo{}.ProcessMemoryLimit); got != expectedProcessMemOffset {
		t.Fatalf("offsetof(ProcessMemoryLimit) = %d, want %d", got, expectedProcessMemOffset)
	}
	if got := unsafe.Sizeof(jobObjectExtendedLimitInfo{}); got != expectedExtendedLimitSize {
		t.Fatalf("sizeof(jobObjectExtendedLimitInfo) = %d, want %d", got, expectedExtendedLimitSize)
	}
}
