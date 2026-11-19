//go:build windows && (amd64 || arm64)

package execrunner

import "unsafe"

// jobObjectBasicLimitInformation mirrors Win32 JOBOBJECT_BASIC_LIMIT_INFORMATION
// on 64-bit Windows (amd64, arm64): size 64, IOInfo at 64, ProcessMemoryLimit
// at 112, and total JOBOBJECT_EXTENDED_LIMIT_INFORMATION size 144.
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

var (
	_ [64 - unsafe.Sizeof(jobObjectBasicLimitInformation{})]byte
	_ [unsafe.Sizeof(jobObjectBasicLimitInformation{}) - 64]byte
	_ [64 - unsafe.Offsetof(jobObjectExtendedLimitInfo{}.IOInfo)]byte
	_ [unsafe.Offsetof(jobObjectExtendedLimitInfo{}.IOInfo) - 64]byte
	_ [112 - unsafe.Offsetof(jobObjectExtendedLimitInfo{}.ProcessMemoryLimit)]byte
	_ [unsafe.Offsetof(jobObjectExtendedLimitInfo{}.ProcessMemoryLimit) - 112]byte
	_ [144 - unsafe.Sizeof(jobObjectExtendedLimitInfo{})]byte
	_ [unsafe.Sizeof(jobObjectExtendedLimitInfo{}) - 144]byte
)
