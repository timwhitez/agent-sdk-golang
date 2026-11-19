//go:build windows && 386

package execrunner

import "unsafe"

// jobObjectBasicLimitInformation mirrors Win32 JOBOBJECT_BASIC_LIMIT_INFORMATION
// on windows/386. MSVC x86 aligns LARGE_INTEGER (int64) to 8 bytes, rounding the
// struct size from 44 up to 48 bytes; Go on 386 aligns int64/uint64 to 4 bytes,
// so explicit 4-byte tail padding is required to keep IOInfo at offset 48,
// ProcessMemoryLimit at offset 96, and total size at 112 bytes.
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
	_                       uint32
}

var (
	_ [48 - unsafe.Sizeof(jobObjectBasicLimitInformation{})]byte
	_ [unsafe.Sizeof(jobObjectBasicLimitInformation{}) - 48]byte
	_ [48 - unsafe.Offsetof(jobObjectExtendedLimitInfo{}.IOInfo)]byte
	_ [unsafe.Offsetof(jobObjectExtendedLimitInfo{}.IOInfo) - 48]byte
	_ [96 - unsafe.Offsetof(jobObjectExtendedLimitInfo{}.ProcessMemoryLimit)]byte
	_ [unsafe.Offsetof(jobObjectExtendedLimitInfo{}.ProcessMemoryLimit) - 96]byte
	_ [112 - unsafe.Sizeof(jobObjectExtendedLimitInfo{})]byte
	_ [unsafe.Sizeof(jobObjectExtendedLimitInfo{}) - 112]byte
)
