//go:build unix

package sandbox

import "syscall"

const openNonblock = syscall.O_NONBLOCK
