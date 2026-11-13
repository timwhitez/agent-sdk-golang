//go:build unix

package agent

import (
	"fmt"
	"os"
	"syscall"
)

func validateToolResultDumpOwner(info os.FileInfo) error {
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat.Uid != uint32(os.Geteuid()) {
		return fmt.Errorf("tool result dump directory must be owned by the current user")
	}
	return nil
}
