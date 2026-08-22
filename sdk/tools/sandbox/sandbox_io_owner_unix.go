//go:build unix

package sandbox

import (
	"os"
	"syscall"
)

// fileOwnerIDs reports the uid/gid recorded in info when the platform exposes
// them through syscall.Stat_t.
func fileOwnerIDs(info os.FileInfo) (uid int, gid int, ok bool) {
	st, castOK := info.Sys().(*syscall.Stat_t)
	if !castOK || st == nil {
		return 0, 0, false
	}
	return int(st.Uid), int(st.Gid), true
}
