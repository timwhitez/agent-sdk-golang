//go:build !unix

package sandbox

import "os"

// fileOwnerIDs reports no ownership information on platforms that do not expose
// uid/gid through os.FileInfo.
func fileOwnerIDs(os.FileInfo) (uid int, gid int, ok bool) {
	return 0, 0, false
}
