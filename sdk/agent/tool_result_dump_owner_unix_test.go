//go:build unix

package agent

import (
	"os"
	"testing"
)

func TestResolveDumpDirRejectsForeignOwner(t *testing.T) {
	if os.Geteuid() != 0 {
		t.Skip("changing owner requires root")
	}
	dir := t.TempDir()
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Chown(dir, 65534, -1); err != nil {
		t.Skip(err)
	}
	defer os.Chown(dir, os.Geteuid(), -1)
	old := toolResultDumpDir
	toolResultDumpDir = func() string { return dir }
	defer func() { toolResultDumpDir = old }()
	if _, err := resolveToolResultDumpDir(); err == nil {
		t.Fatal("foreign owned directory accepted")
	}
}
