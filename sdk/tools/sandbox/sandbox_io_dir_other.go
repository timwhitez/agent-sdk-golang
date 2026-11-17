//go:build !linux

package sandbox

import (
	"fmt"
	"os"
	"path/filepath"
)

// pathAtomicDir is the path-based fallback: it re-checks that dir is not a
// symlink around each step, which narrows but cannot close the swap window.
// shortcut: no dirfd anchoring off Linux (stdlib syscall lacks openat there);
// upgrade by anchoring with os.Root once go.mod requires Go 1.25 (Root.Rename).
type pathAtomicDir struct{ dir string }

func openAtomicDir(dir string) (atomicDir, error) {
	if info, err := os.Lstat(dir); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return nil, &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", dir)}
	} else if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	if err := mkdirAllInheritMode(dir); err != nil {
		return nil, err
	}
	if err := ensureWriteDirNoFollow(dir); err != nil {
		return nil, err
	}
	return pathAtomicDir{dir}, nil
}

func (d pathAtomicDir) lstat(name string) (os.FileInfo, error) {
	return os.Lstat(filepath.Join(d.dir, name))
}

func (d pathAtomicDir) createTemp() (*os.File, string, error) {
	tmp, err := os.CreateTemp(d.dir, ".tmp-*")
	if err != nil {
		return nil, "", err
	}
	if err := ensureWriteDirNoFollow(d.dir); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return nil, "", err
	}
	return tmp, filepath.Base(tmp.Name()), nil
}

func (d pathAtomicDir) rename(tmpName, name string) error {
	if err := ensureWriteDirNoFollow(d.dir); err != nil {
		return err
	}
	return os.Rename(filepath.Join(d.dir, tmpName), filepath.Join(d.dir, name))
}

func (d pathAtomicDir) remove(name string) { _ = os.Remove(filepath.Join(d.dir, name)) }

func (d pathAtomicDir) close() {}
