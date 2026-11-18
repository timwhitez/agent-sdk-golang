//go:build linux

package sandbox

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

var openAtomicDir = openFDAtomicDir

// oPath is O_PATH, which the stdlib syscall package does not export: it lets
// lstat open a symlink leaf itself without needing read permission.
const oPath = 0x200000

// fdAtomicDir pins the destination directory with a directory fd.
type fdAtomicDir struct{ fd int }

// openFDAtomicDir opens dir one component at a time with O_NOFOLLOW|O_DIRECTORY,
// creating missing directories (when create is set) with the mode of their
// existing parent. Any symlink (or non-directory) component fails closed. Components are opened with
// O_PATH so only search (x) permission is needed, as with a path-based open;
// openat, mkdirat, renameat, unlinkat and fstat all accept O_PATH directory fds.
func openFDAtomicDir(dir string, create bool) (atomicDir, error) {
	dir = filepath.Clean(dir)
	start := "."
	if filepath.IsAbs(dir) {
		start = string(filepath.Separator)
	}
	const flags = oPath | syscall.O_DIRECTORY | syscall.O_NOFOLLOW | syscall.O_CLOEXEC
	fd, err := syscall.Open(start, flags, 0)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: start, Err: err}
	}
	walked := start
	for _, name := range strings.Split(dir, string(filepath.Separator)) {
		if name == "" || name == "." {
			continue
		}
		walked = filepath.Join(walked, name)
		next, err := syscall.Openat(fd, name, flags, 0)
		if err == syscall.ENOENT && create {
			next, err = mkdirAtInherit(fd, name)
		}
		if err != nil {
			_ = syscall.Close(fd)
			if err == syscall.ELOOP || err == syscall.ENOTDIR {
				return nil, &SecurityError{Message: fmt.Sprintf("symlink or non-directory path component denied: %q", walked)}
			}
			return nil, &os.PathError{Op: "open", Path: walked, Err: err}
		}
		_ = syscall.Close(fd)
		fd = next
	}
	return fdAtomicDir{fd}, nil
}

// mkdirAtInherit creates name under parent with the parent's permission bits
// and returns an O_PATH fd for it. A directory that appeared concurrently
// (EEXIST) is opened as is: its mode belongs to whoever created it.
func mkdirAtInherit(parent int, name string) (int, error) {
	var st syscall.Stat_t
	if err := syscall.Fstat(parent, &st); err != nil {
		return -1, err
	}
	mode := uint32(st.Mode & 0o777)
	beforeAnchorMkdir()
	if err := syscall.Mkdirat(parent, name, mode); err == syscall.EEXIST {
		return syscall.Openat(parent, name, oPath|syscall.O_DIRECTORY|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
	} else if err != nil {
		return -1, err
	}
	// Chmod explicitly so the inherited bits survive the umask. O_PATH fds cannot
	// be chmod'ed, so use a short-lived read fd; if the new directory is not
	// readable by us the umask-reduced mode stays.
	if rfd, err := syscall.Openat(parent, name, syscall.O_RDONLY|syscall.O_DIRECTORY|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0); err == nil {
		err = syscall.Fchmod(rfd, mode)
		_ = syscall.Close(rfd)
		if err != nil {
			return -1, err
		}
	} else if err != syscall.EACCES {
		return -1, err
	}
	return syscall.Openat(parent, name, oPath|syscall.O_DIRECTORY|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
}

func (d fdAtomicDir) lstat(name string) (os.FileInfo, error) {
	fd, err := syscall.Openat(d.fd, name, oPath|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, &os.PathError{Op: "lstat", Path: name, Err: err}
	}
	f := os.NewFile(uintptr(fd), name)
	defer f.Close()
	return f.Stat()
}

func (d fdAtomicDir) createTemp() (*os.File, string, error) {
	name, err := atomicTempName()
	if err != nil {
		return nil, "", err
	}
	fd, err := syscall.Openat(d.fd, name, syscall.O_RDWR|syscall.O_CREAT|syscall.O_EXCL|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0o600)
	if err != nil {
		return nil, "", &os.PathError{Op: "createtemp", Path: name, Err: err}
	}
	return os.NewFile(uintptr(fd), name), name, nil
}

func (d fdAtomicDir) rename(tmpName, name string) error {
	if err := syscall.Renameat(d.fd, tmpName, d.fd, name); err != nil {
		return &os.LinkError{Op: "rename", Old: tmpName, New: name, Err: err}
	}
	return nil
}

func (d fdAtomicDir) link(name string, dst atomicDir, dstName string) error {
	to, ok := dst.(fdAtomicDir)
	if !ok {
		return errors.New("link: mixed atomic directory implementations")
	}
	oldp, err := syscall.BytePtrFromString(name)
	if err != nil {
		return err
	}
	newp, err := syscall.BytePtrFromString(dstName)
	if err != nil {
		return err
	}
	// The stdlib syscall package does not export linkat; flags 0 never follows a
	// symlink source.
	if _, _, e := syscall.Syscall6(syscall.SYS_LINKAT, uintptr(d.fd), uintptr(unsafe.Pointer(oldp)), uintptr(to.fd), uintptr(unsafe.Pointer(newp)), 0, 0); e != 0 {
		return &os.LinkError{Op: "link", Old: name, New: dstName, Err: e}
	}
	return nil
}

func (d fdAtomicDir) remove(name string) error {
	if err := syscall.Unlinkat(d.fd, name); err != nil {
		return &os.PathError{Op: "unlinkat", Path: name, Err: err}
	}
	return nil
}

func (d fdAtomicDir) close() { _ = syscall.Close(d.fd) }
