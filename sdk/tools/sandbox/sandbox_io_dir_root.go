package sandbox

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

// rootAtomicDir is the portable anchored directory (the default off Linux, and
// exercised on Linux by the tests). It opens the allowed root containing the
// target and walks only the components below it: each one is lstat'ed through
// its pinned parent, opened with Root.OpenRoot and accepted only if the opened
// directory is the same file the lstat saw. A symlink, a swapped component or a
// component that cannot be opened (os.Root needs read permission) fails closed;
// nothing at or below the allowed root is resolved by path.
type rootAtomicDir struct{ r *os.Root }

func openRootAtomicDir(anchor, dir string, create bool) (atomicDir, error) {
	anchor, dir = filepath.Clean(anchor), filepath.Clean(dir)
	rel, err := filepath.Rel(anchor, dir)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return nil, &SecurityError{Message: fmt.Sprintf("directory outside its allowed root: %q (root %q)", dir, anchor)}
	}
	// The anchor itself is opened by path: it is the allowed root (or its
	// parent), whose ancestors are outside the sandbox's concern.
	info, err := os.Lstat(anchor)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, &SecurityError{Message: fmt.Sprintf("symlink or non-directory path component denied: %q", anchor)}
	}
	r, err := os.OpenRoot(anchor)
	if err != nil {
		return nil, anchorOpenError(anchor, err)
	}
	if st, err := r.Stat("."); err != nil || !os.SameFile(info, st) {
		_ = r.Close()
		return nil, &SecurityError{Message: fmt.Sprintf("path changed during open: %q", anchor)}
	}
	walked := anchor
	for _, name := range strings.Split(rel, string(filepath.Separator)) {
		if name == "" || name == "." {
			continue
		}
		walked = filepath.Join(walked, name)
		next, err := openRootStep(r, walked, name, create)
		_ = r.Close()
		if err != nil {
			return nil, err
		}
		r = next
	}
	return rootAtomicDir{r}, nil
}

func anchorOpenError(path string, err error) error {
	if errors.Is(err, fs.ErrPermission) {
		return &SecurityError{Message: fmt.Sprintf("cannot pin directory %q without read permission: %v", path, err)}
	}
	return err
}

// openRootStep opens the component name (at path walked) below parent.
func openRootStep(parent *os.Root, walked, name string, create bool) (*os.Root, error) {
	info, err := parent.Lstat(name)
	var mode os.FileMode
	created := false
	if create && errors.Is(err, fs.ErrNotExist) {
		pinfo, err := parent.Stat(".")
		if err != nil {
			return nil, err
		}
		mode = pinfo.Mode().Perm()
		beforeAnchorMkdir()
		err = parent.Mkdir(name, mode)
		// A directory that appeared concurrently keeps its creator's mode.
		created = err == nil
		if err != nil && !errors.Is(err, fs.ErrExist) {
			return nil, err
		}
		info, err = parent.Lstat(name)
		if err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, &SecurityError{Message: fmt.Sprintf("symlink or non-directory path component denied: %q", walked)}
	}
	child, err := parent.OpenRoot(name)
	if err != nil {
		return nil, anchorOpenError(walked, err)
	}
	if st, err := child.Stat("."); err != nil || !os.SameFile(info, st) {
		_ = child.Close()
		if err != nil {
			return nil, err
		}
		return nil, &SecurityError{Message: fmt.Sprintf("path changed during open: %q", walked)}
	}
	if created {
		// Chmod explicitly so the inherited bits survive the umask. An inherited
		// mode without owner read makes the next component fail closed above.
		if err := child.Chmod(".", mode); err != nil {
			_ = child.Close()
			return nil, err
		}
	}
	return child, nil
}

func (d rootAtomicDir) lstat(name string) (os.FileInfo, error) { return d.r.Lstat(name) }

func (d rootAtomicDir) createTemp() (*os.File, string, error) {
	name, err := atomicTempName()
	if err != nil {
		return nil, "", err
	}
	f, err := d.r.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return nil, "", err
	}
	return f, name, nil
}

func (d rootAtomicDir) rename(tmpName, name string) error { return d.r.Rename(tmpName, name) }

// link never resolves a multi-component path: os.Root cannot link between two
// roots, so across directories the source is copied (content, mode, best-effort
// owner) into a temp file in the pinned destination, which is then hard-linked
// to dstName without replacing anything. A non-regular source is refused.
func (d rootAtomicDir) link(name string, dst atomicDir, dstName string) error {
	to, ok := dst.(rootAtomicDir)
	if !ok {
		return errors.New("link: mixed atomic directory implementations")
	}
	src, err := d.r.Lstat(name)
	if err != nil {
		return err
	}
	if !src.Mode().IsRegular() {
		return &SecurityError{Message: fmt.Sprintf("move source is not a regular file: %q", name)}
	}
	fromDir, err1 := d.r.Stat(".")
	toDir, err2 := to.r.Stat(".")
	if err1 == nil && err2 == nil && os.SameFile(fromDir, toDir) {
		return d.r.Link(name, dstName)
	}
	f, err := d.r.OpenFile(name, os.O_RDONLY, 0)
	if err != nil {
		return err
	}
	defer f.Close()
	if st, err := f.Stat(); err != nil || !os.SameFile(src, st) {
		return &SecurityError{Message: fmt.Sprintf("path changed during open: %q", name)}
	}
	data, err := io.ReadAll(f)
	if err != nil {
		return err
	}
	tmp, tmpName, err := to.createTemp()
	if err != nil {
		return err
	}
	defer func() { _ = to.r.Remove(tmpName) }()
	n, err := writeFileBytes(tmp, data)
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	if err == nil {
		err = tmp.Chmod(src.Mode().Perm())
	}
	if uid, gid, ok := fileOwnerIDs(src); ok && err == nil {
		_ = tmp.Chown(uid, gid) // best effort, as for writes
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return to.r.Link(tmpName, dstName)
}

// remove refuses directories so it matches unlinkat (Root.Remove would also
// remove an empty directory).
func (d rootAtomicDir) remove(name string) error {
	info, err := d.r.Lstat(name)
	if err != nil {
		return err
	}
	if info.IsDir() {
		return &os.PathError{Op: "remove", Path: name, Err: syscall.EISDIR}
	}
	return d.r.Remove(name)
}

func (d rootAtomicDir) close() { _ = d.r.Close() }
