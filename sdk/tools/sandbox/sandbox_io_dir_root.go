package sandbox

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// rootAtomicDir is the portable anchored directory (the default off Linux, and
// exercised on Linux by the tests). Each component is lstat'ed through its
// already-pinned parent, opened with Root.OpenRoot and accepted only if the
// opened directory is the same file the lstat saw, so a component swapped for a
// symlink fails closed instead of being followed.
type rootAtomicDir struct {
	r    *os.Root
	path string
}

func openRootAtomicDir(dir string, create bool) (atomicDir, error) {
	dir = filepath.Clean(dir)
	start, rest := ".", dir
	if filepath.IsAbs(dir) {
		start = filepath.VolumeName(dir) + string(filepath.Separator)
		rest = strings.TrimPrefix(dir[len(filepath.VolumeName(dir)):], string(filepath.Separator))
	}
	r, err := os.OpenRoot(start)
	if err != nil && !errors.Is(err, fs.ErrPermission) {
		return nil, err
	}
	walked := start
	for _, name := range strings.Split(rest, string(filepath.Separator)) {
		if name == "" || name == "." {
			continue
		}
		walked = filepath.Join(walked, name)
		next, err := openRootStep(r, walked, name, create)
		if r != nil {
			_ = r.Close()
		}
		if err != nil {
			return nil, err
		}
		r = next
	}
	if r == nil {
		return nil, &os.PathError{Op: "open", Path: dir, Err: fs.ErrPermission}
	}
	return rootAtomicDir{r: r, path: dir}, nil
}

// openRootStep opens name (at path walked) below parent. A nil parent means the
// parent directory is not readable (os.Root needs read access, unlike the Linux
// O_PATH walk); then the step is path-based. shortcut: a component below a
// non-readable ancestor can still be swapped between the lstat and open by
// path, upgrade if such layouts need full anchoring off Linux. A nil, nil
// result means this directory is not readable either.
func openRootStep(parent *os.Root, walked, name string, create bool) (*os.Root, error) {
	lstat := func() (os.FileInfo, error) {
		if parent != nil {
			return parent.Lstat(name)
		}
		return os.Lstat(walked)
	}
	info, err := lstat()
	var mode os.FileMode
	created := false
	if create && errors.Is(err, fs.ErrNotExist) {
		var pinfo os.FileInfo
		if parent != nil {
			pinfo, err = parent.Stat(".")
		} else {
			pinfo, err = os.Stat(filepath.Dir(walked))
		}
		if err != nil {
			return nil, err
		}
		mode = pinfo.Mode().Perm()
		beforeAnchorMkdir()
		if parent != nil {
			err = parent.Mkdir(name, mode)
		} else {
			err = os.Mkdir(walked, mode)
		}
		// A directory that appeared concurrently keeps its creator's mode.
		created = err == nil
		if err != nil && !errors.Is(err, fs.ErrExist) {
			return nil, err
		}
		info, err = lstat()
	}
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, &SecurityError{Message: fmt.Sprintf("symlink or non-directory path component denied: %q", walked)}
	}
	var child *os.Root
	if parent != nil {
		child, err = parent.OpenRoot(name)
	} else {
		child, err = os.OpenRoot(walked)
	}
	if errors.Is(err, fs.ErrPermission) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if st, err := child.Stat("."); err != nil || !os.SameFile(info, st) {
		_ = child.Close()
		if err != nil {
			return nil, err
		}
		return nil, &SecurityError{Message: fmt.Sprintf("path changed during open: %q", walked)}
	}
	if created {
		// Chmod explicitly so the inherited bits survive the umask.
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

// link hard-links name into dst. os.Root cannot link between two roots, so the
// link runs in a root anchored at the deepest common ancestor of both
// directories. Inside that root os.Root resolves the relative paths again and
// would follow a symlink swapped in after the two directories were pinned
// (staying within the ancestor); the result is therefore checked against the
// pinned directories and rolled back on mismatch.
func (d rootAtomicDir) link(name string, dst atomicDir, dstName string) error {
	to, ok := dst.(rootAtomicDir)
	if !ok {
		return errors.New("link: mixed atomic directory implementations")
	}
	anc := d.path
	for {
		if rel, err := filepath.Rel(anc, to.path); err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			break
		}
		parent := filepath.Dir(anc)
		if parent == anc {
			return fmt.Errorf("link: no common ancestor for %q and %q", d.path, to.path)
		}
		anc = parent
	}
	ad, err := openRootAtomicDir(anc, false)
	if err != nil {
		return err
	}
	defer ad.close()
	common := ad.(rootAtomicDir).r
	relFrom, err := filepath.Rel(anc, filepath.Join(d.path, name))
	if err != nil {
		return err
	}
	relTo, err := filepath.Rel(anc, filepath.Join(to.path, dstName))
	if err != nil {
		return err
	}
	if err := common.Link(relFrom, relTo); err != nil {
		return err
	}
	src, srcErr := d.r.Lstat(name)
	got, gotErr := to.r.Lstat(dstName)
	if srcErr != nil || gotErr != nil || !os.SameFile(src, got) {
		_ = common.Remove(relTo)
		return &SecurityError{Message: fmt.Sprintf("path changed during link: %q -> %q", filepath.Join(d.path, name), filepath.Join(to.path, dstName))}
	}
	return nil
}

func (d rootAtomicDir) remove(name string) error { return d.r.Remove(name) }

func (d rootAtomicDir) close() { _ = d.r.Close() }
