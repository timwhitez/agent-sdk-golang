package sandbox

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ============================================================================
// File I/O Helpers
// ============================================================================

// writeFileBytes is a var for testing write operations.
var writeFileBytes = func(f *os.File, data []byte) (int, error) {
	return f.Write(data)
}

// beforeWriteAnchor is a test hook that runs after the caller revalidated path
// and before the destination directory is opened (writes and patch deletes).
var beforeWriteAnchor = func(path string) {}

// beforeAnchorMkdir is a test hook that runs after an anchored walk found a
// component missing and before it creates it.
var beforeAnchorMkdir = func() {}

// atomicDir is a directory pinned by an anchored walk: it is opened one
// component at a time from the filesystem root without following symlinks, so a
// path component replaced after revalidation cannot redirect a write, link or
// unlink. Linux uses O_PATH directory fds (sandbox_io_dir_linux.go); other
// platforms use os.Root (sandbox_io_dir_root.go). openAtomicDir(dir, create)
// selects the platform implementation; create makes missing directories with
// their parent's permission bits.
type atomicDir interface {
	lstat(name string) (os.FileInfo, error)
	createTemp() (*os.File, string, error)
	rename(tmpName, name string) error
	// link hard-links name to dstName in dst without replacing an existing entry.
	link(name string, dst atomicDir, dstName string) error
	remove(name string) error
	close()
}

func atomicTempName() (string, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return ".tmp-" + hex.EncodeToString(b[:]), nil
}

// removeFileAnchored unlinks path through its anchored parent directory.
func removeFileAnchored(path string) error {
	beforeWriteAnchor(path)
	ad, err := openAtomicDir(filepath.Dir(path), false)
	if err != nil {
		return err
	}
	defer ad.close()
	return ad.remove(filepath.Base(path))
}

// writeFilePreserveMode writes data to path while preserving the existing file
// mode (or using defaultMode if the file doesn't exist). Uses atomic write
// with temp file and rename. Ownership of an existing file is preserved on a
// best-effort basis: rename does not inherit the previous owner, so the temp
// file is chowned back before the swap when the process is privileged enough.
// path must already be symlink-resolved; a symlink on any component fails closed.
func writeFilePreserveMode(path string, data []byte, defaultMode os.FileMode) error {
	beforeWriteAnchor(path)
	dir, name := filepath.Dir(path), filepath.Base(path)
	ad, err := openAtomicDir(dir, true)
	if err != nil {
		return err
	}
	defer ad.close()

	mode := defaultMode
	preserveOwner := false
	var ownerUID, ownerGID int
	if info, err := ad.lstat(name); err == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			return &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", path)}
		}
		mode = info.Mode().Perm()
		ownerUID, ownerGID, preserveOwner = fileOwnerIDs(info)
	} else if !os.IsNotExist(err) {
		return err
	}

	tmp, tmpName, err := ad.createTemp()
	if err != nil {
		return err
	}
	cleanup := func() {
		_ = tmp.Close()
		_ = ad.remove(tmpName)
	}

	n, err := writeFileBytes(tmp, data)
	if err != nil {
		cleanup()
		return err
	}
	if n != len(data) {
		cleanup()
		return io.ErrShortWrite
	}
	if err := tmp.Sync(); err != nil {
		cleanup()
		return err
	}
	if err := tmp.Chmod(mode); err != nil {
		cleanup()
		return err
	}
	if preserveOwner {
		// Best-effort only: unprivileged processes cannot chown, and failing
		// here would break writes that used to succeed. Keep the previous
		// behaviour (owner becomes the current process) in that case.
		_ = tmp.Chown(ownerUID, ownerGID)
	}
	if err := tmp.Close(); err != nil {
		_ = ad.remove(tmpName)
		return err
	}
	if err := ad.rename(tmpName, name); err != nil {
		_ = ad.remove(tmpName)
		return err
	}
	return nil
}

// openFileNoFollow opens path while rejecting symlink leaf targets and
// verifying the opened file still matches the lstat result.
func openFileNoFollow(path string) (*os.File, os.FileInfo, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, nil, err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return nil, nil, &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", path)}
	}
	if !info.Mode().IsRegular() && !info.IsDir() {
		return nil, nil, &SecurityError{Message: fmt.Sprintf("unsupported special file denied: %q", path)}
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	statInfo, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, nil, err
	}
	if !os.SameFile(info, statInfo) {
		_ = f.Close()
		return nil, nil, &SecurityError{Message: fmt.Sprintf("path changed during open: %q", path)}
	}
	return f, statInfo, nil
}

// OpenFileNoFollow opens path while rejecting symlink leaf targets and
// verifying the opened file still matches the lstat result.
func OpenFileNoFollow(path string) (*os.File, os.FileInfo, error) {
	return openFileNoFollow(path)
}

// openReadPath opens a validated sandbox path for reading with double
// validation (before and after open) to detect race conditions.
func (s *Sandbox) openReadPath(path validatedSandboxPath) (*os.File, os.FileInfo, string, error) {
	resolved, err := s.revalidatePathForAccess(path)
	if err != nil {
		return nil, nil, "", err
	}
	f, st, err := openFileNoFollow(resolved)
	if err != nil {
		return nil, nil, "", err
	}
	resolvedAfter, err := s.revalidatePathForAccess(path)
	if err != nil {
		_ = f.Close()
		return nil, nil, "", err
	}
	if !pathsEqual(resolvedAfter, resolved) {
		_ = f.Close()
		return nil, nil, "", &SecurityError{Message: fmt.Sprintf("path changed during access: %q (was %q, now %q)", path.requested, resolved, resolvedAfter)}
	}
	return f, st, resolved, nil
}

// OpenReadAccessPath revalidates an AccessPath and opens it for read using a
// no-follow leaf check.
func (s *Sandbox) OpenReadAccessPath(path AccessPath) (*os.File, os.FileInfo, string, error) {
	return s.openReadPath(path.path)
}

// readAllPath reads the entire contents of a validated sandbox path.
func (s *Sandbox) readAllPath(path validatedSandboxPath) ([]byte, os.FileInfo, string, error) {
	f, st, resolved, err := s.openReadPath(path)
	if err != nil {
		return nil, nil, "", err
	}
	defer f.Close()
	b, err := sandboxReadAll(f)
	if err != nil {
		return nil, nil, "", err
	}
	return b, st, resolved, nil
}

// readAllPathBounded reads a validated sandbox path with a byte limit.
// Returns errFileReadLimitReached if the file is too large.
func (s *Sandbox) readAllPathBounded(path validatedSandboxPath, maxBytes int64) ([]byte, os.FileInfo, string, error) {
	f, st, resolved, err := s.openReadPath(path)
	if err != nil {
		return nil, nil, "", err
	}
	defer f.Close()
	if st.IsDir() {
		return nil, st, resolved, nil
	}
	if maxBytes > 0 && st.Size() > maxBytes {
		return nil, st, resolved, errFileReadLimitReached
	}
	b, err := readAllBounded(f, maxBytes)
	if err != nil {
		return nil, st, resolved, err
	}
	return b, st, resolved, nil
}

// readPathPreviewBounded reads a preview of a validated sandbox path with a byte limit.
// Returns the data, whether it was truncated, and any error.
func (s *Sandbox) readPathPreviewBounded(path validatedSandboxPath, maxBytes int64) ([]byte, os.FileInfo, string, bool, error) {
	f, st, resolved, err := s.openReadPath(path)
	if err != nil {
		return nil, nil, "", false, err
	}
	defer f.Close()
	if st.IsDir() {
		return nil, st, resolved, false, nil
	}
	b, truncated, err := readPreviewBounded(f, maxBytes)
	if err != nil {
		return nil, st, resolved, false, err
	}
	if maxBytes > 0 && st.Size() > maxBytes {
		truncated = true
	}
	return b, st, resolved, truncated, nil
}

// ReadAllAccessPath revalidates an AccessPath and reads the entire file.
func (s *Sandbox) ReadAllAccessPath(path AccessPath) ([]byte, os.FileInfo, string, error) {
	return s.readAllPath(path.path)
}

// resolveCommandWorkingDirAccessPath resolves the working directory for command execution.
// Falls back to RootDir if WorkingDir is not set.
func (s *Sandbox) resolveCommandWorkingDirAccessPath() (AccessPath, string, error) {
	workdir := strings.TrimSpace(s.WorkingDir)
	if workdir == "" {
		workdir = strings.TrimSpace(s.RootDir)
	}
	accessPath, err := s.ResolveAccessPath(workdir)
	if err != nil {
		return AccessPath{}, "", err
	}
	return accessPath, strings.TrimSpace(accessPath.Abs()), nil
}
