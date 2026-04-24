package sandbox

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ============================================================================
// File I/O Helpers
// ============================================================================

// writeFileTempFactory is a var for testing temp file creation.
var writeFileTempFactory = func(dir, pattern string) (*os.File, error) {
	return os.CreateTemp(dir, pattern)
}

// writeFileBytes is a var for testing write operations.
var writeFileBytes = func(f *os.File, data []byte) (int, error) {
	return f.Write(data)
}

// writeFilePreserveMode writes data to path while preserving the existing file
// mode (or using defaultMode if the file doesn't exist). Uses atomic write
// with temp file and rename.
func writeFilePreserveMode(path string, data []byte, defaultMode os.FileMode) error {
	mode := defaultMode
	if info, err := os.Lstat(path); err == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			return &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", path)}
		}
		mode = info.Mode().Perm()
	} else if !os.IsNotExist(err) {
		return err
	}

	dir := filepath.Dir(path)
	if info, err := os.Lstat(dir); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return &SecurityError{Message: fmt.Sprintf("symlink target denied: %q", dir)}
	} else if err != nil && !os.IsNotExist(err) {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if dirFile, _, err := openFileNoFollow(dir); err != nil {
		return err
	} else {
		_ = dirFile.Close()
	}

	tmp, err := writeFileTempFactory(dir, ".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
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
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
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
