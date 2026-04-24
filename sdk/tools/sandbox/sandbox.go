package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type SecurityError struct{ Message string }

func (e *SecurityError) Error() string { return e.Message }

type TodoItem struct {
	Content string `json:"content"`
	Status  string `json:"status"` // pending|in_progress|completed
}

type Sandbox struct {
	RootDir    string
	WorkingDir string

	mu    sync.Mutex
	Todos []TodoItem

	allowedRoots []string
}

type validatedSandboxPath struct {
	requested string
	abs       string
	resolved  string
}

// AccessPath captures a sandbox-validated path and its resolved target.
// It is an opaque token that callers can later revalidate before I/O.
type AccessPath struct {
	path validatedSandboxPath
}

// Abs returns the cleaned absolute path before symlink resolution.
func (p AccessPath) Abs() string { return p.path.abs }

var beforeSandboxPathRevalidate = func(string) {}

func (s *Sandbox) TodosSnapshot() []TodoItem {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]TodoItem, len(s.Todos))
	copy(out, s.Todos)
	return out
}

func (s *Sandbox) ReplaceTodos(todos []TodoItem) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.Todos = append([]TodoItem(nil), todos...)
	s.mu.Unlock()
}

// AllowedRootsSnapshot returns the currently allowed external roots.
func (s *Sandbox) AllowedRootsSnapshot() []string {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, len(s.allowedRoots))
	copy(out, s.allowedRoots)
	return out
}

// ReplaceAllowedRoots replaces the external allowed roots with a cleaned list.
// The roots are assumed to already be normalized (absolute, canonical).
func (s *Sandbox) ReplaceAllowedRoots(roots []string) {
	if s == nil {
		return
	}
	cleaned := make([]string, 0, len(roots))
	seen := map[string]bool{}
	base := strings.TrimSpace(s.RootDir)
	if base == "" {
		base = strings.TrimSpace(s.WorkingDir)
	}
	for _, r := range roots {
		r = strings.TrimSpace(r)
		if r == "" {
			continue
		}
		if !filepath.IsAbs(r) {
			if base != "" {
				r = filepath.Join(base, r)
			}
		}
		r = filepath.Clean(r)
		if r == "" {
			continue
		}
		if seen[r] {
			continue
		}
		seen[r] = true
		cleaned = append(cleaned, r)
	}
	s.mu.Lock()
	s.allowedRoots = cleaned
	s.mu.Unlock()
}

func New(root string) (*Sandbox, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	abs = filepath.Clean(abs)
	return &Sandbox{RootDir: abs, WorkingDir: abs}, nil
}

func (s *Sandbox) Resolve(path string) (string, error) {
	p, err := s.resolveForAccess(path)
	if err != nil {
		return "", err
	}
	return p.abs, nil
}

// ResolveAccessPath validates a path for access and returns an access token
// that can be revalidated immediately before I/O.
func (s *Sandbox) ResolveAccessPath(path string) (AccessPath, error) {
	p, err := s.resolveForAccess(path)
	if err != nil {
		return AccessPath{}, err
	}
	return AccessPath{path: p}, nil
}

// RevalidateAccessPath re-checks that an access token still points to the same
// resolved target and returns the resolved absolute path for immediate use.
func (s *Sandbox) RevalidateAccessPath(path AccessPath) (string, error) {
	return s.revalidatePathForAccess(path.path)
}

func (s *Sandbox) resolveForAccess(path string) (validatedSandboxPath, error) {
	if path == "" {
		return validatedSandboxPath{}, &SecurityError{Message: "empty path"}
	}
	var abs string
	if filepath.IsAbs(path) {
		abs = filepath.Clean(path)
	} else {
		abs = filepath.Clean(filepath.Join(s.WorkingDir, path))
	}
	abs, err := filepath.Abs(abs)
	if err != nil {
		return validatedSandboxPath{}, err
	}
	if !s.isAllowedPath(abs) {
		return validatedSandboxPath{}, &SecurityError{Message: fmt.Sprintf("path escapes sandbox: %q -> %q", path, abs)}
	}
	resolved, err := evalSymlinksForPath(abs)
	if err != nil {
		return validatedSandboxPath{}, err
	}
	resolved = filepath.Clean(resolved)
	if !s.isAllowedPath(resolved) {
		return validatedSandboxPath{}, &SecurityError{Message: fmt.Sprintf("path escapes sandbox via symlink: %q -> %q", path, resolved)}
	}
	return validatedSandboxPath{
		requested: path,
		abs:       abs,
		resolved:  resolved,
	}, nil
}

func (s *Sandbox) revalidatePathForAccess(path validatedSandboxPath) (string, error) {
	if beforeSandboxPathRevalidate != nil {
		beforeSandboxPathRevalidate(path.abs)
	}
	resolved, err := evalSymlinksForPath(path.abs)
	if err != nil {
		return "", err
	}
	resolved = filepath.Clean(resolved)
	if !s.isAllowedPath(resolved) {
		return "", &SecurityError{Message: fmt.Sprintf("path escapes sandbox via symlink: %q -> %q", path.requested, resolved)}
	}
	if !pathsEqual(resolved, path.resolved) {
		return "", &SecurityError{Message: fmt.Sprintf("path changed after validation: %q (was %q, now %q)", path.requested, path.resolved, resolved)}
	}
	return resolved, nil
}

func (s *Sandbox) isAllowedPath(abs string) bool {
	if s == nil {
		return false
	}
	if isWithinRoot(abs, s.RootDir) {
		return true
	}
	s.mu.Lock()
	allowed := append([]string(nil), s.allowedRoots...)
	s.mu.Unlock()
	for _, root := range allowed {
		if isWithinRoot(abs, root) {
			return true
		}
	}
	return false
}

func (s *Sandbox) isAllowedExternalRoot(abs string) bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	allowed := append([]string(nil), s.allowedRoots...)
	s.mu.Unlock()
	for _, root := range allowed {
		if isWithinRoot(abs, root) {
			return true
		}
	}
	return false
}

func (s *Sandbox) normalizeExternalRoot(path string) (string, error) {
	if s == nil {
		return "", fmt.Errorf("nil sandbox")
	}
	p := strings.TrimSpace(path)
	if p == "" {
		return "", fmt.Errorf("empty path")
	}
	var abs string
	if filepath.IsAbs(p) {
		abs = filepath.Clean(p)
	} else {
		base := s.WorkingDir
		if strings.TrimSpace(base) == "" {
			base = s.RootDir
		}
		abs = filepath.Clean(filepath.Join(base, p))
	}
	abs, err := filepath.Abs(abs)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("path does not exist: %s", abs)
		}
		return "", err
	}
	resolved = filepath.Clean(resolved)
	st, err := os.Stat(resolved)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("path does not exist: %s", resolved)
		}
		return "", err
	}
	if st.IsDir() || st.Mode().IsRegular() {
		return resolved, nil
	}
	return "", fmt.Errorf("path is not a file or directory: %s", resolved)
}

// AllowExternalDirectory adds an external path (directory or file) to the sandbox allowlist.
// It returns the normalized path, whether it was newly added, and any error.
func (s *Sandbox) AllowExternalDirectory(path string) (string, bool, error) {
	normalized, err := s.normalizeExternalRoot(path)
	if err != nil {
		return "", false, err
	}
	if isWithinRoot(normalized, s.RootDir) {
		return normalized, false, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, root := range s.allowedRoots {
		if isWithinRoot(normalized, root) {
			return normalized, false, nil
		}
	}
	s.allowedRoots = append(s.allowedRoots, normalized)
	return normalized, true, nil
}

var Key = tools.Dep[*Sandbox]("sandbox")

// Confirmer is used by interactive clients to gate potentially dangerous actions.
// Tools should call Confirm() before executing writes/commands.
type Confirmer interface {
	Confirm(ctx context.Context, action string, detail string) (bool, error)
}

var ConfirmKey = tools.Dep[Confirmer]("confirm")

var ErrMissingConfirmer = errors.New("missing confirmer dependency")
var ErrToolDenied = errors.New("tool action denied")
var errNilConfirmer = errors.New("confirm dependency provider returned nil confirmer")

type missingConfirmer struct {
	err error
}

func (m missingConfirmer) Confirm(ctx context.Context, action string, detail string) (bool, error) {
	if m.err == nil {
		return false, ErrMissingConfirmer
	}
	return false, fmt.Errorf("%w: %v", ErrMissingConfirmer, m.err)
}

func getConfirmer(deps *tools.Container, ctx context.Context) Confirmer {
	c, err := tools.Get(deps, ctx, ConfirmKey)
	if err != nil {
		return missingConfirmer{err: err}
	}
	if c == nil {
		return missingConfirmer{err: errNilConfirmer}
	}
	return c
}

// Tools returns a Claude Code-style toolset bound to the sandbox dependency.
func Tools() []tools.Tool {
	return []tools.Tool{
		bashTool().WithEphemeralKeep(1),
		lsTool().WithEphemeralKeep(1),
		readTool().WithEphemeralKeep(1),
		webfetchTool().WithEphemeralKeep(1),
		writeTool().WithEphemeralKeep(1),
		editTool().WithEphemeralKeep(1),
		multieditTool().WithEphemeralKeep(1),
		applyPatchTool().WithEphemeralKeep(1),
		globTool().WithEphemeralKeep(1),
		externalDirectoryTool().WithEphemeralKeep(1),
		grepTool().WithEphemeralKeep(1),
		// Preferred (opencode-compatible) names.
		todoReadToolNamed("todoread"),
		todoWriteToolNamed("todowrite"),
		// Backward-compatible aliases.
		todoReadToolNamed("todo_read"),
		todoWriteToolNamed("todo_write"),
		doneTool(),
	}
}

// OpenFileNoFollow opens path while rejecting symlink leaf targets and
// verifying the opened file still matches the lstat result.

// OpenReadAccessPath revalidates an AccessPath and opens it for read using a
// no-follow leaf check.

var ()

// ReadAllAccessPath revalidates an AccessPath and reads the entire file.

const (
	maxEditFileBytes        int64 = 5 * 1024 * 1024
	maxReadFileBytes        int64 = 5 * 1024 * 1024
	maxWriteDiffBytes       int64 = 256 * 1024
	binaryDetectSampleBytes       = 8000
	maxReadOutputBytes            = 256 * 1024
	maxReadLineChars              = 2000
)

// Multimodal results can be added later using llm.Content blocks.
func ToolResultText(text string) llm.Content { return llm.TextContent(text) }
