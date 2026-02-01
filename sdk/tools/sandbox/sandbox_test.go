package sandbox

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSandboxResolveBlocksEscape(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	_, err = s.Resolve("../etc/passwd")
	if err == nil {
		t.Fatalf("expected escape error")
	}
}

func TestSandboxResolveAllowsInside(t *testing.T) {
	root := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	p, err := s.Resolve("a/b/../c.txt")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if p == "" {
		t.Fatalf("expected non-empty path")
	}
}

func TestSandboxResolveAllowsExternalAfterAllow(t *testing.T) {
	root := t.TempDir()
	ext := t.TempDir()
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	target := filepath.Join(ext, "file.txt")
	if _, err := s.Resolve(target); err == nil {
		t.Fatalf("expected escape error before allow")
	}
	if _, added, err := s.AllowExternalDirectory(ext); err != nil {
		t.Fatalf("allow external: %v", err)
	} else if !added {
		t.Fatalf("expected external directory to be added")
	}
	if _, err := s.Resolve(target); err != nil {
		t.Fatalf("expected resolve to succeed after allow, got %v", err)
	}
}

func TestSandboxAllowExternalDirectoryResolvesSymlinks(t *testing.T) {
	root := t.TempDir()
	ext := t.TempDir()
	link := filepath.Join(t.TempDir(), "ext-link")
	if err := os.Symlink(ext, link); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	got, _, err := s.AllowExternalDirectory(link)
	if err != nil {
		t.Fatalf("allow external: %v", err)
	}
	if filepath.Clean(got) != filepath.Clean(ext) {
		t.Fatalf("expected resolved path %q, got %q", ext, got)
	}
}
