package sandbox

import "testing"

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
