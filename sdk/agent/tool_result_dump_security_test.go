package agent

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDumpCleanupRejectsUnownedPaths(t *testing.T) {
	for _, name := range []string{"outside", "traversal", "wrong_name", "symlink"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.Chmod(dir, 0700); err != nil {
				t.Fatal(err)
			}
			victim := filepath.Join(t.TempDir(), "agent-tool-result-victim.txt")
			if name == "wrong_name" {
				victim = filepath.Join(dir, "important.txt")
			}
			if err := os.WriteFile(victim, []byte("keep"), 0600); err != nil {
				t.Fatal(err)
			}
			path := victim
			if name == "traversal" {
				path = filepath.Join(dir, "..") + string(os.PathSeparator) + filepath.Base(filepath.Dir(victim)) + string(os.PathSeparator) + filepath.Base(victim)
			}
			if name == "symlink" {
				path = filepath.Join(dir, "agent-tool-result-link.txt")
				if err := os.Symlink(victim, path); err != nil {
					t.Skip(err)
				}
			}
			now := time.Now()
			index := filepath.Join(dir, "agent-tool-result-index-test.json")
			if err := writeToolResultDumpIndex(index, "test", map[string]toolResultDumpLifecycleEntry{path: {CreatedAt: now.Add(-2 * time.Hour), ExpiresAt: now.Add(-time.Hour)}}, now); err != nil {
				t.Fatal(err)
			}
			if err := reclaimToolResultDumpOrphans(dir, now, time.Hour); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(victim); err != nil {
				t.Fatalf("unrelated file removed: %v", err)
			}
			if name == "symlink" {
				if _, err := os.Lstat(path); err != nil {
					t.Fatalf("unowned symlink removed: %v", err)
				}
			}
		})
	}
}

func TestResolveDumpDirRejectsUnsafeDirectory(t *testing.T) {
	old := toolResultDumpDir
	t.Cleanup(func() { toolResultDumpDir = old })
	for _, name := range []string{"permissions", "symlink"} {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "dumps")
			if name == "symlink" {
				if err := os.Symlink(t.TempDir(), dir); err != nil {
					t.Skip(err)
				}
			} else {
				if err := os.Mkdir(dir, 0777); err != nil {
					t.Fatal(err)
				}
				if err := os.Chmod(dir, 0777); err != nil {
					t.Fatal(err)
				}
			}
			toolResultDumpDir = func() string { return dir }
			if _, err := resolveToolResultDumpDir(); err == nil {
				t.Fatal("unsafe directory accepted")
			}
		})
	}
}

func TestDumpCleanupRejectsSymlinkIndex(t *testing.T) {
	dir := t.TempDir()
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(t.TempDir(), "index.json")
	if err := os.WriteFile(target, []byte(`{"version":1}`), 0600); err != nil {
		t.Fatal(err)
	}
	index := filepath.Join(dir, "agent-tool-result-index-link.json")
	if err := os.Symlink(target, index); err != nil {
		t.Skip(err)
	}
	if _, err := readToolResultDumpIndex(index); err == nil {
		t.Fatal("symlink index read")
	}
	if err := reclaimToolResultDumpOrphans(dir, time.Now(), time.Hour); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(index); err != nil {
		t.Fatal("symlink index removed", err)
	}
}

func TestResolveDumpDirAllowsCanonicalParent(t *testing.T) {
	parent := filepath.Join(t.TempDir(), "parent")
	if err := os.Symlink(t.TempDir(), parent); err != nil {
		t.Skip(err)
	}
	old := toolResultDumpDir
	toolResultDumpDir = func() string { return filepath.Join(parent, "dumps") }
	defer func() { toolResultDumpDir = old }()
	dir, err := resolveToolResultDumpDir()
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := filepath.EvalSymlinks(dir)
	if err != nil || canonical != dir {
		t.Fatalf("not canonical: %s %v", dir, err)
	}
}
