package sandbox

import (
	"context"
	"errors"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"os"
	"path/filepath"
	"testing"
)

type cancelFileConfirmer struct{ cancel context.CancelFunc }

func (c cancelFileConfirmer) Confirm(context.Context, string, string) (bool, error) {
	c.cancel()
	return true, nil
}

func TestFileTools_CancelDuringConfirmation(t *testing.T) {
	for _, tc := range []struct{ name, args string }{
		{"write", `{"file_path":"a.txt","content":"after\n"}`},
		{"edit", `{"file_path":"a.txt","old_string":"before","new_string":"after"}`},
		{"multiedit", `{"file_path":"a.txt","edits":[{"old_string":"before","new_string":"after"}]}`},
		{"apply_patch", `{"patch":"*** Begin Patch\n*** Update File: a.txt\n@@\n-before\n+after\n*** End Patch"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, "a.txt")
			if err := os.WriteFile(path, []byte("before\n"), 0600); err != nil {
				t.Fatal(err)
			}
			s, err := New(root)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			deps := tools.NewContainer()
			tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
			tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return cancelFileConfirmer{cancel}, nil })
			for _, tool := range Tools() {
				if tool.Name == tc.name {
					_, err = tool.Execute(ctx, tc.args, deps)
					break
				}
			}
			if !errors.Is(err, context.Canceled) {
				t.Errorf("error = %v", err)
			}
			got, err := os.ReadFile(path)
			if err != nil || string(got) != "before\n" {
				t.Fatalf("file=%q error=%v", got, err)
			}
		})
	}
}

func TestPatch_EmptyRangesAndEOF(t *testing.T) {
	for _, tc := range []struct {
		name, raw, hunks, want string
		fail                   bool
	}{
		{"start", "one\ntwo\n", "@@ -0,0 +1,1 @@\n+inserted", "inserted\none\ntwo\n", false},
		{"middle", "one\ntwo\n", "@@ -1,0 +2,1 @@\n+inserted", "one\ninserted\ntwo\n", false},
		{"end", "one\ntwo\n", "@@ -2,0 +3,1 @@\n+inserted", "one\ntwo\ninserted\n", false},
		{"multiple", "one\ntwo\n", "@@ -1,0 +2,1 @@\n+inserted\n@@ -2,0 +4,1 @@\n+last", "one\ninserted\ntwo\nlast\n", false},
		{"eof", "before\n", "@@\n-before\n+after\n*** End of File", "after\n", false},
		{"eof repeated context", "before\nbefore\n", "@@\n-before\n+after\n*** End of File", "before\nafter\n", false},
		{"eof mismatched", "before\nlast\n", "@@\n-before\n+after\n*** End of File", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ops, err := parsePatchOpsForPreview("*** Begin Patch\n*** Update File: a.txt\n" + tc.hunks + "\n*** End Patch")
			if err != nil {
				t.Fatal(err)
			}
			got, err := applyHunksToContent("a.txt", tc.raw, ops[0].hunks)
			if tc.fail {
				if err == nil {
					t.Fatalf("expected rejection, got %q", got)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("got %q, %v; want %q", got, err, tc.want)
			}
		})
	}
}

func TestSandbox_SymlinkRoot(t *testing.T) {
	base := t.TempDir()
	real := filepath.Join(base, "real")
	alias := filepath.Join(base, "alias")
	if err := os.MkdirAll(filepath.Join(real, "child"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(real, alias); err != nil {
		t.Skip(err)
	}
	if err := os.WriteFile(filepath.Join(real, "hello.txt"), []byte("hello\n"), 0600); err != nil {
		t.Fatal(err)
	}
	sb, err := New(alias)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return sb, nil })
	if _, err := readTool().Execute(context.Background(), `{"file_path":"hello.txt"}`, deps); err != nil {
		t.Fatal(err)
	}
	for _, root := range []string{alias, filepath.Join(alias, "child"), filepath.Join(alias, "missing")} {
		s, err := New(root)
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range []string{"file", filepath.Join(root, "file")} {
			if _, err := s.Resolve(path); err != nil {
				t.Errorf("root %s path %s: %v", root, path, err)
			}
		}
	}
	s, err := New(alias)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(base, filepath.Join(real, "escape")); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Resolve("escape/secret"); err == nil {
		t.Fatal("escape accepted")
	}
	token, err := s.ResolveAccessPath(filepath.Join(alias, "file"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(alias); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(base, alias); err != nil {
		t.Fatal(err)
	}
	if _, err := s.RevalidateAccessPath(token); err == nil {
		t.Fatal("changed root accepted")
	}
}

func TestPatch_EOFMultipleFiles(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"a.txt", "b.txt"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte("before\n"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	s, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	patch := "*** Begin Patch\n*** Update File: a.txt\n@@\n-before\n+after\n*** End of File\n*** Update File: b.txt\n@@\n-before\n+after\n*** End of File\n*** End Patch"
	if _, err := applyPatchToSandbox(s, patch); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"a.txt", "b.txt"} {
		got, err := os.ReadFile(filepath.Join(root, name))
		if err != nil || string(got) != "after\n" {
			t.Fatalf("%s: %q %v", name, got, err)
		}
	}
}
