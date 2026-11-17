package sandbox

import "testing"

func applyTestPatch(t *testing.T, raw, hunks string) (string, error) {
	t.Helper()
	ops, err := parsePatchOpsForPreview("*** Begin Patch\n*** Update File: a.txt\n" + hunks + "\n*** End Patch")
	if err != nil {
		return "", err
	}
	return applyHunksToContent("a.txt", raw, ops[0].hunks)
}

func TestApplyPatch_ScopeHeaders(t *testing.T) {
	dup := "class A:\n    def run(self):\n        x = 1\nclass B:\n    def run(self):\n        x = 1\n    def other(self):\n        x = 1\n"
	for _, tc := range []struct {
		name, raw, hunks, want string
		fail                   bool
	}{
		{"second scope", dup, "@@ class B:\n-        x = 1\n+        x = 2", "class A:\n    def run(self):\n        x = 1\nclass B:\n    def run(self):\n        x = 2\n    def other(self):\n        x = 1\n", false},
		{"nested scopes", dup, "@@ class B:\n@@     def other(self):\n-        x = 1\n+        x = 3", "class A:\n    def run(self):\n        x = 1\nclass B:\n    def run(self):\n        x = 1\n    def other(self):\n        x = 3\n", false},
		{"trimmed scope", dup, "@@ def other(self):\n-        x = 1\n+        x = 3", "class A:\n    def run(self):\n        x = 1\nclass B:\n    def run(self):\n        x = 1\n    def other(self):\n        x = 3\n", false},
		{"scope missing", dup, "@@ class C:\n-        x = 1\n+        x = 2", "", true},
		{"substring is not a scope", "# class B: note\nx\n", "@@ class B:\n-x\n+y", "", true},
		{"scope after previous hunk", "a\nfoo\nb\nfoo\n", "@@ -1,1 +1,1 @@\n-a\n+A\n@@ foo\n-b\n+B", "A\nfoo\nB\nfoo\n", false},
		{"numbered header with trailer", "a\nb\n", "@@ -2,1 +2,1 @@ trailer\n-b\n+B", "a\nB\n", false},
		{"bare header", "a\nb\n", "@@\n-b\n+B", "a\nB\n", false},
		{"unclosed range stays invalid", "a\n", "@@ -1,1 +1,1\n-a\n+b", "", true},
		{"scope without body", "a\n", "@@ a", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := applyTestPatch(t, tc.raw, tc.hunks)
			if tc.fail {
				if err == nil {
					t.Fatalf("expected error, got %q", got)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("got %q, %v; want %q", got, err, tc.want)
			}
		})
	}
}

func TestApplyPatch_PreservesLineEndings(t *testing.T) {
	for _, tc := range []struct {
		name, raw, hunks, want string
	}{
		{"mixed keeps untouched", "a\r\nb\nc\r\n", "@@\n-b\n+B", "a\r\nB\nc\r\n"},
		{"bare cr", "a\rb\rc\r", "@@\n-b\n+B", "a\rB\rc\r"},
		{"crlf", "a\r\nb\r\n", "@@\n-a\n+A\n+extra", "A\r\nextra\r\nb\r\n"},
		{"lf", "a\nb\n", "@@\n a\n+x\n b", "a\nx\nb\n"},
		{"no trailing newline", "a\r\nb", "@@\n-b\n+B", "a\r\nB"},
		{"append after unterminated last", "a\nb", "@@\n b\n+c", "a\nb\nc"},
		{"delete last", "a\r\nb\r\n", "@@\n a\n-b", "a\r\n"},
		{"single empty line", "\n", "@@ -1,1 +1,1 @@\n-\n+x", "x\n"},
		{"single empty crlf line", "\r\n", "@@ -1,1 +1,1 @@\n-\n+x", "x\r\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := applyTestPatch(t, tc.raw, tc.hunks)
			if err != nil || got != tc.want {
				t.Fatalf("got %q, %v; want %q", got, err, tc.want)
			}
		})
	}
}
