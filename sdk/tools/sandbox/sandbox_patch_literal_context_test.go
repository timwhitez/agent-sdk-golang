package sandbox

import "testing"

func TestApplyPatchLiteralControlTextInContext(t *testing.T) {
	for _, literal := range []string{"@@ Section", "@@ -1,1 +1,1 @@", "*** End of File", "*** End Patch", "*** Update File: unrelated.txt"} {
		t.Run(literal, func(t *testing.T) {
			for _, header := range []string{"@@", "@@ scope"} {
				raw := "scope\n" + literal + "\nold\n"
				want := "scope\n" + literal + "\nnew\n"
				got, err := applyTestPatch(t, raw, header+"\n "+literal+"\n-old\n+new")
				if err != nil || got != want {
					t.Fatalf("header=%q got=%q err=%v want=%q", header, got, err, want)
				}
			}
		})
	}
}
