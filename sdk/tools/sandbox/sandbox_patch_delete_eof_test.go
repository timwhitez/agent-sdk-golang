package sandbox

import "testing"

func TestApplyPatchDeletingUnterminatedLastLinePreservesPrecedingEnds(t *testing.T) {
	for _, end := range []string{"\n", "\r\n", "\r"} {
		for _, trailing := range []string{"", end} {
			for _, blank := range []string{"", end, end + end} {
				raw := "kept" + end + blank + "discard" + trailing
				want := "kept" + end + blank
				got, err := applyTestPatch(t, raw, "@@\n-discard")
				if err != nil || got != want {
					t.Fatalf("input=%q got=%q err=%v want=%q", raw, got, err, want)
				}
			}
		}
	}
}
