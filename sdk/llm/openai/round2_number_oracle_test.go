package openai

import (
	"math/big"
	"testing"
)

// Use exact rational arithmetic as an independent oracle for the decimal
// normalization used by opaque snapshot comparison.
func TestRound2OpaqueNumberEqualityMatchesExactRationals(t *testing.T) {
	values := []string{
		"0", "-0.000e99", "0e-400", "1", "1.0", "10e-1", "0.01e2",
		"-1", "-1000e-3", "0.0001", "1e-4", "100.001", "100001e-3",
		"9007199254740992", "9007199254740993", "9007199254740993000e-3",
		"-9007199254740993", "-9007199254740992", "1e400", "100e398", "1e-400", "10e-401",
		"1.23456789012345678901234567890123456789", "123456789012345678901234567890123456789e-38",
		"1.23456789012345678901234567890123456788",
	}
	for _, left := range values {
		l, ok := new(big.Rat).SetString(left)
		if !ok {
			t.Fatalf("invalid oracle value %q", left)
		}
		for _, right := range values {
			r, ok := new(big.Rat).SetString(right)
			if !ok {
				t.Fatalf("invalid oracle value %q", right)
			}
			want := l.Cmp(r) == 0
			if got := responsesJSONEqual([]byte(`{"nested":[`+left+`]}`), []byte(`{"nested":[`+right+`]}`)); got != want {
				t.Fatalf("%s vs %s: equality=%t, rational oracle=%t", left, right, got, want)
			}
		}
	}
}
