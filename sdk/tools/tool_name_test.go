package tools

import "testing"

func TestNormalizeToolNameStripsMultiplePrefixes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "dot prefixes", in: "tools.function.read", want: "read"},
		{name: "colon prefixes", in: "functions:tool:webfetch", want: "webfetch"},
		{name: "mixed case and spaces", in: "  TOOLS.Function.Read  ", want: "read"},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := NormalizeToolName(tt.in); got != tt.want {
				t.Fatalf("NormalizeToolName(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestNormalizeToolNameDoesNotStripNonPrefixWords(t *testing.T) {
	t.Parallel()

	got := NormalizeToolName("functional.read")
	if got != "functionalread" {
		t.Fatalf("NormalizeToolName stripped non-prefix token: got %q", got)
	}
}
