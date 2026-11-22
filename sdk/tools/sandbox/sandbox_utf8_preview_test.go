package sandbox

import (
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestReadLineLimitCountsUnicodeCharacters(t *testing.T) {
	for _, count := range []int{700, 2000, 2001} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			line := strings.Repeat("界", count)
			got, err := readUnicodeFixture(t, []byte(line+"\n"), map[string]any{})
			want := "   1  " + line
			if count > maxReadLineChars {
				want = "   1  " + strings.Repeat("界", maxReadLineChars-3) + "...\n... (lines truncated to 2000 chars)"
			}
			if err != nil || got != want || !utf8.ValidString(got) {
				t.Fatalf("count=%d valid=%t runes=%d err=%v", count, utf8.ValidString(got), utf8.RuneCountInString(got), err)
			}
		})
	}
}

func TestGrepPreviewPreservesUTF8Boundary(t *testing.T) {
	for _, line := range []string{strings.Repeat("界", 700), strings.Repeat("😀", 201), strings.Repeat("a界", 200)} {
		got := formatGrepLine("file.txt", 1, line, true)
		if !utf8.ValidString(got) {
			t.Fatalf("invalid UTF-8 preview: %q", got)
		}
		if want := "file.txt:1: " + string([]rune(line)[:197]) + "..."; got != want {
			t.Fatalf("preview=%q want=%q", got, want)
		}
	}
}
