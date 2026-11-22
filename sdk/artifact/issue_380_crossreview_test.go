package artifact

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestR2MappedRangeStaysExactAcrossBudgetsAndReencoding(t *testing.T) {
	const start = int64(777)
	content := strings.Repeat("界🙂<tag>\n\"quoted\"\\", 300)
	manifest := validTestManifest([]byte(strings.Repeat("x", int(start)) + content))
	manifest.Preview = Preview{Kind: PreviewKindRange, Ranges: []Range{{Unit: RangeUnitBytes, Start: start, End: start + int64(len(content))}}, Truncated: true}
	original := Envelope{Manifest: manifest, Preview: content, Continuation: &Continuation{ObjectRef: manifest.ObjectRef, RangeUnit: RangeUnitBytes, Next: start + int64(len(content))}}
	inputSnapshot := original.Clone()
	codec := JSONEnvelopeCodec{}
	estimate := func(text string) int { return utf8.RuneCountInString(text) }
	for _, byteBudget := range []int{1, 900, 1200, 1400, 1800, 2400, 3600, 8000} {
		for _, tokenBudget := range []int{600, 1600, 8000} {
			t.Run(fmt.Sprintf("bytes_%d_tokens_%d", byteBudget, tokenBudget), func(t *testing.T) {
				budget := Budget{MaxBytes: byteBudget, MaxTokens: tokenBudget, EstimateTokens: estimate}
				text, out, err := codec.Encode(original, budget)
				if errors.Is(err, ErrEnvelopeBudgetTooSmall) {
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if len(text) > byteBudget || estimate(text) > tokenBudget || !utf8.ValidString(text) || !strings.HasPrefix(content, out.Preview) {
					t.Fatalf("budget or Unicode/source prefix violated: bytes=%d tokens=%d preview=%q", len(text), estimate(text), out.Preview)
				}
				if out.Continuation.Next != start+int64(len(out.Preview)) {
					t.Fatalf("cursor=%d preview bytes=%d", out.Continuation.Next, len(out.Preview))
				}
				if len(out.Preview) > 0 {
					if got := out.Manifest.Preview.Ranges; len(got) != 1 || got[0].Start != start || got[0].End != out.Continuation.Next {
						t.Fatalf("range does not map exact visible bytes: %+v", got)
					}
				} else if len(out.Manifest.Preview.Ranges) != 0 {
					t.Fatal("empty preview claimed source coverage")
				}
				decoded, found, err := codec.Decode(text)
				if err != nil || !found || !reflect.DeepEqual(decoded, out) {
					t.Fatalf("codec self-roundtrip differs: found=%v err=%v", found, err)
				}
				secondText, second, err := codec.Encode(decoded, budget)
				if err != nil || secondText != text || !reflect.DeepEqual(second, out) {
					t.Fatalf("same-budget reencoding changed source view: err=%v", err)
				}
			})
		}
	}
	if !reflect.DeepEqual(original, inputSnapshot) {
		t.Fatal("encoding mutated original range/measurements/cursor")
	}
}
