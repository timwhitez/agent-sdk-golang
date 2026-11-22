package artifact

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestRangePreviewCoverageShrinksWithVisibleBytes(t *testing.T) {
	content := strings.Repeat("abcdefghij", 1000)
	manifest := validTestManifest([]byte(content))
	const start, end = int64(500), int64(7500)
	manifest.Preview = Preview{Kind: PreviewKindRange, Ranges: []Range{{Unit: RangeUnitBytes, Start: start, End: end}}, Truncated: true}
	envelope := Envelope{Manifest: manifest, Preview: content[start:end]}
	encoded, out, err := (JSONEnvelopeCodec{}).Encode(envelope, Budget{MaxBytes: 2000, MaxTokens: 10000})
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) > 2000 || len(out.Preview) >= int(end-start) {
		t.Fatal("fixture did not exercise range truncation")
	}
	if len(out.Manifest.Preview.Ranges) != 1 {
		t.Fatalf("range projection lost: %+v", out.Manifest.Preview)
	}
	r := out.Manifest.Preview.Ranges[0]
	if r.Start != start || r.End != start+int64(len(out.Preview)) {
		t.Fatalf("visible %d bytes from offset %d but returned range=[%d,%d); omitted bytes are still claimed as visible", len(out.Preview), start, r.Start, r.End)
	}
	if envelope.Manifest.Preview.Ranges[0].End != end {
		t.Fatal("encoding mutated the caller's range")
	}
}

func TestRangePreviewContinuationTracksMappedSourceOffset(t *testing.T) {
	content := strings.Repeat("界🙂", 1000)
	const start = int64(500)
	end := start + int64(len(content))
	for _, next := range []int64{0, end, end + 100} {
		manifest := validTestManifest([]byte(strings.Repeat("x", int(start)) + content))
		manifest.Preview = Preview{Kind: PreviewKindRange, Ranges: []Range{{Unit: RangeUnitBytes, Start: start, End: end}}, Truncated: true}
		envelope := Envelope{Manifest: manifest, Preview: content, Continuation: &Continuation{ObjectRef: manifest.ObjectRef, RangeUnit: RangeUnitBytes, Next: next}}
		_, out, err := (JSONEnvelopeCodec{}).Encode(envelope, Budget{MaxBytes: 2000, MaxTokens: 10000})
		if err != nil {
			t.Fatal(err)
		}
		want := start + int64(len(out.Preview))
		if next == end+100 {
			want = next // A host cursor outside this view has its own meaning.
		}
		if out.Continuation.Next != want || envelope.Continuation.Next != next {
			t.Fatalf("next=%d: returned cursor=%d want=%d; original=%d", next, out.Continuation.Next, want, envelope.Continuation.Next)
		}
	}
}

func TestStructuredPreviewCoveragePreservedWhenItFits(t *testing.T) {
	content := strings.Repeat("source content\n", 500)
	for _, preview := range []Preview{
		{Kind: PreviewKindRange, Ranges: []Range{{Unit: RangeUnitBytes, Start: 500, End: 500 + int64(len(content))}}, Truncated: true},
		{Kind: PreviewKindRange, Ranges: []Range{{Unit: RangeUnitLines, Start: 5, End: 505}}, Truncated: true},
		{Kind: PreviewKindHeadTail, Ranges: []Range{{Unit: RangeUnitBytes, Start: 0, End: 10}, {Unit: RangeUnitBytes, Start: 10000, End: 10010}}, Truncated: true},
		{Kind: PreviewKindNested, Ranges: []Range{{Unit: RangeUnitLines, Start: 50, End: 60}}, Truncated: true},
	} {
		t.Run(string(preview.Kind)+string(preview.Ranges[0].Unit), func(t *testing.T) {
			manifest := validTestManifest([]byte(content))
			manifest.Preview = preview
			_, out, err := (JSONEnvelopeCodec{}).Encode(Envelope{Manifest: manifest, Preview: content}, Budget{MaxBytes: 32000, MaxTokens: 32000})
			if err != nil || out.Preview != content || !reflect.DeepEqual(out.Manifest.Preview, preview) {
				t.Fatalf("fitting structured preview changed: %+v err=%v", out.Manifest.Preview, err)
			}
		})
	}
}

func TestStructuredPreviewRejectsUnmappableTruncation(t *testing.T) {
	content := strings.Repeat("source content\n", 500)
	for _, preview := range []Preview{
		{Kind: PreviewKindRange, Ranges: []Range{{Unit: RangeUnitLines, Start: 5, End: 505}}, Truncated: true},
		{Kind: PreviewKindRange, Ranges: []Range{{Unit: RangeUnitBytes, Start: 500, End: 600}}, Truncated: true},
		{Kind: PreviewKindHeadTail, Ranges: []Range{{Unit: RangeUnitBytes, Start: 0, End: 10}, {Unit: RangeUnitBytes, Start: 10000, End: 10010}}, Truncated: true},
		{Kind: PreviewKindNested, Ranges: []Range{{Unit: RangeUnitLines, Start: 50, End: 60}}, Truncated: true},
	} {
		t.Run(string(preview.Kind)+string(preview.Ranges[0].Unit), func(t *testing.T) {
			manifest := validTestManifest([]byte(content))
			manifest.Preview = preview
			_, _, err := (JSONEnvelopeCodec{}).Encode(Envelope{Manifest: manifest, Preview: content}, Budget{MaxBytes: 2000, MaxTokens: 10000})
			if !errors.Is(err, ErrEnvelopeBudgetTooSmall) {
				t.Fatalf("shortened a view without knowing its source mapping: err=%v", err)
			}
		})
	}
}
