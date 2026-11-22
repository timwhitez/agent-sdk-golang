package artifact

import (
	"strings"
	"testing"
)

func TestD3ReencodedPrefixContinuationMatchesNewCoverage(t *testing.T) {
	content := strings.Repeat("0123456789", 700)
	manifest := validTestManifest([]byte(content))
	manifest.Preview = Preview{Kind: PreviewKindFull}
	envelope := Envelope{Manifest: manifest, Preview: content, Continuation: &Continuation{ObjectRef: manifest.ObjectRef, RangeUnit: RangeUnitBytes}}
	codec := JSONEnvelopeCodec{}
	text, _, err := codec.Encode(envelope, Budget{MaxBytes: 7000, MaxTokens: 10000})
	if err != nil {
		t.Fatal(err)
	}
	first, found, err := codec.Decode(text)
	if !found || err != nil || first.Continuation.Next != int64(len(first.Preview)) {
		t.Fatalf("first boundary: %v, %v", found, err)
	}
	text, _, err = codec.Encode(first, Budget{MaxBytes: 2000, MaxTokens: 10000})
	if err != nil {
		t.Fatal(err)
	}
	second, found, err := codec.Decode(text)
	if !found || err != nil || len(second.Preview) >= len(first.Preview) {
		t.Fatalf("second boundary: %v, %v", found, err)
	}
	if next, end := second.Continuation.Next, second.Manifest.Preview.Ranges[0].End; next != end {
		t.Fatalf("new displayed byte prefix=[0,%d), continuation starts at %d; following it skips %d source bytes (%d before second encoding)", end, next, next-end, len(first.Preview))
	}
}

func TestReencodingPreservesHostContinuations(t *testing.T) {
	content := strings.Repeat("0123456789", 700)
	for _, tc := range []struct {
		name string
		unit RangeUnit
		next int64
	}{
		{"byte cursor outside preview", RangeUnitBytes, 6500},
		{"line cursor", RangeUnitLines, 17},
	} {
		t.Run(tc.name, func(t *testing.T) {
			manifest := validTestManifest([]byte(content))
			manifest.Preview = Preview{Kind: PreviewKindPrefix, Ranges: []Range{{Unit: RangeUnitBytes, Start: 0, End: 6000}}, Truncated: true}
			envelope := Envelope{Manifest: manifest, Preview: content[:6000], Continuation: &Continuation{ObjectRef: manifest.ObjectRef, RangeUnit: tc.unit, Next: tc.next}}
			_, out, err := (JSONEnvelopeCodec{}).Encode(envelope, Budget{MaxBytes: 2000, MaxTokens: 10000})
			if err != nil || out.Continuation == nil || out.Continuation.Next != tc.next || envelope.Continuation.Next != tc.next {
				t.Fatalf("host cursor changed: %+v err=%v", out.Continuation, err)
			}
		})
	}
}
