package artifact

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestEnvelopeDecodeRejectsAlteredVisibleContent(t *testing.T) {
	const preview = "original preview"
	manifest := validTestManifest([]byte(preview))
	manifest.Preview = Preview{Kind: PreviewKindFull}
	text, _, err := (JSONEnvelopeCodec{}).Encode(Envelope{Manifest: manifest, Preview: preview}, Budget{MaxBytes: 4096, MaxTokens: 4096})
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal([]byte(strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(text, EnvelopeStartMarker), EnvelopeEndMarker))), &payload); err != nil {
		t.Fatal(err)
	}
	payload["preview"] = json.RawMessage(`"modified preview"`)
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	tampered := EnvelopeStartMarker + "\n" + string(raw) + "\n" + EnvelopeEndMarker
	decoded, found, err := (JSONEnvelopeCodec{}).Decode(tampered)
	if !found {
		t.Fatal("framed envelope not recognized")
	}
	if err == nil {
		t.Fatalf("modified preview accepted: visible hash=%s actual=%s", decoded.Manifest.VisibleMeasurement.SHA256, DigestSHA256([]byte(decoded.Preview)))
	}
}

func TestEnvelopeEncodeRejectsInvalidUTF8Preview(t *testing.T) {
	preview := "raw-" + string([]byte{0xff}) + "-output"
	manifest := validTestManifest([]byte(preview))
	manifest.Preview = Preview{Kind: PreviewKindFull}
	_, _, err := (JSONEnvelopeCodec{}).Encode(Envelope{Manifest: manifest, Preview: preview}, Budget{MaxBytes: 4096, MaxTokens: 4096})
	if err == nil || !strings.Contains(err.Error(), "valid UTF-8") {
		t.Fatalf("invalid UTF-8 preview must be rejected before JSON conversion: %v", err)
	}
}

func TestEnvelopeDecodeChecksProvidedVisibleMeasurements(t *testing.T) {
	const preview = "界🙂visible\ntext"
	for _, tc := range []struct {
		name    string
		visible Measurement
		wantErr string
	}{
		{"unknown measurements", Measurement{}, ""},
		{"bytes only", Measurement{Bytes: int64Pointer(int64(len(preview)))}, ""},
		{"hash only", Measurement{SHA256: DigestSHA256([]byte(preview))}, ""},
		{"whitespace around hash", Measurement{SHA256: " " + DigestSHA256([]byte(preview)) + " "}, ""},
		{"incorrect bytes", Measurement{Bytes: int64Pointer(int64(len(preview)) + 1)}, "byte count"},
		{"incorrect hash", Measurement{SHA256: DigestSHA256([]byte("different text"))}, "sha256"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			manifest := validTestManifest([]byte(preview))
			manifest.VisibleMeasurement = tc.visible
			payload, err := json.Marshal(Envelope{Manifest: manifest, Preview: preview})
			if err != nil {
				t.Fatal(err)
			}
			decoded, found, err := (JSONEnvelopeCodec{}).Decode(EnvelopeStartMarker + "\n" + string(payload) + "\n" + EnvelopeEndMarker)
			if !found {
				t.Fatal("framed envelope was not recognized")
			}
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("Decode error=%v, want %s", err, tc.wantErr)
				}
				return
			}
			if err != nil || decoded.Preview != preview {
				t.Fatalf("valid optional measurements rejected: %v", err)
			}
		})
	}
}

func TestEnvelopeEncodeRecomputesVisibleMeasurementsForUnicode(t *testing.T) {
	const preview = "界🙂 \ufffd\nvisible"
	manifest := validTestManifest([]byte(preview))
	manifest.Preview = Preview{Kind: PreviewKindFull, Ranges: []Range{{Unit: RangeUnitBytes, Start: 0, End: int64(len(preview))}}}
	manifest.VisibleMeasurement = Measurement{Bytes: int64Pointer(100), SHA256: DigestSHA256([]byte("old view"))}
	text, normalized, err := (JSONEnvelopeCodec{}).Encode(Envelope{Manifest: manifest, Preview: preview}, Budget{MaxBytes: 4096, MaxTokens: 4096})
	if err != nil {
		t.Fatal(err)
	}
	decoded, found, err := (JSONEnvelopeCodec{}).Decode(text)
	if err != nil || !found || decoded.Preview != preview {
		t.Fatalf("Unicode roundtrip failed: found=%v err=%v preview=%q", found, err, decoded.Preview)
	}
	for _, measurement := range []Measurement{normalized.Manifest.VisibleMeasurement, decoded.Manifest.VisibleMeasurement} {
		if measurement.Bytes == nil || *measurement.Bytes != int64(len(preview)) || measurement.SHA256 != DigestSHA256([]byte(preview)) {
			t.Fatalf("visible measurements were not recomputed: %+v", measurement)
		}
	}
}
