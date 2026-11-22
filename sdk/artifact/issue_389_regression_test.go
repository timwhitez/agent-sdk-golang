package artifact

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestCompleteEnvelopeFrameConsumesAllPayload(t *testing.T) {
	codec := JSONEnvelopeCodec{}
	preview := "Example " + EnvelopeStartMarker + " and " + EnvelopeEndMarker + " in quoted source."
	manifest := validTestManifest([]byte(preview))
	text, normalized, err := codec.Encode(Envelope{Manifest: manifest, Preview: preview}, Budget{MaxBytes: 8000, MaxTokens: 8000})
	if err != nil {
		t.Fatal(err)
	}
	var raw bytes.Buffer
	encoder := json.NewEncoder(&raw)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(normalized); err != nil {
		t.Fatal(err)
	}
	literalMarkers := EnvelopeStartMarker + "\n" + raw.String() + EnvelopeEndMarker
	for _, tc := range []struct {
		name    string
		text    string
		wantErr bool
	}{
		{"single frame", text, false},
		{"outer whitespace", " \n\t" + text + "\n\t ", false},
		{"literal markers in JSON string", literalMarkers, false},
		{"two frames", text + "\n" + text, true},
		{"extra end marker", text + "\n" + EnvelopeEndMarker, true},
		{"tail before last end marker", text + "\nimportant tail\n" + EnvelopeEndMarker, true},
		{"garbage in payload", strings.TrimSuffix(text, EnvelopeEndMarker) + "garbage\n" + EnvelopeEndMarker, true},
		{"second JSON value", strings.TrimSuffix(text, EnvelopeEndMarker) + "{}\n" + EnvelopeEndMarker, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !IsFramedEnvelope(tc.text) {
				t.Fatal("fixture is not a complete frame")
			}
			decoded, found, err := codec.Decode(tc.text)
			if !found || (err != nil) != tc.wantErr {
				t.Fatalf("found=%v err=%v; want found=true and error=%v", found, err, tc.wantErr)
			}
			if !tc.wantErr && decoded.Preview != preview {
				t.Fatalf("valid preview changed: %q", decoded.Preview)
			}
		})
	}
	// Decode historically locates an embedded envelope. SDK boundaries use
	// IsFramedEnvelope before Decode; tightening complete frames does not change
	// the direct decoder's compatibility with ordinary surrounding prose.
	decoded, found, err := codec.Decode("leading prose\n" + text + "\ntrailing prose")
	if err != nil || !found || decoded.Preview != preview {
		t.Fatalf("legacy embedded decode changed: found=%v err=%v", found, err)
	}
}
