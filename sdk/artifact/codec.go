package artifact

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

const (
	EnvelopeStartMarker = "<tool_object_envelope_v1>"
	EnvelopeEndMarker   = "</tool_object_envelope_v1>"
)

var ErrEnvelopeBudgetTooSmall = errors.New("artifact envelope budget too small")

type Budget struct {
	MaxBytes       int
	MaxTokens      int
	EstimateTokens func(string) int
}

type JSONEnvelopeCodec struct{}

func (JSONEnvelopeCodec) Encode(envelope Envelope, budget Budget) (string, Envelope, error) {
	if budget.MaxBytes <= 0 || budget.MaxTokens <= 0 {
		return "", Envelope{}, fmt.Errorf("%w: byte and token budgets must be positive", ErrEnvelopeBudgetTooSmall)
	}
	if err := envelope.Validate(); err != nil {
		return "", Envelope{}, err
	}
	estimate := budget.EstimateTokens
	estimate = safeTokenEstimator(estimate)

	normalized := envelope.Clone()
	byteRange, hasByteRange := previewByteRange(normalized.Manifest.Preview, len(normalized.Preview))
	updateByteContinuation := false
	continuationStart := int64(0)
	if continuation := normalized.Continuation; continuation != nil && continuation.RangeUnit == RangeUnitBytes {
		updateByteContinuation = continuation.Next == 0
		// Re-encoding a prefix produced by this codec must move its cursor
		// back when the budget shrinks. An unrelated host cursor is left as-is.
		preview := normalized.Manifest.Preview
		if (preview.Kind == PreviewKindPrefix || preview.Kind == PreviewKindFull) && len(preview.Ranges) == 1 {
			r := preview.Ranges[0]
			if r.Unit == RangeUnitBytes && r.Start == 0 && r.End == int64(len(normalized.Preview)) && continuation.Next == r.End {
				updateByteContinuation = true
			}
		}
		if hasByteRange {
			continuationStart = byteRange.Start
			if continuation.Next == byteRange.End {
				updateByteContinuation = true
			}
		}
	}
	render := func(preview string) (string, Envelope, error) {
		candidate := normalized.Clone()
		candidate.Preview = preview
		candidate.Manifest.Preview = normalizePreview(candidate.Manifest.Preview, len(preview), len(preview) < len(normalized.Preview))
		if hasByteRange && len(preview) > 0 {
			candidate.Manifest.Preview.Ranges[0].End = byteRange.Start + int64(len(preview))
		}
		candidate.Manifest.VisibleMeasurement = visibleMeasurement(preview, estimate, !candidate.Manifest.Preview.Truncated)
		if updateByteContinuation {
			candidate.Continuation.Next = continuationStart + int64(len(preview))
		}
		if err := candidate.Validate(); err != nil {
			return "", Envelope{}, err
		}
		payload, err := json.Marshal(candidate)
		if err != nil {
			return "", Envelope{}, fmt.Errorf("artifact envelope encode: %w", err)
		}
		encoded := EnvelopeStartMarker + "\n" + string(payload) + "\n" + EnvelopeEndMarker
		return encoded, candidate, nil
	}

	full, fullEnvelope, err := render(envelope.Preview)
	if err != nil {
		return "", Envelope{}, err
	}
	if fitsEnvelopeBudget(full, budget, estimate) {
		return full, fullEnvelope, nil
	}
	preview := normalized.Manifest.Preview
	if len(preview.Ranges) > 0 && preview.Kind != PreviewKindFull && preview.Kind != PreviewKindPrefix && !hasByteRange {
		// A structured or multi-range view can include separators or other
		// transformations. Without an exact byte mapping, a shorter string
		// cannot truthfully retain its original source coverage.
		return "", Envelope{}, fmt.Errorf("%w: cannot shorten preview without a byte range mapping", ErrEnvelopeBudgetTooSmall)
	}
	fixed, fixedEnvelope, err := render("")
	if err != nil {
		return "", Envelope{}, err
	}
	if !fitsEnvelopeBudget(fixed, budget, estimate) {
		return "", Envelope{}, fmt.Errorf("%w: fixed envelope fields exceed max_bytes=%d or max_tokens=%d", ErrEnvelopeBudgetTooSmall, budget.MaxBytes, budget.MaxTokens)
	}

	low, high := 0, len(envelope.Preview)
	bestText := fixed
	bestEnvelope := fixedEnvelope
	for low <= high {
		mid := low + (high-low)/2
		preview := utf8Prefix(envelope.Preview, mid)
		encoded, candidate, err := render(preview)
		if err != nil {
			return "", Envelope{}, err
		}
		if fitsEnvelopeBudget(encoded, budget, estimate) {
			bestText = encoded
			bestEnvelope = candidate
			low = mid + 1
			continue
		}
		high = mid - 1
	}
	if bestEnvelope.Preview == "" && envelope.Preview != "" {
		return bestText, bestEnvelope, nil
	}
	return bestText, bestEnvelope, nil
}

func previewByteRange(preview Preview, previewBytes int) (Range, bool) {
	if preview.Kind != PreviewKindRange || len(preview.Ranges) != 1 {
		return Range{}, false
	}
	r := preview.Ranges[0]
	return r, r.Unit == RangeUnitBytes && r.End-r.Start == int64(previewBytes)
}

// IsFramedEnvelope reports whether text has the shape JSONEnvelopeCodec.Encode
// produces: after trimming whitespace it starts with the start marker and ends
// with the end marker. Ordinary text that merely mentions a marker is not framed.
func IsFramedEnvelope(text string) bool {
	text = strings.TrimSpace(text)
	return strings.HasPrefix(text, EnvelopeStartMarker) && strings.HasSuffix(text, EnvelopeEndMarker)
}

func (JSONEnvelopeCodec) Decode(text string) (Envelope, bool, error) {
	start := strings.Index(text, EnvelopeStartMarker)
	end := strings.Index(text, EnvelopeEndMarker)
	if start < 0 && end < 0 {
		return Envelope{}, false, nil
	}
	if start < 0 || end < 0 || end <= start {
		return Envelope{}, true, fmt.Errorf("artifact envelope decode: incomplete envelope markers")
	}
	payloadStart := start + len(EnvelopeStartMarker)
	payload := strings.TrimSpace(text[payloadStart:end])
	if payload == "" {
		return Envelope{}, true, fmt.Errorf("artifact envelope decode: empty payload")
	}
	var envelope Envelope
	if err := json.Unmarshal([]byte(payload), &envelope); err != nil {
		return Envelope{}, true, fmt.Errorf("artifact envelope decode: %w", err)
	}
	if err := envelope.Validate(); err != nil {
		return Envelope{}, true, err
	}
	return envelope, true, nil
}

func fitsEnvelopeBudget(encoded string, budget Budget, estimate func(string) int) bool {
	return len(encoded) <= budget.MaxBytes && estimate(encoded) <= budget.MaxTokens
}

func approximateTokens(text string) int {
	if text == "" {
		return 0
	}
	return (len(text) + 3) / 4
}

func safeTokenEstimator(estimate func(string) int) func(string) int {
	if estimate == nil {
		return approximateTokens
	}
	return func(text string) int {
		if text == "" {
			return 0
		}
		if tokens := estimate(text); tokens > 0 {
			return tokens
		}
		return approximateTokens(text)
	}
}

func visibleMeasurement(preview string, estimate func(string) int, complete bool) Measurement {
	bytesCount := int64(len(preview))
	tokenCount := int64(estimate(preview))
	lineCount := int64(0)
	if preview != "" {
		lineCount = int64(strings.Count(preview, "\n") + 1)
	}
	return Measurement{
		Bytes:             &bytesCount,
		Lines:             &lineCount,
		EstimatorTokens:   &tokenCount,
		SHA256:            DigestSHA256([]byte(preview)),
		MeasurementSource: "provider_codec",
		Complete:          &complete,
	}
}

func normalizePreview(preview Preview, visibleBytes int, shortened bool) Preview {
	out := preview
	if shortened {
		out.Truncated = true
	}
	if visibleBytes == 0 {
		out.Ranges = nil
		if out.Kind == PreviewKindFull {
			out.Kind = PreviewKindNone
		}
		return out
	}
	switch out.Kind {
	case PreviewKindPrefix, PreviewKindFull:
		out.Ranges = []Range{{Unit: RangeUnitBytes, Start: 0, End: int64(visibleBytes)}}
	}
	if out.Kind == PreviewKindFull && out.Truncated {
		out.Kind = PreviewKindPrefix
	}
	return out
}

func utf8Prefix(text string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(text) <= maxBytes {
		return text
	}
	cut := maxBytes
	for cut > 0 && !utf8.ValidString(text[:cut]) {
		cut--
	}
	return text[:cut]
}
