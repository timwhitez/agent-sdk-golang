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
	render := func(preview string) (string, Envelope, error) {
		candidate := normalized.Clone()
		candidate.Preview = preview
		candidate.Manifest.Preview = normalizePreview(candidate.Manifest.Preview, len(preview), len(preview) < len(normalized.Preview))
		candidate.Manifest.VisibleMeasurement = visibleMeasurement(preview, estimate, !candidate.Manifest.Preview.Truncated)
		if candidate.Continuation != nil && candidate.Continuation.RangeUnit == RangeUnitBytes && candidate.Continuation.Next == 0 {
			candidate.Continuation.Next = int64(len(preview))
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

	fixed, fixedEnvelope, err := render("")
	if err != nil {
		return "", Envelope{}, err
	}
	if !fitsEnvelopeBudget(fixed, budget, estimate) {
		return "", Envelope{}, fmt.Errorf("%w: fixed envelope fields exceed max_bytes=%d or max_tokens=%d", ErrEnvelopeBudgetTooSmall, budget.MaxBytes, budget.MaxTokens)
	}
	full, fullEnvelope, err := render(envelope.Preview)
	if err != nil {
		return "", Envelope{}, err
	}
	if fitsEnvelopeBudget(full, budget, estimate) {
		return full, fullEnvelope, nil
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
