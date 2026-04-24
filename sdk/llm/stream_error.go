package llm

import (
	"errors"
	"strings"
)

// AsError returns the best error representation for a StreamErrorEvent.
// If Err is nil, it falls back to any available metadata.
func (e StreamErrorEvent) AsError() error {
	if e.Err != nil {
		return e.Err
	}
	if e.Provider == "" && e.StatusCode == 0 && strings.TrimSpace(e.Message) == "" && e.RetryAfter == 0 {
		return errors.New("stream error")
	}
	msg := strings.TrimSpace(e.Message)
	if msg == "" {
		msg = "stream error"
	}
	if e.StatusCode == 429 {
		return &RateLimitError{Provider: e.Provider, Message: msg, RetryAfter: e.RetryAfter}
	}
	return &ProviderError{
		Provider:   e.Provider,
		StatusCode: e.StatusCode,
		Message:    msg,
		RetryAfter: e.RetryAfter,
	}
}
