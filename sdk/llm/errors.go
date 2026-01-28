package llm

import "fmt"

// ProviderError represents an HTTP/API error from a model provider.
// StatusCode may be 0 when unavailable.
type ProviderError struct {
	Provider   string
	StatusCode int
	Message    string
}

func (e *ProviderError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.StatusCode != 0 {
		return fmt.Sprintf("%s error (%d): %s", e.Provider, e.StatusCode, e.Message)
	}
	return fmt.Sprintf("%s error: %s", e.Provider, e.Message)
}

// RateLimitError is a convenience type for retry logic.
type RateLimitError struct {
	Provider string
	Message  string
}

func (e *RateLimitError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%s rate limited: %s", e.Provider, e.Message)
}
