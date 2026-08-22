package llm

import (
	"errors"
	"fmt"
	"time"
)

// ProviderError represents an HTTP/API error from a model provider.
// StatusCode may be 0 when unavailable.
type ProviderError struct {
	Provider   string
	StatusCode int
	Message    string
	RetryAfter time.Duration
}

func (e *ProviderError) Error() string {
	if e == nil {
		return "<nil>"
	}
	msg := e.Message
	if e.RetryAfter > 0 {
		if msg == "" {
			msg = fmt.Sprintf("retry after %s", e.RetryAfter)
		} else {
			msg = fmt.Sprintf("%s (retry after %s)", msg, e.RetryAfter)
		}
	}
	if e.StatusCode != 0 {
		return fmt.Sprintf("%s error (%d): %s", e.Provider, e.StatusCode, msg)
	}
	return fmt.Sprintf("%s error: %s", e.Provider, msg)
}

// RateLimitError is a convenience type for retry logic.
type RateLimitError struct {
	Provider   string
	Message    string
	RetryAfter time.Duration
}

func (e *RateLimitError) Error() string {
	if e == nil {
		return "<nil>"
	}
	msg := e.Message
	if e.RetryAfter > 0 {
		if msg == "" {
			msg = fmt.Sprintf("retry after %s", e.RetryAfter)
		} else {
			msg = fmt.Sprintf("%s (retry after %s)", msg, e.RetryAfter)
		}
	}
	return fmt.Sprintf("%s rate limited: %s", e.Provider, msg)
}

// SteeringInterruptError is returned when a user steering message
// interrupts an in-progress LLM stream. This is a special error type
// that signals the agent loop to immediately incorporate the steering
// message and continue execution, rather than treating it as a fatal error.
type SteeringInterruptError struct {
	Message string
}

func (e *SteeringInterruptError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Message == "" {
		return "stream interrupted by user steering"
	}
	return fmt.Sprintf("stream interrupted by user steering: %s", e.Message)
}

// IsSteeringInterrupt reports whether err is a SteeringInterruptError.
func IsSteeringInterrupt(err error) bool {
	var steer *SteeringInterruptError
	return err != nil && errors.As(err, &steer)
}

// StreamIdleTimeoutError is returned when a streaming provider stops producing
// events for too long before the stream reaches a terminal state.
//
// It implements net.Error-style timeout semantics so higher layers can classify
// it as a timeout without relying on string matching.
type StreamIdleTimeoutError struct {
	Duration time.Duration
}

func (e *StreamIdleTimeoutError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Duration <= 0 {
		return "stream idle timeout"
	}
	return fmt.Sprintf("stream idle timeout after %s", e.Duration)
}

func (e *StreamIdleTimeoutError) Timeout() bool { return true }

func (e *StreamIdleTimeoutError) Temporary() bool { return true }

// IncompleteStreamError reports that a streaming provider ended transport
// delivery without the explicit terminal event required by the SDK contract.
// Partial content may still be returned alongside this error.
type IncompleteStreamError struct {
	Provider string
	Model    string
	Message  string
}

func (e *IncompleteStreamError) Error() string {
	if e == nil {
		return "<nil>"
	}
	provider := e.Provider
	if provider == "" {
		provider = "provider"
	}
	message := e.Message
	if message == "" {
		message = "stream closed before terminal event"
	}
	if e.Model != "" {
		return fmt.Sprintf("%s stream incomplete for model %s: %s", provider, e.Model, message)
	}
	return fmt.Sprintf("%s stream incomplete: %s", provider, message)
}
