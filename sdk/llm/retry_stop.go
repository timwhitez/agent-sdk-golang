package llm

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// RetryStopError ends automatic retries of the failed invocation. RetryAfter
// retains the original parsed server floor; Cause retains provider evidence.
// Callers must check IsRetryStop before unwrapping/classifying Cause. Starting
// another automatic retry immediately would violate the server floor.
type RetryStopError struct {
	RetryAfter time.Duration
	Reason     string // "wait_limit" or "deadline"
	Cause      error  `json:"-"`
}

func (e *RetryStopError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return fmt.Sprintf("automatic retry stopped: server floor %s (%s)", e.RetryAfter, e.Reason)
}
func (e *RetryStopError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}
func (e *RetryStopError) AutomaticRetryStopped() bool { return e != nil }

// IsRetryStop also recognizes the same marker implemented by a host wrapper.
func IsRetryStop(err error) bool {
	var stop interface{ AutomaticRetryStopped() bool }
	return errors.As(err, &stop) && stop.AutomaticRetryStopped()
}

// RetryAfterForError extracts the parsed hint without changing classification.
func RetryAfterForError(err error) time.Duration {
	var rate *RateLimitError
	if errors.As(err, &rate) && rate != nil {
		return rate.RetryAfter
	}
	var provider *ProviderError
	if errors.As(err, &provider) && provider != nil {
		return provider.RetryAfter
	}
	return 0
}

// BoundRetryAfter converts a valid hint that cannot fit this retry owner's
// finite wait/deadline to a terminal typed stop. Otherwise it returns err.
// Invoke only after deciding that the original error is retryable.
func BoundRetryAfter(ctx context.Context, err error, maxWait time.Duration) error {
	if err == nil || IsRetryStop(err) {
		return err
	}
	hint := RetryAfterForError(err)
	if hint <= 0 {
		return err
	}
	reason := ""
	if maxWait > 0 && hint > maxWait {
		reason = "wait_limit"
	}
	if deadline, ok := ctx.Deadline(); ok && hint >= time.Until(deadline) {
		reason = "deadline"
	}
	if reason == "" {
		return err
	}
	return &RetryStopError{RetryAfter: hint, Reason: reason, Cause: err}
}
