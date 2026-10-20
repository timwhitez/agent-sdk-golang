package llm

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"
)

type streamOutcomeKey struct{}

// StreamOutcome retains producer facts for exactly one concrete InvokeStream.
// It is host-owned, invocation-local and never serialized into model input.
// Delivery cancellation does not erase facts already produced by the provider.
type StreamOutcome struct {
	mu                        sync.Mutex
	started, complete, reused bool
	visible                   bool
	err, terminal             error
	finalError                error
	hintCause                 error
	retryAfter                time.Duration
	notBefore                 time.Time
	lastHintCause             error
	lastHintNotBefore         time.Time
}

// StreamOutcomeSnapshot is a bounded receipt, independent of event delivery.
// Complete is true only after the original producer closes, with one owner.
// Error preserves a native retry stop first, otherwise the latest observed error.
// Cancellation never replaces an already observed provider error. The historical
// maximum absolute Retry-After floor and its original cause are separate facts;
// a later permanent error must remain available to a host's retry classifier.
// Errors may describe an earlier transient failure after a successful internal
// retry. FinalError records only an actual terminal producer error; nil alone
// does not define success, and Complete only certifies producer closure. All causes retain
// the same private diagnostic policy as InvokeStream errors.
type StreamOutcomeSnapshot struct {
	Started, Complete, Visible bool
	Error                      error
	FinalError                 error
	RetryAfterCause            error
	RetryAfter                 time.Duration
	RetryAfterNotBefore        time.Time
}

// WithStreamOutcome creates a fresh scope. Do not reuse it for another call,
// including another invocation on the same model or an automatic retry.
func WithStreamOutcome(ctx context.Context) (context.Context, *StreamOutcome) {
	o := &StreamOutcome{}
	return context.WithValue(ctx, streamOutcomeKey{}, o), o
}

// StreamOutcomeRecorder belongs to one accepted provider stream. Its parser
// and forwarder may publish concurrently; only the original forwarder finishes
// the receipt after source closure.
type StreamOutcomeRecorder struct{ outcome *StreamOutcome }

// BeginStreamOutcome claims the optional host scope. Custom/legacy callers
// without a scope retain their existing behavior and receive a nil recorder.
// Multiple producers claiming one scope invalidate its completion receipt.
func BeginStreamOutcome(ctx context.Context) *StreamOutcomeRecorder {
	o, _ := ctx.Value(streamOutcomeKey{}).(*StreamOutcome)
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.started {
		o.reused = true
		return nil
	}
	o.started = true
	return &StreamOutcomeRecorder{outcome: o}
}

// Observe records an actual parsed event before any lossy/cancelable delivery.
// Observe and Finish are nil-safe when the caller did not request a receipt.
func (r *StreamOutcomeRecorder) Observe(event StreamEvent) {
	if r == nil || r.outcome == nil {
		return
	}
	o := r.outcome
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.complete {
		return
	}
	switch e := event.(type) {
	case StreamErrorEvent:
		err := e.AsError()
		if IsRetryStop(err) && o.terminal == nil {
			o.terminal = err
		}
		if o.err == nil || (!errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded)) {
			o.err = err
		}
		hint := RetryAfterForError(err)
		if hint > 0 {
			at := time.Now().Add(hint)
			origin := retryAfterCauseIdentity(err)
			if origin != nil && origin == o.lastHintCause {
				// Delivery/native-stop wrapping of this same parsed provider cause
				// must not restart its absolute floor after waiting/cancellation.
				at = o.lastHintNotBefore
			} else {
				o.lastHintCause, o.lastHintNotBefore = origin, at
			}
			if at.After(o.notBefore) {
				o.notBefore, o.retryAfter, o.hintCause = at, hint, err
			}
		}
	case StreamTextDeltaEvent:
		o.visible = o.visible || e.Delta != ""
	case StreamThinkingDeltaEvent:
		o.visible = o.visible || e.Delta != ""
	case StreamToolCallDeltaEvent:
		o.visible = o.visible || strings.TrimSpace(e.ID) != "" || e.NameDelta != "" || e.ArgumentsDelta != ""
	}
}

// RecordFinalError records an actual terminal producer error before its public
// delivery. Do not call it for an internal retry's historical parse observation.
// A successful recovered stream leaves FinalError nil. Native stops stay sticky.
func (r *StreamOutcomeRecorder) RecordFinalError(err error) {
	if r == nil || r.outcome == nil || err == nil {
		return
	}
	r.Observe(StreamErrorEvent{Err: err})
	o := r.outcome
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.complete || IsRetryStop(o.finalError) {
		return
	}
	if o.finalError == nil || IsRetryStop(err) || (!errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded)) {
		o.finalError = err
	}
}

// Positive parsed hints come from these native pointer error types, including
// through RetryStopError wrapping. No arbitrary error-interface comparison or
// unbounded per-error map is needed to retain the current parser observation.
func retryAfterCauseIdentity(err error) error {
	var rate *RateLimitError
	if errors.As(err, &rate) && rate != nil {
		return rate
	}
	var provider *ProviderError
	if errors.As(err, &provider) && provider != nil {
		return provider
	}
	return nil
}

// Finish publishes after source closure, before downstream channel closure.
// It is not a server success receipt and does not overwrite error/visible facts.
func (r *StreamOutcomeRecorder) Finish() {
	if r == nil || r.outcome == nil {
		return
	}
	r.outcome.mu.Lock()
	r.outcome.complete = true
	r.outcome.mu.Unlock()
}

// Snapshot returns immutable values under synchronization. Missing or reused
// scopes cannot certify completion. A finished receipt never accepts late facts.
func (o *StreamOutcome) Snapshot() StreamOutcomeSnapshot {
	if o == nil {
		return StreamOutcomeSnapshot{}
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	err := o.err
	if o.terminal != nil {
		err = o.terminal
	}
	return StreamOutcomeSnapshot{Started: o.started, Complete: o.complete && !o.reused, Visible: o.visible, Error: err, FinalError: o.finalError, RetryAfterCause: o.hintCause, RetryAfter: o.retryAfter, RetryAfterNotBefore: o.notBefore}
}
