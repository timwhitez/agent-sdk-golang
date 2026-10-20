package llm

import (
	"context"
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
	retryAfter                time.Duration
	notBefore                 time.Time
}

// StreamOutcomeSnapshot is a bounded receipt, independent of event delivery.
// Complete is true only after the original producer closes, with one owner.
// Error preserves a native retry stop first, otherwise the largest parsed
// Retry-After cause, otherwise an ordinary error. Treat native causes as private
// provider errors, with the same diagnostic policy as InvokeStream errors.
type StreamOutcomeSnapshot struct {
	Started, Complete, Visible bool
	Error                      error
	RetryAfter                 time.Duration
	RetryAfterNotBefore        time.Time
}

// WithStreamOutcome creates a fresh scope. Do not reuse it for another call,
// including another invocation on the same model or an automatic retry.
func WithStreamOutcome(ctx context.Context) (context.Context, *StreamOutcome) {
	o := &StreamOutcome{}
	return context.WithValue(ctx, streamOutcomeKey{}, o), o
}

// StreamOutcomeRecorder is owned by a provider's original forwarding producer.
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
		hint := RetryAfterForError(err)
		if hint > 0 {
			at := time.Now().Add(hint)
			if at.After(o.notBefore) {
				o.notBefore, o.retryAfter, o.err = at, hint, err
			}
		} else if o.err == nil {
			o.err = err
		}
	case StreamTextDeltaEvent:
		o.visible = o.visible || e.Delta != ""
	case StreamThinkingDeltaEvent:
		o.visible = o.visible || e.Delta != ""
	case StreamToolCallDeltaEvent:
		o.visible = o.visible || strings.TrimSpace(e.ID) != "" || e.NameDelta != "" || e.ArgumentsDelta != ""
	}
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
	return StreamOutcomeSnapshot{Started: o.started, Complete: o.complete && !o.reused, Visible: o.visible, Error: err, RetryAfter: o.retryAfter, RetryAfterNotBefore: o.notBefore}
}
