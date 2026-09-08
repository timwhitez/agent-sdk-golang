package llm

import "context"

type warningSinkKey struct{}
type warningSinkValue struct{ sink func(string, ...any) }

// WithWarningSink binds diagnostics to one invocation without mutating a shared
// model. Derived cancellation/deadline contexts preserve this binding. A nil
// sink masks an inherited binding and restores the provider's fallback.
func WithWarningSink(ctx context.Context, sink func(string, ...any)) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, warningSinkKey{}, warningSinkValue{sink: sink})
}

// WarningSink returns the invocation binding, or fallback when no binding exists.
// Providers should capture it in their call-local state before starting streams
// or retries. The callback must support the concurrency of its owning caller.
func WarningSink(ctx context.Context, fallback func(string, ...any)) func(string, ...any) {
	if ctx != nil {
		if value, ok := ctx.Value(warningSinkKey{}).(warningSinkValue); ok && value.sink != nil {
			return value.sink
		}
	}
	return fallback
}
