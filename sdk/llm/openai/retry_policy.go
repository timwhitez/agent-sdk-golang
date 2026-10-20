package openai

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/binary"
	"math"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	defaultRetryMaxAttempts = 10
	defaultRetryBaseDelay   = 1 * time.Second
	defaultRetryMaxDelay    = 60 * time.Second
)

var backoffRandRead = cryptorand.Read

type retryBackoffTestHookKey struct{}
type retryBackoffTestHooks struct{ waiting func(error, time.Duration) }

type retryPolicy struct {
	maxRetries int
	baseDelay  time.Duration
	maxDelay   time.Duration
}

func resolveRetryPolicy(maxRetries int, baseDelay, maxDelay time.Duration) retryPolicy {
	if maxRetries <= 0 {
		maxRetries = defaultRetryMaxAttempts
	}
	if baseDelay <= 0 {
		baseDelay = defaultRetryBaseDelay
	}
	if maxDelay <= 0 {
		maxDelay = defaultRetryMaxDelay
	}
	return retryPolicy{
		maxRetries: maxRetries,
		baseDelay:  baseDelay,
		maxDelay:   maxDelay,
	}
}

func resolveRetryDelay(attempt int, baseDelay, maxDelay time.Duration, retryAfter time.Duration) time.Duration {
	d := exponentialBackoffDelay(attempt, baseDelay, maxDelay)
	if retryAfter > d {
		d = retryAfter
	}
	// Keep a bounded 10% jitter so many clients avoid synchronized retries.
	jitter := time.Duration(randomBackoffFraction() * float64(d) * 0.1)
	if jitter > time.Duration(math.MaxInt64)-d {
		return time.Duration(math.MaxInt64)
	}
	return d + jitter
}

// maxHintWait is separate from the local exponential backoff ceiling.
func sleepRetryBackoff(ctx context.Context, attempt int, baseDelay, maxDelay, retryAfter, maxHintWait time.Duration, cause error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if maxHintWait <= 0 {
		maxHintWait = defaultRetryMaxDelay
	}
	bounded := llm.BoundRetryAfter(ctx, cause, maxHintWait)
	if llm.IsRetryStop(bounded) {
		return bounded
	}
	d := resolveRetryDelay(attempt, baseDelay, maxDelay, retryAfter)
	if retryAfter > 0 {
		// Jitter may consume spare waiting capacity, but never lower the floor.
		if d > maxHintWait {
			d = maxHintWait
		}
		if deadline, ok := ctx.Deadline(); ok && d >= time.Until(deadline) {
			return &llm.RetryStopError{RetryAfter: retryAfter, Reason: "deadline", Cause: cause}
		}
	}
	t := time.NewTimer(d)
	defer t.Stop()
	if hook, _ := ctx.Value(retryBackoffTestHookKey{}).(*retryBackoffTestHooks); hook != nil && hook.waiting != nil {
		hook.waiting(cause, d)
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func randomBackoffFraction() float64 {
	const scale = float64(1 << 53)

	var b [8]byte
	if _, err := backoffRandRead(b[:]); err == nil {
		sample := binary.BigEndian.Uint64(b[:]) >> 11
		return float64(sample) / scale
	}

	// Keep non-deterministic-enough jitter if entropy source is unavailable.
	n := uint64(time.Now().UnixNano())
	n ^= n << 13
	n ^= n >> 7
	n ^= n << 17
	return float64(n>>11) / scale
}
