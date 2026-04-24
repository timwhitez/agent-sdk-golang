package openai

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/binary"
	"time"
)

const (
	defaultRetryMaxAttempts = 10
	defaultRetryBaseDelay   = 1 * time.Second
	defaultRetryMaxDelay    = 60 * time.Second
)

var backoffRandRead = cryptorand.Read

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
		if d > maxDelay {
			d = maxDelay
		}
	}
	// Keep a bounded 10% jitter so many clients avoid synchronized retries.
	jitter := time.Duration(randomBackoffFraction() * float64(d) * 0.1)
	return d + jitter
}

func sleepRetryBackoff(ctx context.Context, attempt int, baseDelay, maxDelay, retryAfter time.Duration) {
	d := resolveRetryDelay(attempt, baseDelay, maxDelay, retryAfter)
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return
	case <-t.C:
		return
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
