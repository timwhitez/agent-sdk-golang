package sandbox

import (
	"fmt"
	"math"
	"time"
)

const maxSandboxTimeoutSeconds = 24 * 60 * 60

func checkedSandboxTimeout(rawSeconds, defaultSeconds int) (int, time.Duration, error) {
	seconds := rawSeconds
	if seconds <= 0 {
		seconds = defaultSeconds
	}
	if seconds <= 0 {
		return 0, 0, fmt.Errorf("timeout must resolve to a positive number of seconds")
	}
	if int64(seconds) > math.MaxInt64/int64(time.Second) {
		return 0, 0, fmt.Errorf("timeout %d seconds exceeds time.Duration range", seconds)
	}
	if seconds > maxSandboxTimeoutSeconds {
		return 0, 0, fmt.Errorf("timeout %d seconds exceeds the maximum allowed %d seconds", seconds, maxSandboxTimeoutSeconds)
	}
	duration := time.Duration(seconds) * time.Second
	if duration <= 0 {
		return 0, 0, fmt.Errorf("timeout %d seconds overflowed time.Duration", seconds)
	}
	return seconds, duration, nil
}
