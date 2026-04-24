package anthropic

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

var maxProviderResponseBytes int64 = 8 * 1024 * 1024

type oversizedResponseBodyError struct {
	Endpoint string
	Read     int64
	Limit    int64
}

func (e *oversizedResponseBodyError) Error() string {
	if e == nil {
		return ""
	}
	return fmt.Sprintf(
		"provider response body too large (read=%d bytes, limit=%d bytes, endpoint=%s) - Request a smaller response or retry with lower max output tokens",
		e.Read,
		e.Limit,
		e.Endpoint,
	)
}

func readResponseBodyLimited(body io.ReadCloser, endpoint string) ([]byte, error) {
	defer body.Close()

	limit := maxProviderResponseBytes
	if limit <= 0 {
		limit = 8 * 1024 * 1024
	}

	reader := &io.LimitedReader{R: body, N: limit + 1}
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, &oversizedResponseBodyError{
			Endpoint: endpoint,
			Read:     int64(len(data)),
			Limit:    limit,
		}
	}
	return data, nil
}

func anthropicReadBodyError(statusCode int, retryAfter time.Duration, err error) error {
	if err == nil {
		return nil
	}
	var oversized *oversizedResponseBodyError
	if !errors.As(err, &oversized) {
		return err
	}
	msg := oversized.Error()
	if statusCode == http.StatusTooManyRequests {
		return &llm.RateLimitError{Provider: "anthropic", Message: msg, RetryAfter: retryAfter}
	}
	return &llm.ProviderError{Provider: "anthropic", StatusCode: statusCode, Message: msg, RetryAfter: retryAfter}
}
