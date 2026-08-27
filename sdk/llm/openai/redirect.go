package openai

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
)

const openAIMaxRedirects = 10

type openAIOrigin struct {
	scheme string
	host   string
	port   string
}

type openAIRetryDecision interface {
	openAIRetryable() bool
}

type openAIRedirectPolicyError struct {
	err error
}

func newOpenAIRedirectPolicyError(err error) *openAIRedirectPolicyError {
	return &openAIRedirectPolicyError{err: err}
}

func (e *openAIRedirectPolicyError) Error() string {
	if e == nil || e.err == nil {
		return "openai: redirect rejected"
	}
	return e.err.Error()
}

func (e *openAIRedirectPolicyError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.err
}

func (*openAIRedirectPolicyError) openAIRetryable() bool { return false }

type openAISanitizedRequestError struct{}

func (openAISanitizedRequestError) Error() string         { return "HTTP request failed" }
func (openAISanitizedRequestError) openAIRetryable() bool { return false }

type openAISanitizedNetworkError struct {
	retryable bool
}

func (openAISanitizedNetworkError) Error() string { return "network request failed" }
func (e openAISanitizedNetworkError) openAIRetryable() bool {
	return e.retryable
}

type openAISanitizedTimeoutError struct{}

func (openAISanitizedTimeoutError) Error() string         { return "network request timed out" }
func (openAISanitizedTimeoutError) Timeout() bool         { return true }
func (openAISanitizedTimeoutError) Temporary() bool       { return true }
func (openAISanitizedTimeoutError) openAIRetryable() bool { return true }

func originForOpenAIURL(value *url.URL) (openAIOrigin, error) {
	if value == nil {
		return openAIOrigin{}, fmt.Errorf("openai: redirect URL is missing")
	}
	scheme := strings.ToLower(strings.TrimSpace(value.Scheme))
	host := strings.ToLower(strings.TrimSpace(value.Hostname()))
	if scheme == "" || host == "" {
		return openAIOrigin{}, fmt.Errorf("openai: redirect URL has no origin")
	}
	port := strings.TrimSpace(value.Port())
	if port == "" {
		switch scheme {
		case "http":
			port = "80"
		case "https":
			port = "443"
		}
	}
	return openAIOrigin{scheme: scheme, host: host, port: port}, nil
}

func validateOpenAIRedirectTarget(origin openAIOrigin, target *url.URL) error {
	destination, err := originForOpenAIURL(target)
	if err != nil {
		return err
	}
	if origin.scheme == "https" && destination.scheme != "https" {
		return fmt.Errorf("openai: refusing HTTPS redirect downgrade to %s://%s:%s", destination.scheme, destination.host, destination.port)
	}
	if origin != destination {
		return fmt.Errorf("openai: refusing cross-origin redirect to %s://%s:%s", destination.scheme, destination.host, destination.port)
	}
	if target.User != nil {
		return errors.New("openai: refusing redirect with embedded URL credentials")
	}
	return nil
}

func openAIOriginLabel(value *url.URL) string {
	origin, err := originForOpenAIURL(value)
	if err != nil {
		return "(redacted URL)"
	}
	return origin.scheme + "://" + net.JoinHostPort(origin.host, origin.port)
}

func sanitizeOpenAIHTTPError(err error) error {
	if err == nil {
		return nil
	}
	var urlErr *url.Error
	if !errors.As(err, &urlErr) || urlErr == nil {
		return err
	}
	safeURL := "(redacted URL)"
	if parsed, parseErr := url.Parse(urlErr.URL); parseErr == nil {
		safeURL = openAIOriginLabel(parsed)
	}
	var cause error = openAISanitizedRequestError{}
	var policyErr *openAIRedirectPolicyError
	if errors.As(urlErr.Err, &policyErr) && policyErr != nil {
		cause = policyErr
	} else if errors.Is(urlErr.Err, context.Canceled) {
		cause = context.Canceled
	} else if errors.Is(urlErr.Err, context.DeadlineExceeded) {
		cause = context.DeadlineExceeded
	} else {
		var netErr net.Error
		if errors.As(urlErr.Err, &netErr) {
			if netErr.Timeout() {
				cause = openAISanitizedTimeoutError{}
			} else {
				cause = openAISanitizedNetworkError{retryable: retryableOpenAINetworkError(urlErr.Err)}
			}
		}
	}
	return &url.Error{Op: urlErr.Op, URL: safeURL, Err: cause}
}

func retryableOpenAINetworkError(err error) bool {
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) && dnsErr != nil && dnsErr.IsNotFound {
		return false
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr != nil {
		return true
	}
	var temporary interface{ Temporary() bool }
	return errors.As(err, &temporary) && temporary.Temporary()
}

// redirectSafeOpenAIHTTPClient clones base and installs a mandatory policy
// before any redirected request is sent. Allowed same-origin redirects still
// invoke the caller callback, after which the target is checked again in case
// that callback changed req.URL.
func redirectSafeOpenAIHTTPClient(base *http.Client) *http.Client {
	if base == nil {
		base = &http.Client{}
	}
	cloned := *base
	callerCheck := base.CheckRedirect
	cloned.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) == 0 || via[0] == nil || via[0].URL == nil {
			return newOpenAIRedirectPolicyError(errors.New("openai: redirect is missing its origin request"))
		}
		if len(via) >= openAIMaxRedirects {
			return newOpenAIRedirectPolicyError(fmt.Errorf("openai: stopped after %d redirects", openAIMaxRedirects))
		}
		origin, err := originForOpenAIURL(via[0].URL)
		if err != nil {
			return newOpenAIRedirectPolicyError(err)
		}
		if req == nil {
			return newOpenAIRedirectPolicyError(errors.New("openai: redirect request is missing"))
		}
		if err := validateOpenAIRedirectTarget(origin, req.URL); err != nil {
			return newOpenAIRedirectPolicyError(err)
		}
		if callerCheck != nil {
			if err := callerCheck(req, via); err != nil {
				return err
			}
			if err := validateOpenAIRedirectTarget(origin, req.URL); err != nil {
				return newOpenAIRedirectPolicyError(err)
			}
		}
		return nil
	}
	return &cloned
}
