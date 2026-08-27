package openai

import (
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
	return &url.Error{Op: urlErr.Op, URL: safeURL, Err: urlErr.Err}
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
			return fmt.Errorf("openai: redirect is missing its origin request")
		}
		if len(via) >= openAIMaxRedirects {
			return fmt.Errorf("openai: stopped after %d redirects", openAIMaxRedirects)
		}
		origin, err := originForOpenAIURL(via[0].URL)
		if err != nil {
			return err
		}
		if req == nil {
			return fmt.Errorf("openai: redirect request is missing")
		}
		if err := validateOpenAIRedirectTarget(origin, req.URL); err != nil {
			return err
		}
		if callerCheck != nil {
			if err := callerCheck(req, via); err != nil {
				return err
			}
			if err := validateOpenAIRedirectTarget(origin, req.URL); err != nil {
				return err
			}
		}
		return nil
	}
	return &cloned
}
