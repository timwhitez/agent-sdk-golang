package anthropic

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

const anthropicMaxRedirects = 10

type anthropicOrigin struct {
	scheme string
	host   string
	port   string
}

func originForAnthropicURL(value *url.URL) (anthropicOrigin, error) {
	if value == nil {
		return anthropicOrigin{}, fmt.Errorf("anthropic: redirect URL is missing")
	}
	scheme := strings.ToLower(strings.TrimSpace(value.Scheme))
	host := strings.ToLower(strings.TrimSpace(value.Hostname()))
	if scheme == "" || host == "" {
		return anthropicOrigin{}, fmt.Errorf("anthropic: redirect URL has no origin")
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
	return anthropicOrigin{scheme: scheme, host: host, port: port}, nil
}

func validateAnthropicRedirectTarget(origin anthropicOrigin, target *url.URL) error {
	destination, err := originForAnthropicURL(target)
	if err != nil {
		return err
	}
	if origin.scheme == "https" && destination.scheme != "https" {
		return fmt.Errorf("anthropic: refusing HTTPS redirect downgrade to %s", target.Redacted())
	}
	if origin != destination {
		return fmt.Errorf("anthropic: refusing cross-origin redirect to %s", target.Redacted())
	}
	return nil
}

// redirectSafeHTTPClient clones base and installs a mandatory policy
// before any redirected request is sent. The original client is never
// mutated. Allowed same-origin redirects still invoke the caller's
// callback, after which the target is checked again in case the
// callback changed req.URL.
func redirectSafeHTTPClient(base *http.Client) *http.Client {
	if base == nil {
		base = &http.Client{}
	}
	cloned := *base
	callerCheck := base.CheckRedirect
	cloned.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) == 0 || via[0] == nil || via[0].URL == nil {
			return fmt.Errorf("anthropic: redirect is missing its origin request")
		}
		if len(via) >= anthropicMaxRedirects {
			return fmt.Errorf("anthropic: stopped after %d redirects", anthropicMaxRedirects)
		}
		origin, err := originForAnthropicURL(via[0].URL)
		if err != nil {
			return err
		}
		if req == nil {
			return fmt.Errorf("anthropic: redirect request is missing")
		}
		if err := validateAnthropicRedirectTarget(origin, req.URL); err != nil {
			return err
		}
		if callerCheck != nil {
			if err := callerCheck(req, via); err != nil {
				return err
			}
			if err := validateAnthropicRedirectTarget(origin, req.URL); err != nil {
				return err
			}
		}
		return nil
	}
	return &cloned
}
