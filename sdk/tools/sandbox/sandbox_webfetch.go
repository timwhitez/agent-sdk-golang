package sandbox

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// ============================================================================
// webfetch tool: Fetch a URL over HTTP(S)
// ============================================================================

// webfetchArgs are the arguments for the webfetch tool.
type webfetchArgs struct {
	URL      string            `json:"url"`
	Method   string            `json:"method,omitempty"`  // GET|HEAD (default GET)
	Headers  map[string]string `json:"headers,omitempty"` // best-effort
	Timeout  int               `json:"timeout,omitempty"` // seconds
	MaxBytes int               `json:"max_bytes,omitempty"`
}

// webfetchMaxRedirects is the maximum number of redirects to follow.
const webfetchMaxRedirects = 10

// webfetchLookupIPAddrs is a package-level variable that can be overridden in tests.
var webfetchLookupIPAddrs = func(ctx context.Context, host string) ([]net.IPAddr, error) {
	return net.DefaultResolver.LookupIPAddr(ctx, host)
}

// SetWebfetchLookupIPAddrs sets the IP address lookup function for webfetch.
// This is primarily used for testing.
func SetWebfetchLookupIPAddrs(fn func(context.Context, string) ([]net.IPAddr, error)) {
	webfetchLookupIPAddrs = fn
}

// webfetchTool returns the webfetch tool implementation.
func webfetchTool() tools.Tool {
	return tools.Func[webfetchArgs]("webfetch", "Fetch a URL over HTTP(S) and return the response body (best-effort)", func(ctx context.Context, a webfetchArgs, deps *tools.Container) (any, error) {
		conf := getConfirmer(deps, ctx)
		rawURL := strings.TrimSpace(a.URL)
		if rawURL == "" {
			return "", fmt.Errorf("missing url")
		}
		u, err := url.Parse(rawURL)
		if err != nil {
			return "", fmt.Errorf("invalid url: %w", err)
		}
		scheme := strings.ToLower(strings.TrimSpace(u.Scheme))
		if scheme != "http" && scheme != "https" {
			return "", fmt.Errorf("only http/https is supported")
		}
		if err := validateWebfetchDestinationURL(ctx, u, "request target"); err != nil {
			return "", err
		}
		method := strings.ToUpper(strings.TrimSpace(a.Method))
		if method == "" {
			method = http.MethodGet
		}
		if method != http.MethodGet && method != http.MethodHead {
			return "", fmt.Errorf("only GET/HEAD is supported")
		}
		timeout := a.Timeout
		if timeout <= 0 {
			timeout = 30
		}
		maxBytes := a.MaxBytes
		if maxBytes <= 0 {
			maxBytes = 1024 * 1024
		}
		if maxBytes > 5*1024*1024 {
			maxBytes = 5 * 1024 * 1024
		}

		meta := attachToolCallMeta(ctx, map[string]any{
			"category": "network",
			"summary":  fmt.Sprintf("%s %s", method, rawURL),
			"url":      rawURL,
			"raw":      fmt.Sprintf("%s %s (timeout=%ds, max_bytes=%d)", method, rawURL, timeout, maxBytes),
		})
		ok, err := conf.Confirm(ctx, "webfetch", buildConfirmDetail(meta))
		if err != nil {
			return "", err
		}
		if !ok {
			denied, denyErr := denyToolResult(ctx, "webfetch", "user denied request")
			return denied.PlainText(), denyErr
		}

		hc := &http.Client{
			Timeout: time.Duration(timeout) * time.Second,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) >= webfetchMaxRedirects {
					return fmt.Errorf("stopped after %d redirects", webfetchMaxRedirects)
				}
				if req == nil || req.URL == nil {
					return fmt.Errorf("invalid redirect target")
				}
				return validateWebfetchDestinationURL(req.Context(), req.URL, "redirect target")
			},
		}
		req, err := http.NewRequestWithContext(ctx, method, rawURL, nil)
		if err != nil {
			return "", fmt.Errorf("build request: %w", err)
		}
		for k, v := range a.Headers {
			kk := strings.TrimSpace(k)
			vv := strings.TrimSpace(v)
			if kk != "" && vv != "" {
				req.Header.Set(kk, vv)
			}
		}
		resp, err := hc.Do(req)
		if err != nil {
			return "", fmt.Errorf("request failed: %w", err)
		}
		defer func() { _ = resp.Body.Close() }()

		var body []byte
		truncated := false
		if method != http.MethodHead {
			body, err = io.ReadAll(io.LimitReader(resp.Body, int64(maxBytes)+1))
			if err != nil {
				return "", fmt.Errorf("read response body after %d bytes (partial body): %w", len(body), err)
			}
			if len(body) > maxBytes {
				truncated = true
				body = body[:maxBytes]
			}
		}
		text := strings.TrimSpace(string(body))
		if text == "" {
			text = "(no body)"
		}
		if truncated {
			notice := fmt.Sprintf("Response body truncated after %d bytes. Increase max_bytes to see more.", maxBytes)
			return fmt.Sprintf("%s\n%s\n\n%s", resp.Status, notice, text), nil
		}
		return fmt.Sprintf("%s\n\n%s", resp.Status, text), nil
	})
}

// validateWebfetchDestinationURL validates that a URL destination is safe.
func validateWebfetchDestinationURL(ctx context.Context, target *url.URL, stage string) error {
	if target == nil {
		return fmt.Errorf("invalid url: missing host")
	}
	host := strings.TrimSpace(target.Hostname())
	if host == "" {
		return fmt.Errorf("invalid url: missing host")
	}
	return validateWebfetchHostDestination(ctx, host, stage)
}

// validateWebfetchHostDestination validates that a host destination is safe.
func validateWebfetchHostDestination(ctx context.Context, host, stage string) error {
	host = strings.TrimSpace(host)
	if host == "" {
		return fmt.Errorf("invalid url: missing host")
	}
	if stage == "" {
		stage = "request target"
	}
	if ip := parseWebfetchLiteralIP(host); ip != nil {
		if class := classifyWebfetchAddress(ip); class != "" {
			return webfetchDestinationDeniedError(stage, host, ip.String(), class)
		}
		return nil
	}
	addrs, err := webfetchLookupIPAddrs(ctx, host)
	if err != nil {
		return fmt.Errorf("cannot resolve %s %q: %v. Check hostname spelling and retry with a public URL", stage, host, err)
	}
	if len(addrs) == 0 {
		return fmt.Errorf("cannot resolve %s %q: no addresses returned. Check hostname spelling and retry with a public URL", stage, host)
	}
	usable := 0
	for _, addr := range addrs {
		if addr.IP == nil {
			continue
		}
		usable++
		if class := classifyWebfetchAddress(addr.IP); class != "" {
			return webfetchDestinationDeniedError(stage, host, addr.IP.String(), class)
		}
	}
	if usable == 0 {
		return fmt.Errorf("cannot resolve %s %q: no usable addresses returned. Check hostname spelling and retry with a public URL", stage, host)
	}
	return nil
}

// parseWebfetchLiteralIP parses a host string as a literal IP address.
func parseWebfetchLiteralIP(host string) net.IP {
	host = strings.TrimSpace(host)
	if host == "" {
		return nil
	}
	if idx := strings.LastIndex(host, "%"); idx > 0 {
		host = host[:idx]
	}
	return net.ParseIP(host)
}

// classifyWebfetchAddress classifies an IP address for security filtering.
// Returns a non-empty classification string if the address should be denied.
func classifyWebfetchAddress(ip net.IP) string {
	if ip == nil {
		return ""
	}
	if ip.IsLoopback() {
		return "loopback"
	}
	if ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
		return "link-local"
	}
	if ip.IsPrivate() {
		if ip.To4() != nil {
			return "private RFC1918"
		}
		return "private"
	}
	return ""
}

// webfetchDestinationDeniedError returns an error for a denied destination.
func webfetchDestinationDeniedError(stage, host, ip, class string) error {
	if stage == "" {
		stage = "request target"
	}
	return fmt.Errorf("blocked %s %q: resolved to %s (%s). Use a public internet URL (loopback/private/link-local destinations are denied)", stage, host, ip, class)
}
