package sandbox

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
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

// webfetchDialContext is the final socket dial seam. The production path is
// always called with a literal IP selected from the just-validated DNS result.
var webfetchDialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
	return (&net.Dialer{}).DialContext(ctx, network, address)
}

// webfetchDoRequest is the HTTP execution seam used by package tests. The
// production value always executes the request with the validated client built
// below; unlike replacing http.DefaultTransport, overriding this seam cannot
// silently alter the transport policy in a deployed process.
var webfetchDoRequest = func(client *http.Client, request *http.Request) (*http.Response, error) {
	return client.Do(request)
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
		// Before the user authorizes network access, validate only syntax and
		// literal-IP policy. Hostname resolution is itself an observable network
		// action and is deferred to the socket-bound validator after confirmation.
		if err := validateWebfetchPreConfirmationURL(u, "request target"); err != nil {
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

		hc := newWebfetchHTTPClient(time.Duration(timeout) * time.Second)
		hc.CheckRedirect = func(req *http.Request, via []*http.Request) error {
			if len(via) >= webfetchMaxRedirects {
				return newWebfetchSafeRequestDiagnostic(fmt.Errorf("stopped after %d redirects", webfetchMaxRedirects))
			}
			if req == nil || req.URL == nil {
				return newWebfetchSafeRequestDiagnostic(errors.New("invalid redirect target"))
			}
			if len(via) == 0 || via[len(via)-1] == nil || via[len(via)-1].URL == nil {
				return newWebfetchSafeRequestDiagnostic(errors.New("invalid redirect source"))
			}
			previous := via[len(via)-1].URL
			if !sameWebfetchOrigin(previous, req.URL) {
				return newWebfetchSafeRequestDiagnostic(fmt.Errorf(
					"redirect target changes origin from %s to %s; retry the target URL directly so the new origin is confirmed",
					webfetchOriginLabel(previous),
					webfetchOriginLabel(req.URL),
				))
			}
			if err := validateWebfetchDestinationURL(req.Context(), req.URL, "redirect target"); err != nil {
				return newWebfetchSafeRequestDiagnostic(err)
			}
			return nil
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
		resp, err := webfetchDoRequest(hc, req)
		if err != nil {
			return "", fmt.Errorf("request failed: %w", sanitizeWebfetchRequestError(err))
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

func newWebfetchHTTPClient(timeout time.Duration) *http.Client {
	// Build from fixed, package-owned defaults instead of cloning the process
	// global transport. A mutated *http.Transport can carry DialTLSContext or
	// DialTLS hooks that bypass DialContext for HTTPS, as well as proxy or TLS
	// policy that is outside WebFetch's destination-validation boundary.
	transport := &http.Transport{
		Proxy:                 nil,
		DialContext:           dialValidatedWebfetchDestination,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: time.Second,
	}
	return &http.Client{Timeout: timeout, Transport: transport}
}

// dialValidatedWebfetchDestination resolves and classifies the exact address
// set used for this socket, then dials a selected literal IP. The original host
// remains on the request URL, preserving the HTTP Host header and TLS SNI.
func dialValidatedWebfetchDestination(ctx context.Context, network, address string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, newWebfetchSafeRequestDiagnostic(fmt.Errorf("invalid webfetch dial address: %w", err))
	}
	addrs, err := resolveAndValidateWebfetchHost(ctx, host, "socket target")
	if err != nil {
		return nil, newWebfetchSafeRequestDiagnostic(err)
	}
	var dialErrors []string
	for _, ip := range addrs {
		conn, dialErr := webfetchDialContext(ctx, network, net.JoinHostPort(ip.String(), port))
		if dialErr == nil {
			return conn, nil
		}
		dialErrors = append(dialErrors, dialErr.Error())
	}
	return nil, newWebfetchSafeRequestDiagnostic(fmt.Errorf("cannot connect to validated socket target %q: %s", host, strings.Join(dialErrors, "; ")))
}

// validateWebfetchPreConfirmationURL performs only local checks. A hostname is
// intentionally not resolved until after the user has approved the request.
func validateWebfetchPreConfirmationURL(target *url.URL, stage string) error {
	if target == nil {
		return fmt.Errorf("invalid url: missing host")
	}
	if target.User != nil {
		return fmt.Errorf("%s must not include embedded URL credentials; remove userinfo and retry with an explicitly confirmed header", stage)
	}
	host := strings.TrimSpace(target.Hostname())
	if host == "" {
		return fmt.Errorf("invalid url: missing host")
	}
	if ip := parseWebfetchLiteralIP(host); ip != nil {
		if class := classifyWebfetchAddress(ip); class != "" {
			return webfetchDestinationDeniedError(stage, host, ip.String(), class)
		}
	}
	return nil
}

func sameWebfetchOrigin(left, right *url.URL) bool {
	if left == nil || right == nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(left.Scheme), strings.TrimSpace(right.Scheme)) &&
		strings.EqualFold(strings.TrimSpace(left.Hostname()), strings.TrimSpace(right.Hostname())) &&
		webfetchEffectivePort(left) == webfetchEffectivePort(right)
}

func webfetchEffectivePort(target *url.URL) string {
	if target == nil {
		return ""
	}
	if port := strings.TrimSpace(target.Port()); port != "" {
		return port
	}
	switch strings.ToLower(strings.TrimSpace(target.Scheme)) {
	case "http":
		return "80"
	case "https":
		return "443"
	default:
		return ""
	}
}

func webfetchOriginLabel(target *url.URL) string {
	if target == nil {
		return "(invalid origin)"
	}
	scheme := strings.ToLower(strings.TrimSpace(target.Scheme))
	host := strings.ToLower(strings.TrimSpace(target.Hostname()))
	port := webfetchEffectivePort(target)
	if host == "" || port == "" {
		return "(invalid origin)"
	}
	return scheme + "://" + net.JoinHostPort(host, port)
}

func sanitizeWebfetchRequestError(err error) error {
	if err == nil {
		return nil
	}
	var urlErr *url.Error
	if !errors.As(err, &urlErr) || urlErr == nil {
		return err
	}
	safeURL := "(redacted URL)"
	if parsed, parseErr := url.Parse(urlErr.URL); parseErr == nil {
		safeURL = webfetchOriginLabel(parsed)
	}
	var cause error = errors.New("HTTP request failed")
	var safeDiagnostic *webfetchSafeRequestDiagnostic
	if errors.As(urlErr.Err, &safeDiagnostic) && safeDiagnostic != nil {
		cause = safeDiagnostic
	} else if errors.Is(urlErr.Err, context.Canceled) {
		cause = context.Canceled
	} else if errors.Is(urlErr.Err, context.DeadlineExceeded) {
		cause = context.DeadlineExceeded
	} else {
		var netErr net.Error
		if errors.As(urlErr.Err, &netErr) && netErr.Timeout() {
			cause = webfetchRequestTimeoutError{}
		}
	}
	return &url.Error{Op: urlErr.Op, URL: safeURL, Err: cause}
}

// webfetchSafeRequestDiagnostic marks errors built exclusively from
// destination-validation state controlled by this package. net/http may place
// an untrusted Location header verbatim in a nested *url.Error; only marked
// diagnostics are safe to expose after Client.Do returns.
type webfetchSafeRequestDiagnostic struct {
	err error
}

type webfetchRequestTimeoutError struct{}

func (webfetchRequestTimeoutError) Error() string   { return "network request timed out" }
func (webfetchRequestTimeoutError) Timeout() bool   { return true }
func (webfetchRequestTimeoutError) Temporary() bool { return true }

func newWebfetchSafeRequestDiagnostic(err error) *webfetchSafeRequestDiagnostic {
	return &webfetchSafeRequestDiagnostic{err: err}
}

func (e *webfetchSafeRequestDiagnostic) Error() string {
	if e == nil || e.err == nil {
		return "HTTP request failed"
	}
	return e.err.Error()
}

func (e *webfetchSafeRequestDiagnostic) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.err
}

// validateWebfetchDestinationURL validates that a URL destination is safe.
func validateWebfetchDestinationURL(ctx context.Context, target *url.URL, stage string) error {
	if target == nil {
		return fmt.Errorf("invalid url: missing host")
	}
	if target.User != nil {
		return fmt.Errorf("%s must not include embedded URL credentials", stage)
	}
	host := strings.TrimSpace(target.Hostname())
	if host == "" {
		return fmt.Errorf("invalid url: missing host")
	}
	return validateWebfetchHostDestination(ctx, host, stage)
}

// validateWebfetchHostDestination validates that a host destination is safe.
func validateWebfetchHostDestination(ctx context.Context, host, stage string) error {
	_, err := resolveAndValidateWebfetchHost(ctx, host, stage)
	return err
}

func resolveAndValidateWebfetchHost(ctx context.Context, host, stage string) ([]net.IP, error) {
	host = strings.TrimSpace(host)
	if host == "" {
		return nil, fmt.Errorf("invalid url: missing host")
	}
	if stage == "" {
		stage = "request target"
	}
	if ip := parseWebfetchLiteralIP(host); ip != nil {
		if class := classifyWebfetchAddress(ip); class != "" {
			return nil, webfetchDestinationDeniedError(stage, host, ip.String(), class)
		}
		return []net.IP{append(net.IP(nil), ip...)}, nil
	}
	addrs, err := webfetchLookupIPAddrs(ctx, host)
	if err != nil {
		return nil, fmt.Errorf("cannot resolve %s %q: %v. Check hostname spelling and retry with a public URL", stage, host, err)
	}
	if len(addrs) == 0 {
		return nil, fmt.Errorf("cannot resolve %s %q: no addresses returned. Check hostname spelling and retry with a public URL", stage, host)
	}
	validated := make([]net.IP, 0, len(addrs))
	for _, addr := range addrs {
		if addr.IP == nil {
			continue
		}
		if class := classifyWebfetchAddress(addr.IP); class != "" {
			return nil, webfetchDestinationDeniedError(stage, host, addr.IP.String(), class)
		}
		validated = append(validated, append(net.IP(nil), addr.IP...))
	}
	if len(validated) == 0 {
		return nil, fmt.Errorf("cannot resolve %s %q: no usable addresses returned. Check hostname spelling and retry with a public URL", stage, host)
	}
	return validated, nil
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

// WebfetchAddressPolicy captures the host-configurable part of webfetch
// destination filtering. The zero value is the fail-closed policy: every
// non-public destination class is denied.
//
// It exists so hosts embedding this SDK share one destination classifier
// instead of maintaining their own copy that can drift out of sync.
type WebfetchAddressPolicy struct {
	// AllowPrivateIPv4 permits RFC1918 IPv4 destinations (10/8, 172.16/12,
	// 192.168/16). Only hosts that intentionally target local development
	// services should opt in; the zero value denies them.
	AllowPrivateIPv4 bool
}

// webfetchRestrictedAddrRules denies IANA special-purpose ranges that are
// neither loopback, link-local nor private but must never be a webfetch
// destination.
var webfetchRestrictedAddrRules = []struct {
	prefix netip.Prefix
	class  string
}{
	{prefix: mustParseWebfetchPrefix("0.0.0.0/8"), class: "this-network"},
	{prefix: mustParseWebfetchPrefix("100.64.0.0/10"), class: "carrier-grade nat"},
	{prefix: mustParseWebfetchPrefix("192.0.0.0/24"), class: "iana special-purpose"},
	{prefix: mustParseWebfetchPrefix("192.0.2.0/24"), class: "documentation test-net-1"},
	{prefix: mustParseWebfetchPrefix("198.18.0.0/15"), class: "benchmark testing"},
	{prefix: mustParseWebfetchPrefix("198.51.100.0/24"), class: "documentation test-net-2"},
	{prefix: mustParseWebfetchPrefix("203.0.113.0/24"), class: "documentation test-net-3"},
	{prefix: mustParseWebfetchPrefix("240.0.0.0/4"), class: "reserved"},
}

// webfetchMetadataAddrs are cloud instance-metadata endpoints. They are denied
// independently of WebfetchAddressPolicy because some of them live inside
// otherwise routable ranges, and because a host that allows private IPv4 must
// still not be able to read instance credentials.
var webfetchMetadataAddrs = []netip.Addr{
	netip.MustParseAddr("169.254.169.254"), // AWS / GCP / Azure / OpenStack IMDS
	netip.MustParseAddr("169.254.170.2"),   // AWS ECS task metadata
	netip.MustParseAddr("168.63.129.16"),   // Azure platform (WireServer)
	netip.MustParseAddr("100.100.100.200"), // Alibaba Cloud metadata
	netip.MustParseAddr("192.0.0.192"),     // Oracle Cloud metadata
	netip.MustParseAddr("fd00:ec2::254"),   // AWS IMDS over IPv6
}

// mustParseWebfetchPrefix parses static CIDR literals for package-level rules.
// It must only be used with compile-time constants during initialization.
func mustParseWebfetchPrefix(cidr string) netip.Prefix {
	prefix, err := netip.ParsePrefix(cidr)
	if err != nil {
		panic(fmt.Sprintf("invalid webfetch prefix %q: %v", cidr, err))
	}
	return prefix
}

// webfetchNAT64WellKnownPrefix is the RFC 6052 well-known NAT64 prefix.
var webfetchNAT64WellKnownPrefix = mustParseWebfetchPrefix("64:ff9b::/96")

// webfetchNAT64LocalUsePrefix is the RFC 8215 local-use NAT64 prefix. Any /96
// inside it embeds an IPv4 address in its low 32 bits.
var webfetchNAT64LocalUsePrefix = mustParseWebfetchPrefix("64:ff9b:1::/48")

// webfetch6to4Prefix is the RFC 3056 6to4 prefix: bytes 2..5 hold the IPv4.
var webfetch6to4Prefix = mustParseWebfetchPrefix("2002::/16")

// webfetchTeredoPrefix is the RFC 4380 Teredo prefix: bytes 4..7 hold the
// server IPv4 and bytes 12..15 the client IPv4, obfuscated by XOR with 0xff.
var webfetchTeredoPrefix = mustParseWebfetchPrefix("2001::/32")

// embeddedWebfetchIPv4Addrs returns the IPv4 addresses carried inside an IPv6
// transitional address, together with a label naming the embedding form.
//
// IPv6 has several ways to tunnel an IPv4 destination. netip.Addr.Unmap only
// undoes the ::ffff: (IPv4-mapped) form, so `64:ff9b::a9fe:a9fe` — the NAT64
// spelling of the 169.254.169.254 instance-metadata endpoint — reached the
// classifier as an ordinary global-unicast IPv6 address and was allowed. Every
// embedded form must be unwrapped and the inner IPv4 classified on its own.
func embeddedWebfetchIPv4Addrs(addr netip.Addr) []struct {
	form string
	ip   netip.Addr
} {
	var out []struct {
		form string
		ip   netip.Addr
	}
	if !addr.Is6() || addr.Is4In6() {
		return out
	}
	b := addr.As16()
	add := func(form string, v4 netip.Addr) {
		if v4.IsValid() {
			out = append(out, struct {
				form string
				ip   netip.Addr
			}{form: form, ip: v4})
		}
	}
	v4From := func(o0, o1, o2, o3 byte) netip.Addr {
		return netip.AddrFrom4([4]byte{o0, o1, o2, o3})
	}
	switch {
	case webfetchNAT64WellKnownPrefix.Contains(addr):
		add("NAT64", v4From(b[12], b[13], b[14], b[15]))
	case webfetchNAT64LocalUsePrefix.Contains(addr):
		add("NAT64", v4From(b[12], b[13], b[14], b[15]))
	case webfetch6to4Prefix.Contains(addr):
		add("6to4", v4From(b[2], b[3], b[4], b[5]))
	case webfetchTeredoPrefix.Contains(addr):
		add("Teredo server", v4From(b[4], b[5], b[6], b[7]))
		add("Teredo client", v4From(b[12]^0xff, b[13]^0xff, b[14]^0xff, b[15]^0xff))
	}
	return out
}

// ClassifyWebfetchAddress is the single webfetch destination classifier shared
// by this SDK and its embedding hosts. It returns a non-empty classification
// string when the address must be denied, and "" only for addresses that are
// positively identified as acceptable public destinations — an address that
// cannot be interpreted is denied rather than allowed.
func ClassifyWebfetchAddress(ip net.IP, policy WebfetchAddressPolicy) string {
	if ip == nil {
		return "invalid address"
	}
	addr, ok := netip.AddrFromSlice(ip)
	if !ok {
		return "invalid address"
	}
	addr = addr.Unmap()
	if !addr.IsValid() {
		return "invalid address"
	}
	// Reject IPv6 forms that tunnel an IPv4 destination whose own class is
	// denied. The embedded IPv4 is classified with the same policy, so a NAT64 /
	// 6to4 / Teredo spelling of a metadata or loopback address is denied exactly
	// like its plain IPv4 spelling.
	for _, embedded := range embeddedWebfetchIPv4Addrs(addr) {
		if class := ClassifyWebfetchAddress(embedded.ip.AsSlice(), policy); class != "" {
			return embedded.form + " embedded " + class
		}
	}
	for _, metadata := range webfetchMetadataAddrs {
		if addr == metadata {
			return "cloud instance metadata"
		}
	}
	if addr.IsLoopback() {
		return "loopback"
	}
	if addr.IsLinkLocalUnicast() || addr.IsLinkLocalMulticast() {
		return "link-local"
	}
	if addr.IsPrivate() {
		if !addr.Is4() {
			return "private IPv6 ULA"
		}
		if !policy.AllowPrivateIPv4 {
			return "private RFC1918"
		}
	}
	for _, rule := range webfetchRestrictedAddrRules {
		if rule.prefix.Contains(addr) {
			return rule.class
		}
	}
	if !addr.IsGlobalUnicast() {
		return "non-global-unicast"
	}
	return ""
}

// classifyWebfetchAddress applies the fail-closed policy used by the SDK's own
// webfetch tool.
func classifyWebfetchAddress(ip net.IP) string {
	return ClassifyWebfetchAddress(ip, WebfetchAddressPolicy{})
}

// webfetchDestinationDeniedError returns an error for a denied destination.
func webfetchDestinationDeniedError(stage, host, ip, class string) error {
	if stage == "" {
		stage = "request target"
	}
	return fmt.Errorf("blocked %s %q: resolved to %s (%s). Use a public internet URL (loopback, private, link-local, cloud-metadata, and special-use destinations are denied)", stage, host, ip, class)
}
