from pathlib import Path

path = Path("sdk/tools/sandbox/sandbox_webfetch.go")
text = path.read_text()
old_var = '''var webfetchLookupIPAddrs = func(ctx context.Context, host string) ([]net.IPAddr, error) {
\treturn net.DefaultResolver.LookupIPAddr(ctx, host)
}
'''
new_var = '''var webfetchLookupIPAddrs = func(ctx context.Context, host string) ([]net.IPAddr, error) {
\treturn net.DefaultResolver.LookupIPAddr(ctx, host)
}

// webfetchDialContext is the final socket dial seam. The production path is
// always called with a literal IP selected from the just-validated DNS result.
var webfetchDialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
\treturn (&net.Dialer{}).DialContext(ctx, network, address)
}
'''
if text.count(old_var) != 1:
    raise SystemExit(f"lookup variable anchor count={text.count(old_var)}")
text = text.replace(old_var, new_var)
old_client = '''\t\thc := &http.Client{
\t\t\tTimeout: time.Duration(timeout) * time.Second,
\t\t\tCheckRedirect: func(req *http.Request, via []*http.Request) error {
\t\t\t\tif len(via) >= webfetchMaxRedirects {
\t\t\t\t\treturn fmt.Errorf("stopped after %d redirects", webfetchMaxRedirects)
\t\t\t\t}
\t\t\t\tif req == nil || req.URL == nil {
\t\t\t\t\treturn fmt.Errorf("invalid redirect target")
\t\t\t\t}
\t\t\t\treturn validateWebfetchDestinationURL(req.Context(), req.URL, "redirect target")
\t\t\t},
\t\t}
'''
new_client = '''\t\thc := newWebfetchHTTPClient(time.Duration(timeout) * time.Second)
\t\thc.CheckRedirect = func(req *http.Request, via []*http.Request) error {
\t\t\tif len(via) >= webfetchMaxRedirects {
\t\t\t\treturn fmt.Errorf("stopped after %d redirects", webfetchMaxRedirects)
\t\t\t}
\t\t\tif req == nil || req.URL == nil {
\t\t\t\treturn fmt.Errorf("invalid redirect target")
\t\t\t}
\t\t\treturn validateWebfetchDestinationURL(req.Context(), req.URL, "redirect target")
\t\t}
'''
if text.count(old_client) != 1:
    raise SystemExit(f"client anchor count={text.count(old_client)}")
text = text.replace(old_client, new_client)
anchor = '''// validateWebfetchDestinationURL validates that a URL destination is safe.
func validateWebfetchDestinationURL(ctx context.Context, target *url.URL, stage string) error {
'''
insert = '''func newWebfetchHTTPClient(timeout time.Duration) *http.Client {
\tvar transport *http.Transport
\tif base, ok := http.DefaultTransport.(*http.Transport); ok && base != nil {
\t\ttransport = base.Clone()
\t} else {
\t\ttransport = &http.Transport{ForceAttemptHTTP2: true}
\t}
\t// A proxy would resolve the target independently and defeat destination
\t// pinning. Webfetch therefore connects directly to the validated address.
\ttransport.Proxy = nil
\ttransport.DialContext = dialValidatedWebfetchDestination
\treturn &http.Client{Timeout: timeout, Transport: transport}
}

// dialValidatedWebfetchDestination resolves and classifies the exact address
// set used for this socket, then dials a selected literal IP. The original host
// remains on the request URL, preserving the HTTP Host header and TLS SNI.
func dialValidatedWebfetchDestination(ctx context.Context, network, address string) (net.Conn, error) {
\thost, port, err := net.SplitHostPort(address)
\tif err != nil {
\t\treturn nil, fmt.Errorf("invalid webfetch dial address %q: %w", address, err)
\t}
\taddrs, err := resolveAndValidateWebfetchHost(ctx, host, "socket target")
\tif err != nil {
\t\treturn nil, err
\t}
\tvar dialErrors []string
\tfor _, ip := range addrs {
\t\tconn, dialErr := webfetchDialContext(ctx, network, net.JoinHostPort(ip.String(), port))
\t\tif dialErr == nil {
\t\t\treturn conn, nil
\t\t}
\t\tdialErrors = append(dialErrors, dialErr.Error())
\t}
\treturn nil, fmt.Errorf("cannot connect to validated socket target %q: %s", host, strings.Join(dialErrors, "; "))
}

// validateWebfetchDestinationURL validates that a URL destination is safe.
func validateWebfetchDestinationURL(ctx context.Context, target *url.URL, stage string) error {
'''
if text.count(anchor) != 1:
    raise SystemExit(f"dial helper anchor count={text.count(anchor)}")
text = text.replace(anchor, insert)
old_validate = '''func validateWebfetchHostDestination(ctx context.Context, host, stage string) error {
\thost = strings.TrimSpace(host)
\tif host == "" {
\t\treturn fmt.Errorf("invalid url: missing host")
\t}
\tif stage == "" {
\t\tstage = "request target"
\t}
\tif ip := parseWebfetchLiteralIP(host); ip != nil {
\t\tif class := classifyWebfetchAddress(ip); class != "" {
\t\t\treturn webfetchDestinationDeniedError(stage, host, ip.String(), class)
\t\t}
\t\treturn nil
\t}
\taddrs, err := webfetchLookupIPAddrs(ctx, host)
\tif err != nil {
\t\treturn fmt.Errorf("cannot resolve %s %q: %v. Check hostname spelling and retry with a public URL", stage, host, err)
\t}
\tif len(addrs) == 0 {
\t\treturn fmt.Errorf("cannot resolve %s %q: no addresses returned. Check hostname spelling and retry with a public URL", stage, host)
\t}
\tusable := 0
\tfor _, addr := range addrs {
\t\tif addr.IP == nil {
\t\t\tcontinue
\t\t}
\t\tusable++
\t\tif class := classifyWebfetchAddress(addr.IP); class != "" {
\t\t\treturn webfetchDestinationDeniedError(stage, host, addr.IP.String(), class)
\t\t}
\t}
\tif usable == 0 {
\t\treturn fmt.Errorf("cannot resolve %s %q: no usable addresses returned. Check hostname spelling and retry with a public URL", stage, host)
\t}
\treturn nil
}
'''
new_validate = '''func validateWebfetchHostDestination(ctx context.Context, host, stage string) error {
\t_, err := resolveAndValidateWebfetchHost(ctx, host, stage)
\treturn err
}

func resolveAndValidateWebfetchHost(ctx context.Context, host, stage string) ([]net.IP, error) {
\thost = strings.TrimSpace(host)
\tif host == "" {
\t\treturn nil, fmt.Errorf("invalid url: missing host")
\t}
\tif stage == "" {
\t\tstage = "request target"
\t}
\tif ip := parseWebfetchLiteralIP(host); ip != nil {
\t\tif class := classifyWebfetchAddress(ip); class != "" {
\t\t\treturn nil, webfetchDestinationDeniedError(stage, host, ip.String(), class)
\t\t}
\t\treturn []net.IP{append(net.IP(nil), ip...)}, nil
\t}
\taddrs, err := webfetchLookupIPAddrs(ctx, host)
\tif err != nil {
\t\treturn nil, fmt.Errorf("cannot resolve %s %q: %v. Check hostname spelling and retry with a public URL", stage, host, err)
\t}
\tif len(addrs) == 0 {
\t\treturn nil, fmt.Errorf("cannot resolve %s %q: no addresses returned. Check hostname spelling and retry with a public URL", stage, host)
\t}
\tvalidated := make([]net.IP, 0, len(addrs))
\tfor _, addr := range addrs {
\t\tif addr.IP == nil {
\t\t\tcontinue
\t\t}
\t\tif class := classifyWebfetchAddress(addr.IP); class != "" {
\t\t\treturn nil, webfetchDestinationDeniedError(stage, host, addr.IP.String(), class)
\t\t}
\t\tvalidated = append(validated, append(net.IP(nil), addr.IP...))
\t}
\tif len(validated) == 0 {
\t\treturn nil, fmt.Errorf("cannot resolve %s %q: no usable addresses returned. Check hostname spelling and retry with a public URL", stage, host)
\t}
\treturn validated, nil
}
'''
if text.count(old_validate) != 1:
    raise SystemExit(f"host validator anchor count={text.count(old_validate)}")
path.write_text(text.replace(old_validate, new_validate))

Path("sdk/tools/sandbox/sandbox_webfetch_rebinding_test.go").write_text(r'''package sandbox

import (
	"context"
	"encoding/json"
	"net"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type allowWebfetchConfirmer struct{}

func (allowWebfetchConfirmer) Confirm(context.Context, string, string) (bool, error) {
	return true, nil
}

func TestWebfetchRejectsReboundAddressUsedForSocket(t *testing.T) {
	originalLookup := webfetchLookupIPAddrs
	originalDial := webfetchDialContext
	t.Cleanup(func() {
		webfetchLookupIPAddrs = originalLookup
		webfetchDialContext = originalDial
	})
	var lookups atomic.Int32
	webfetchLookupIPAddrs = func(context.Context, string) ([]net.IPAddr, error) {
		if lookups.Add(1) == 1 {
			return []net.IPAddr{{IP: net.ParseIP("8.8.8.8")}}, nil
		}
		return []net.IPAddr{{IP: net.ParseIP("127.0.0.1")}}, nil
	}
	var dials atomic.Int32
	webfetchDialContext = func(context.Context, string, string) (net.Conn, error) {
		dials.Add(1)
		return nil, context.Canceled
	}

	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) {
		return allowWebfetchConfirmer{}, nil
	})
	out, err := webfetchTool().Execute(context.Background(), string(marshalWebfetchRebindingJSON(t, map[string]any{"url": "http://rebind.example/"})), deps)
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "loopback") {
		t.Fatalf("webfetch error = %v, output=%q; want rebound loopback denial", err, out.PlainText())
	}
	if lookups.Load() != 2 {
		t.Fatalf("resolver calls = %d, want preflight and socket lookup", lookups.Load())
	}
	if dials.Load() != 0 {
		t.Fatalf("socket dial attempted %d time(s) after denied rebound address", dials.Load())
	}
}

func marshalWebfetchRebindingJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
''')
