from pathlib import Path
import re

source = Path("sdk/tools/sandbox/sandbox_webfetch.go")
text = source.read_text()
old_vars = '''var webfetchDialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
\treturn (&net.Dialer{}).DialContext(ctx, network, address)
}
'''
new_vars = '''var webfetchDialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
\treturn (&net.Dialer{}).DialContext(ctx, network, address)
}

// webfetchDoRequest is the HTTP execution seam used by package tests. The
// production value always executes the request with the validated client built
// below; unlike replacing http.DefaultTransport, overriding this seam cannot
// silently alter the transport policy in a deployed process.
var webfetchDoRequest = func(client *http.Client, request *http.Request) (*http.Response, error) {
\treturn client.Do(request)
}
'''
if text.count(old_vars) != 1:
    raise SystemExit(f"request seam anchor count={text.count(old_vars)}")
text = text.replace(old_vars, new_vars)
text = text.replace('resp, err := hc.Do(req)', 'resp, err := webfetchDoRequest(hc, req)', 1)
old_factory = '''func newWebfetchHTTPClient(timeout time.Duration) *http.Client {
\tif base, ok := http.DefaultTransport.(*http.Transport); ok && base != nil {
\t\ttransport := base.Clone()
\t\t// A proxy would resolve the target independently and defeat destination
\t\t// pinning. Webfetch therefore connects directly to the validated address.
\t\ttransport.Proxy = nil
\t\ttransport.DialContext = dialValidatedWebfetchDestination
\t\treturn &http.Client{Timeout: timeout, Transport: transport}
\t}
\t// A non-*http.Transport value is an explicit host/test replacement that may
\t// implement its own connection policy. Preserve that injection seam instead
\t// of silently bypassing it with a new default transport.
\treturn &http.Client{Timeout: timeout, Transport: http.DefaultTransport}
}
'''
new_factory = '''func newWebfetchHTTPClient(timeout time.Duration) *http.Client {
\tvar transport *http.Transport
\tif base, ok := http.DefaultTransport.(*http.Transport); ok && base != nil {
\t\ttransport = base.Clone()
\t} else {
\t\t// Do not inherit a process-global custom RoundTripper: it could resolve or
\t\t// proxy the hostname independently and bypass socket-bound validation.
\t\ttransport = &http.Transport{
\t\t\tForceAttemptHTTP2:     true,
\t\t\tMaxIdleConns:          100,
\t\t\tIdleConnTimeout:       90 * time.Second,
\t\t\tTLSHandshakeTimeout:   10 * time.Second,
\t\t\tExpectContinueTimeout: time.Second,
\t\t}
\t}
\t// A proxy would resolve the target independently and defeat destination
\t// pinning. Webfetch therefore connects directly to the validated address.
\ttransport.Proxy = nil
\ttransport.DialContext = dialValidatedWebfetchDestination
\treturn &http.Client{Timeout: timeout, Transport: transport}
}
'''
if text.count(old_factory) != 1:
    raise SystemExit(f"safe transport anchor count={text.count(old_factory)}")
source.write_text(text.replace(old_factory, new_factory))

tests = Path("sdk/tools/sandbox/sandbox_test.go")
test_text = tests.read_text()
pattern = re.compile(
    r'\torigTransport := http\.DefaultTransport\n'
    r'\thttp\.DefaultTransport = roundTripFunc\((func\([^\n]*\) \(\*http\.Response, error\) \{.*?\n\t\})\)\n'
    r'\tt\.Cleanup\(func\(\) \{ http\.DefaultTransport = origTransport \}\)',
    re.S,
)

def replace_transport(match: re.Match[str]) -> str:
    fn = match.group(1)
    fn = fn.replace('func(r *http.Request)', 'func(_ *http.Client, r *http.Request)', 1)
    fn = fn.replace('func(*http.Request)', 'func(_ *http.Client, _ *http.Request)', 1)
    if '*http.Client' not in fn.split('\n', 1)[0]:
        raise SystemExit(f"unsupported webfetch test transport signature: {fn.splitlines()[0]}")
    return '\torigDo := webfetchDoRequest\n\twebfetchDoRequest = ' + fn + '\n\tt.Cleanup(func() { webfetchDoRequest = origDo })'

updated, count = pattern.subn(replace_transport, test_text)
if count != 4:
    raise SystemExit(f"expected 4 WebFetch transport fixtures, replaced {count}")
tests.write_text(updated)

review_test = Path("sdk/tools/sandbox/sandbox_webfetch_transport_policy_test.go")
review_test.write_text(r'''package sandbox

import (
	"context"
	"net"
	"net/http"
	"testing"
)

type unsafeWebfetchRoundTripper struct{}

func (unsafeWebfetchRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, context.Canceled
}

func TestWebfetchDoesNotInheritCustomDefaultRoundTripper(t *testing.T) {
	originalTransport := http.DefaultTransport
	http.DefaultTransport = unsafeWebfetchRoundTripper{}
	t.Cleanup(func() { http.DefaultTransport = originalTransport })

	client := newWebfetchHTTPClient(0)
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("WebFetch transport = %T, want controlled *http.Transport", client.Transport)
	}
	if transport.Proxy != nil {
		t.Fatal("WebFetch controlled transport retained proxy resolution")
	}
	if transport.DialContext == nil {
		t.Fatal("WebFetch controlled transport has no validated DialContext")
	}
}

func TestValidatedWebfetchDialUsesLiteralApprovedAddress(t *testing.T) {
	originalLookup := webfetchLookupIPAddrs
	originalDial := webfetchDialContext
	t.Cleanup(func() {
		webfetchLookupIPAddrs = originalLookup
		webfetchDialContext = originalDial
	})
	webfetchLookupIPAddrs = func(context.Context, string) ([]net.IPAddr, error) {
		return []net.IPAddr{{IP: net.ParseIP("8.8.8.8")}}, nil
	}
	var dialed string
	webfetchDialContext = func(_ context.Context, _, address string) (net.Conn, error) {
		dialed = address
		return nil, context.Canceled
	}
	_, _ = dialValidatedWebfetchDestination(context.Background(), "tcp", "example.test:443")
	if dialed != "8.8.8.8:443" {
		t.Fatalf("dialed address = %q, want approved literal IP", dialed)
	}
}
''')
