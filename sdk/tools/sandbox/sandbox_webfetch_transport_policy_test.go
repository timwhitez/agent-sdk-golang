package sandbox

import (
	"context"
	"crypto/tls"
	"net"
	"net/http"
	"net/url"
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

func TestWebfetchDoesNotInheritDefaultTransportTLSDialBypass(t *testing.T) {
	originalTransport := http.DefaultTransport
	hostTransport := &http.Transport{
		Proxy: func(*http.Request) (*url.URL, error) {
			return url.Parse("http://127.0.0.1:8080")
		},
		DialTLS: func(string, string) (net.Conn, error) {
			return nil, context.Canceled
		},
		DialTLSContext: func(context.Context, string, string) (net.Conn, error) {
			return nil, context.Canceled
		},
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, // test-only unsafe host policy
	}
	http.DefaultTransport = hostTransport
	t.Cleanup(func() { http.DefaultTransport = originalTransport })

	client := newWebfetchHTTPClient(0)
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("WebFetch transport = %T, want controlled *http.Transport", client.Transport)
	}
	if transport == hostTransport {
		t.Fatal("WebFetch reused the process-global transport")
	}
	if transport.Proxy != nil || transport.DialTLS != nil || transport.DialTLSContext != nil {
		t.Fatal("WebFetch inherited a proxy or TLS dial hook that can bypass validated DialContext")
	}
	if transport.TLSClientConfig != nil {
		t.Fatal("WebFetch inherited process-global TLS policy")
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
