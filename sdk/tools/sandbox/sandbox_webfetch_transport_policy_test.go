package sandbox

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
