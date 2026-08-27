package sandbox

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestSameWebfetchOrigin(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		left  string
		right string
		want  bool
	}{
		{name: "same origin path", left: "https://example.test/start", right: "https://example.test/next", want: true},
		{name: "explicit default port", left: "https://example.test/start", right: "https://example.test:443/next", want: true},
		{name: "different host", left: "https://example.test/start", right: "https://other.test/next", want: false},
		{name: "different port", left: "https://example.test:443/start", right: "https://example.test:8443/next", want: false},
		{name: "scheme downgrade", left: "https://example.test/start", right: "http://example.test/next", want: false},
		{name: "scheme upgrade", left: "http://example.test/start", right: "https://example.test/next", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			left, err := url.Parse(tc.left)
			if err != nil {
				t.Fatal(err)
			}
			right, err := url.Parse(tc.right)
			if err != nil {
				t.Fatal(err)
			}
			if got := sameWebfetchOrigin(left, right); got != tc.want {
				t.Fatalf("sameWebfetchOrigin(%q, %q) = %v, want %v", tc.left, tc.right, got, tc.want)
			}
		})
	}
}

func TestWebfetchToolRejectsCrossOriginRedirectBeforeLookupOrCredentialForward(t *testing.T) {
	const secret = "redirect-secret-sentinel"
	useSandboxWebfetchResolver(t, func(_ context.Context, host string) ([]net.IPAddr, error) {
		if host == "other.test" {
			t.Fatalf("cross-origin redirect performed DNS lookup before new confirmation")
		}
		return []net.IPAddr{{IP: net.ParseIP("93.184.216.34")}}, nil
	})

	origDo := webfetchDoRequest
	calls := 0
	webfetchDoRequest = func(client *http.Client, initial *http.Request) (*http.Response, error) {
		calls++
		if initial.Header.Get("X-API-Key") != secret || initial.Header.Get("Authorization") != "Bearer "+secret {
			return nil, errors.New("initial credential headers missing")
		}
		redirect, err := http.NewRequestWithContext(initial.Context(), http.MethodGet, "https://other.test/next", nil)
		if err != nil {
			return nil, err
		}
		redirect.Header = initial.Header.Clone()
		if client.CheckRedirect == nil {
			return nil, errors.New("WebFetch client has no redirect policy")
		}
		return nil, client.CheckRedirect(redirect, []*http.Request{initial})
	}
	t.Cleanup(func() { webfetchDoRequest = origDo })

	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })
	result, err := webfetchTool().Execute(context.Background(), `{"url":"https://example.test/start","headers":{"X-API-Key":"`+secret+`","Authorization":"Bearer `+secret+`"}}`, deps)
	if err == nil {
		t.Fatalf("cross-origin redirect unexpectedly succeeded: %q", result.PlainText())
	}
	if calls != 1 {
		t.Fatalf("outbound call count = %d, want one", calls)
	}
	message := err.Error()
	if !strings.Contains(message, "changes origin") || !strings.Contains(message, "directly") {
		t.Fatalf("redirect error is not actionable: %q", message)
	}
	if strings.Contains(message, secret) {
		t.Fatalf("redirect error exposed credential value: %q", message)
	}
}

func TestWebfetchToolAllowsSameOriginRedirect(t *testing.T) {
	useSandboxPublicWebfetchResolver(t)
	origDo := webfetchDoRequest
	webfetchDoRequest = func(client *http.Client, initial *http.Request) (*http.Response, error) {
		redirect, err := http.NewRequestWithContext(initial.Context(), http.MethodGet, "https://example.test/next", nil)
		if err != nil {
			return nil, err
		}
		redirect.Header = initial.Header.Clone()
		if err := client.CheckRedirect(redirect, []*http.Request{initial}); err != nil {
			return nil, err
		}
		return &http.Response{
			Status:     "200 OK",
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader("ok")),
			Request:    redirect,
		}, nil
	}
	t.Cleanup(func() { webfetchDoRequest = origDo })

	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })
	result, err := webfetchTool().Execute(context.Background(), `{"url":"https://example.test/start","headers":{"X-Trace":"ok"}}`, deps)
	if err != nil {
		t.Fatalf("same-origin redirect: %v", err)
	}
	if !strings.Contains(result.PlainText(), "ok") {
		t.Fatalf("same-origin response = %q", result.PlainText())
	}
}

func TestWebfetchRejectsEmbeddedURLCredentialsWithoutEchoingThem(t *testing.T) {
	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })
	result, err := webfetchTool().Execute(context.Background(), `{"url":"https://audit-user:audit-password@example.test/private"}`, deps)
	if err == nil {
		t.Fatalf("embedded credentials unexpectedly accepted: %q", result.PlainText())
	}
	if strings.Contains(err.Error(), "audit-user") || strings.Contains(err.Error(), "audit-password") {
		t.Fatalf("embedded credential error echoed userinfo: %q", err.Error())
	}
}

func TestWebfetchRedirectErrorSanitizesCredentialBearingLocation(t *testing.T) {
	useSandboxPublicWebfetchResolver(t)
	tests := []struct {
		name       string
		location   string
		wantOrigin string
	}{
		{name: "cross origin", location: "https://audit-user:audit-password@other.test/private?token=audit-query-secret", wantOrigin: "other.test:443"},
		{name: "same origin userinfo", location: "https://audit-user:audit-password@example.test/private?token=audit-query-secret", wantOrigin: "example.test:443"},
		{name: "malformed location", location: "https://audit-user:audit-password@other.test/%zz?token=audit-query-secret", wantOrigin: "example.test:443"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			origDo := webfetchDoRequest
			webfetchDoRequest = func(client *http.Client, initial *http.Request) (*http.Response, error) {
				calls := 0
				client.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
					calls++
					if calls > 1 {
						t.Fatal("credential-bearing redirect reached a second transport request")
					}
					return &http.Response{
						Status:     "302 Found",
						StatusCode: http.StatusFound,
						Header:     http.Header{"Location": []string{tc.location}},
						Body:       io.NopCloser(strings.NewReader("")),
						Request:    request,
					}, nil
				})
				return client.Do(initial)
			}
			t.Cleanup(func() { webfetchDoRequest = origDo })

			deps := tools.NewContainer()
			tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })
			result, err := webfetchTool().Execute(context.Background(), `{"url":"https://example.test/start"}`, deps)
			if err == nil {
				t.Fatalf("credential-bearing redirect unexpectedly succeeded: %q", result.PlainText())
			}
			combined := err.Error() + "\n" + result.PlainText()
			for _, secret := range []string{"audit-user", "audit-password", "audit-query-secret", "/private"} {
				if strings.Contains(combined, secret) {
					t.Fatalf("redirect diagnostics leaked %q: %q", secret, combined)
				}
			}
			if !strings.Contains(combined, tc.wantOrigin) {
				t.Fatalf("sanitized diagnostics lost actionable origin: %q", combined)
			}
		})
	}
}

func TestSanitizeWebfetchRequestErrorPreservesTimeoutSemanticsWithoutDetails(t *testing.T) {
	const secret = "audit-timeout-secret"
	err := sanitizeWebfetchRequestError(&url.Error{
		Op:  "Get",
		URL: "https://example.test/private?token=" + secret,
		Err: secretWebfetchTimeoutError{message: secret},
	})
	var sanitized *url.Error
	if !errors.As(err, &sanitized) {
		t.Fatalf("sanitized error = %T, want *url.Error", err)
	}
	if !sanitized.Timeout() {
		t.Fatalf("sanitized timeout error lost Timeout semantics: %v", sanitized)
	}
	if strings.Contains(sanitized.Error(), secret) || strings.Contains(sanitized.Error(), "/private") {
		t.Fatalf("sanitized timeout error leaked request details: %q", sanitized.Error())
	}
}

type secretWebfetchTimeoutError struct {
	message string
}

func (e secretWebfetchTimeoutError) Error() string { return e.message }
func (secretWebfetchTimeoutError) Timeout() bool   { return true }
func (secretWebfetchTimeoutError) Temporary() bool { return true }
