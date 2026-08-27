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
