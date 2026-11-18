package sandbox

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type recordingWebfetchConfirmer struct{ details []string }

func (c *recordingWebfetchConfirmer) Confirm(_ context.Context, _ string, detail string) (bool, error) {
	c.details = append(c.details, detail)
	return true, nil
}

// Issue #243: every header sent must be visible in the confirmation, with
// credential values redacted and force_confirm set only for credential headers.
func TestWebfetchConfirmationShowsHeaders(t *testing.T) {
	cases := []struct {
		name       string
		headers    string
		wantNames  []string
		wantAbsent []string
		force      bool
	}{
		{"custom header", `{"X-Exfil":"secret-from-workspace","Accept":"text/html"}`, []string{"Accept", "X-Exfil"}, nil, false},
		{"authorization", `{"authorization":"Bearer topsecret"}`, []string{"Authorization"}, []string{"topsecret"}, true},
		{"cookie and api key", `{"Cookie":"sid=abc123","x-api-key":"k-987"}`, []string{"Cookie", "X-Api-Key"}, []string{"abc123", "k-987"}, true},
		{"proxy authorization", `{"Proxy-Authorization":"Basic zzz"}`, []string{"Proxy-Authorization"}, []string{"zzz"}, true},
		{"no headers", `{}`, nil, nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var sent *http.Request
			orig := webfetchDoRequest
			webfetchDoRequest = func(_ *http.Client, req *http.Request) (*http.Response, error) {
				sent = req
				return &http.Response{StatusCode: 200, Status: "200 OK", Body: io.NopCloser(strings.NewReader("ok")), Header: http.Header{}}, nil
			}
			t.Cleanup(func() { webfetchDoRequest = orig })
			conf := &recordingWebfetchConfirmer{}
			deps := tools.NewContainer()
			tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return conf, nil })
			if _, err := webfetchTool().Execute(context.Background(), `{"url":"https://example.com/x","headers":`+tc.headers+`}`, deps); err != nil {
				t.Fatal(err)
			}
			if len(conf.details) != 1 {
				t.Fatalf("confirmations = %d", len(conf.details))
			}
			detail := conf.details[0]
			var meta map[string]any
			if err := json.Unmarshal([]byte(detail), &meta); err != nil {
				t.Fatal(err)
			}
			for _, n := range tc.wantNames {
				if !strings.Contains(detail, n) || sent.Header.Get(n) == "" {
					t.Errorf("header %s not shown (%s) or not sent", n, detail)
				}
			}
			for _, v := range tc.wantAbsent {
				if strings.Contains(detail, v) {
					t.Errorf("credential value %q leaked into confirmation: %s", v, detail)
				}
			}
			if force, _ := meta["force_confirm"].(bool); force != tc.force {
				t.Errorf("force_confirm = %v, want %v", force, tc.force)
			}
			if len(tc.wantNames) == 0 && strings.Contains(detail, "custom header") {
				t.Errorf("header-less request mentions headers: %s", detail)
			}
		})
	}
}

// Issue #251: per-call transports must not leave idle keep-alive connections.
func TestWebfetchDoesNotLeakIdleConnections(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "ok") }))
	defer srv.Close()
	origLookup, origDial := webfetchLookupIPAddrs, webfetchDialContext
	webfetchLookupIPAddrs = func(context.Context, string) ([]net.IPAddr, error) {
		return []net.IPAddr{{IP: net.ParseIP("93.184.216.34")}}, nil
	}
	webfetchDialContext = func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, srv.Listener.Addr().String())
	}
	t.Cleanup(func() { webfetchLookupIPAddrs, webfetchDialContext = origLookup, origDial })
	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return &recordingWebfetchConfirmer{}, nil })
	before := runtime.NumGoroutine()
	const calls = 20
	for i := 0; i < calls; i++ {
		if _, err := webfetchTool().Execute(context.Background(), `{"url":"http://public.example/"}`, deps); err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(200 * time.Millisecond)
	if extra := runtime.NumGoroutine() - before; extra >= calls {
		t.Fatalf("%d webfetch calls left %d extra goroutines", calls, extra)
	}
}

// Issue #267: credential headers are matched by rule, not by a short list.
func TestDescribeWebfetchHeadersCredentialRule(t *testing.T) {
	cases := []struct {
		name string
		cred bool
	}{
		{"Authorization", true}, {"Cookie", true}, {"Proxy-Authorization", true}, {"Proxy-Foo", true},
		{"X-Api-Key", true}, {"X-Goog-Api-Key", true}, {"Api-Key", true}, {"X-ApiKey", true},
		{"X-Auth-Token", true}, {"Private-Token", true}, {"X-Amz-Security-Token", true},
		{"Ocp-Apim-Subscription-Key", true}, {"X-Access-Key", true}, {"X-Client-Secret", true},
		{"X-Password", true}, {"x-auth-TOKEN", true},
		// A CSRF token is a session secret, so it is deliberately treated as a credential.
		{"X-CSRF-Token", true},
		{"User-Agent", false}, {"Accept", false}, {"Content-Type", false}, {"X-Request-Id", false},
		{"Accept-Language", false}, {"Cache-Control", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, preview, cred := describeWebfetchHeaders(map[string]string{tc.name: "value-123"})
			if cred != tc.cred {
				t.Fatalf("credential = %v, want %v", cred, tc.cred)
			}
			if shown := strings.Contains(preview[0], "value-123"); shown == tc.cred {
				t.Fatalf("preview %q: value shown = %v, credential = %v", preview[0], shown, tc.cred)
			}
		})
	}
}

// Truncation of a long header value must cut on a rune boundary.
func TestDescribeWebfetchHeadersTruncatesOnRuneBoundary(t *testing.T) {
	// 199 ASCII bytes then 3-byte runes: byte 197 falls inside the first rune.
	long := strings.Repeat("a", 199) + strings.Repeat("世", 10)
	_, preview, _ := describeWebfetchHeaders(map[string]string{"X-Note": long})
	got := strings.TrimPrefix(preview[0], "X-Note: ")
	if !utf8.ValidString(got) || !strings.HasSuffix(got, "...") {
		t.Fatalf("preview not valid truncated UTF-8: %q", got)
	}
	if want := strings.Repeat("a", 197) + "..."; got != want {
		t.Fatalf("got %d bytes, want %q", len(got), want)
	}
	short := strings.Repeat("世", 66) // 198 bytes, under the limit
	if _, p, _ := describeWebfetchHeaders(map[string]string{"X-Note": short}); p[0] != "X-Note: "+short {
		t.Fatalf("short value altered: %q", p[0])
	}
}
