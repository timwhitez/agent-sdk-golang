package sandbox

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestNonStrictWebfetchPreservesArbitraryHeaders(t *testing.T) {
	useSandboxPublicWebfetchResolver(t)
	origDo := webfetchDoRequest
	t.Cleanup(func() { webfetchDoRequest = origDo })
	calls := 0
	webfetchDoRequest = func(_ *http.Client, r *http.Request) (*http.Response, error) {
		calls++
		if r.Header.Get("X-Fixture-First") != "one" || r.Header.Get("X-Fixture-Second") != "two" {
			t.Fatalf("arbitrary headers = %#v", r.Header)
		}
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("fixture")), Request: r}, nil
	}
	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowConfirmer{}, nil })
	var fetch tools.Tool
	for _, tool := range Tools() {
		if tool.Name == "webfetch" {
			fetch = tool
		}
	}
	if fetch.Definition().Strict {
		t.Fatal("webfetch should use non-strict schema")
	}
	result, err := fetch.Execute(context.Background(), `{"url":"https://example.test/","headers":{"X-Fixture-First":"one","X-Fixture-Second":"two"}}`, deps)
	if err != nil || calls != 1 || !strings.Contains(result.PlainText(), "fixture") {
		t.Fatalf("webfetch: %q, %v; calls=%d", result.PlainText(), err, calls)
	}
}
