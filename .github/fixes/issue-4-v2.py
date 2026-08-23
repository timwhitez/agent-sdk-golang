from pathlib import Path

path = Path("sdk/tools/sandbox/sandbox_webfetch.go")
text = path.read_text(encoding="utf-8")
old = '''\t\tif err := validateWebfetchDestinationURL(ctx, u, "request target"); err != nil {
\t\t\treturn "", err
\t\t}
'''
new = '''\t\t// Before the user authorizes network access, validate only syntax and
\t\t// literal-IP policy. Hostname resolution is itself an observable network
\t\t// action and is deferred to the socket-bound validator after confirmation.
\t\tif err := validateWebfetchPreConfirmationURL(u, "request target"); err != nil {
\t\t\treturn "", err
\t\t}
'''
if text.count(old) != 1:
    raise SystemExit(f"pre-confirm validation anchor count={text.count(old)}")
text = text.replace(old, new, 1)
anchor = '''// validateWebfetchDestinationURL validates that a URL destination is safe.
func validateWebfetchDestinationURL(ctx context.Context, target *url.URL, stage string) error {
'''
insert = '''// validateWebfetchPreConfirmationURL performs only local checks. A hostname is
// intentionally not resolved until after the user has approved the request.
func validateWebfetchPreConfirmationURL(target *url.URL, stage string) error {
\tif target == nil {
\t\treturn fmt.Errorf("invalid url: missing host")
\t}
\thost := strings.TrimSpace(target.Hostname())
\tif host == "" {
\t\treturn fmt.Errorf("invalid url: missing host")
\t}
\tif ip := parseWebfetchLiteralIP(host); ip != nil {
\t\tif class := classifyWebfetchAddress(ip); class != "" {
\t\t\treturn webfetchDestinationDeniedError(stage, host, ip.String(), class)
\t\t}
\t}
\treturn nil
}

// validateWebfetchDestinationURL validates that a URL destination is safe.
func validateWebfetchDestinationURL(ctx context.Context, target *url.URL, stage string) error {
'''
if text.count(anchor) != 1:
    raise SystemExit(f"helper anchor count={text.count(anchor)}")
path.write_text(text.replace(anchor, insert, 1), encoding="utf-8")

Path("sdk/tools/sandbox/sandbox_webfetch_confirmation_test.go").write_text(r'''package sandbox

import (
	"context"
	"net"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type denyWebfetchConfirmer struct{ calls atomic.Int32 }

func (c *denyWebfetchConfirmer) Confirm(context.Context, string, string) (bool, error) {
	c.calls.Add(1)
	return false, nil
}

func TestWebfetchDenialPerformsNoDNSOrDial(t *testing.T) {
	originalLookup := webfetchLookupIPAddrs
	originalDial := webfetchDialContext
	t.Cleanup(func() { webfetchLookupIPAddrs = originalLookup; webfetchDialContext = originalDial })
	var lookups atomic.Int32
	webfetchLookupIPAddrs = func(context.Context, string) ([]net.IPAddr, error) {
		lookups.Add(1)
		return []net.IPAddr{{IP: net.ParseIP("8.8.8.8")}}, nil
	}
	var dials atomic.Int32
	webfetchDialContext = func(context.Context, string, string) (net.Conn, error) {
		dials.Add(1)
		return nil, context.Canceled
	}
	confirmer := &denyWebfetchConfirmer{}
	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return confirmer, nil })

	out, err := webfetchTool().Execute(context.Background(), `{"url":"https://denied.example/private"}`, deps)
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "denied") {
		t.Fatalf("denied webfetch error = %v, output=%q", err, out.PlainText())
	}
	if confirmer.calls.Load() != 1 { t.Fatalf("confirmation calls = %d, want 1", confirmer.calls.Load()) }
	if lookups.Load() != 0 { t.Fatalf("DNS lookups before/after denial = %d, want 0", lookups.Load()) }
	if dials.Load() != 0 { t.Fatalf("socket dials before/after denial = %d, want 0", dials.Load()) }
}

func TestWebfetchBlockedLiteralIsRejectedWithoutPrompt(t *testing.T) {
	confirmer := &denyWebfetchConfirmer{}
	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return confirmer, nil })
	_, err := webfetchTool().Execute(context.Background(), `{"url":"http://127.0.0.1/"}`, deps)
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "loopback") { t.Fatalf("literal loopback error = %v", err) }
	if confirmer.calls.Load() != 0 { t.Fatalf("blocked literal prompted %d time(s)", confirmer.calls.Load()) }
}
''', encoding="utf-8")

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
func (allowWebfetchConfirmer) Confirm(context.Context, string, string) (bool, error) { return true, nil }

func TestWebfetchRejectsReboundAddressUsedForSocket(t *testing.T) {
	originalLookup := webfetchLookupIPAddrs
	originalDial := webfetchDialContext
	t.Cleanup(func() { webfetchLookupIPAddrs = originalLookup; webfetchDialContext = originalDial })
	var lookups atomic.Int32
	webfetchLookupIPAddrs = func(context.Context, string) ([]net.IPAddr, error) {
		lookups.Add(1)
		return []net.IPAddr{{IP: net.ParseIP("127.0.0.1")}}, nil
	}
	var dials atomic.Int32
	webfetchDialContext = func(context.Context, string, string) (net.Conn, error) { dials.Add(1); return nil, context.Canceled }
	deps := tools.NewContainer()
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return allowWebfetchConfirmer{}, nil })
	out, err := webfetchTool().Execute(context.Background(), string(marshalWebfetchRebindingJSON(t, map[string]any{"url": "http://rebind.example/"})), deps)
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "loopback") {
		t.Fatalf("webfetch error = %v, output=%q; want socket-bound loopback denial", err, out.PlainText())
	}
	if lookups.Load() != 1 { t.Fatalf("resolver calls = %d, want exactly the socket lookup", lookups.Load()) }
	if dials.Load() != 0 { t.Fatalf("socket dial attempted %d time(s) after denied address", dials.Load()) }
}

func marshalWebfetchRebindingJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil { t.Fatal(err) }
	return data
}
''', encoding="utf-8")
