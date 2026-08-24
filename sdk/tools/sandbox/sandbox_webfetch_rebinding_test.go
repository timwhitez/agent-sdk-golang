package sandbox

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
	if lookups.Load() != 1 {
		t.Fatalf("resolver calls = %d, want exactly the socket lookup", lookups.Load())
	}
	if dials.Load() != 0 {
		t.Fatalf("socket dial attempted %d time(s) after denied address", dials.Load())
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
