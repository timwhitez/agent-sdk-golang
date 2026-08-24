package sandbox

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestCheckedSandboxTimeoutBoundaries(t *testing.T) {
	seconds, duration, err := checkedSandboxTimeout(maxSandboxTimeoutSeconds, 30)
	if err != nil {
		t.Fatalf("maximum timeout rejected: %v", err)
	}
	if seconds != maxSandboxTimeoutSeconds || duration != time.Duration(maxSandboxTimeoutSeconds)*time.Second {
		t.Fatalf("resolved timeout = %d/%s", seconds, duration)
	}
	if _, _, err := checkedSandboxTimeout(maxSandboxTimeoutSeconds+1, 30); err == nil {
		t.Fatal("timeout above practical maximum was accepted")
	}
	overflow := int64(math.MaxInt64/int64(time.Second)) + 1
	if int64(int(overflow)) == overflow {
		if _, _, err := checkedSandboxTimeout(int(overflow), 30); err == nil || !strings.Contains(err.Error(), "time.Duration range") {
			t.Fatalf("overflow timeout error = %v", err)
		}
	}
}

type countingTimeoutConfirmer struct{ calls atomic.Int32 }

func (c *countingTimeoutConfirmer) Confirm(context.Context, string, string) (bool, error) {
	c.calls.Add(1)
	return true, nil
}

func timeoutTestDeps(t *testing.T, confirmer Confirmer) *tools.Container {
	t.Helper()
	sandbox, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return sandbox, nil })
	tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return confirmer, nil })
	return deps
}

func TestBashRejectsOversizedTimeoutBeforeConfirmation(t *testing.T) {
	confirmer := &countingTimeoutConfirmer{}
	deps := timeoutTestDeps(t, confirmer)
	args, _ := json.Marshal(map[string]any{"command": "echo should-not-run", "timeout": maxSandboxTimeoutSeconds + 1})
	_, err := bashTool().Execute(context.Background(), string(args), deps)
	if err == nil || !strings.Contains(err.Error(), "maximum allowed") {
		t.Fatalf("bash oversized timeout error = %v", err)
	}
	if confirmer.calls.Load() != 0 {
		t.Fatalf("bash requested confirmation %d time(s) before rejecting timeout", confirmer.calls.Load())
	}
}

func TestWebfetchRejectsOversizedTimeoutBeforeConfirmation(t *testing.T) {
	useSandboxPublicWebfetchResolver(t)
	confirmer := &countingTimeoutConfirmer{}
	deps := timeoutTestDeps(t, confirmer)
	args, _ := json.Marshal(map[string]any{"url": "https://example.test", "timeout": maxSandboxTimeoutSeconds + 1})
	_, err := webfetchTool().Execute(context.Background(), string(args), deps)
	if err == nil || !strings.Contains(err.Error(), "maximum allowed") {
		t.Fatalf("webfetch oversized timeout error = %v", err)
	}
	if confirmer.calls.Load() != 0 {
		t.Fatalf("webfetch requested confirmation %d time(s) before rejecting timeout", confirmer.calls.Load())
	}
}
