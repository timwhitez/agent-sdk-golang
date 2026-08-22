from pathlib import Path

Path("sdk/tools/sandbox/sandbox_timeout.go").write_text(r'''package sandbox

import (
	"fmt"
	"math"
	"time"
)

const maxSandboxTimeoutSeconds = 24 * 60 * 60

func checkedSandboxTimeout(rawSeconds, defaultSeconds int) (int, time.Duration, error) {
	seconds := rawSeconds
	if seconds <= 0 {
		seconds = defaultSeconds
	}
	if seconds <= 0 {
		return 0, 0, fmt.Errorf("timeout must resolve to a positive number of seconds")
	}
	if int64(seconds) > math.MaxInt64/int64(time.Second) {
		return 0, 0, fmt.Errorf("timeout %d seconds exceeds time.Duration range", seconds)
	}
	if seconds > maxSandboxTimeoutSeconds {
		return 0, 0, fmt.Errorf("timeout %d seconds exceeds the maximum allowed %d seconds", seconds, maxSandboxTimeoutSeconds)
	}
	duration := time.Duration(seconds) * time.Second
	if duration <= 0 {
		return 0, 0, fmt.Errorf("timeout %d seconds overflowed time.Duration", seconds)
	}
	return seconds, duration, nil
}
''')

bash = Path("sdk/tools/sandbox/sandbox_bash.go")
text = bash.read_text()
text = text.replace('\t"strings"\n\t"time"\n', '\t"strings"\n')
old = '''\t\tcmd0 := strings.TrimSpace(a.Command)
\t\tmeta := attachToolCallMeta(ctx, map[string]any{
'''
new = '''\t\tcmd0 := strings.TrimSpace(a.Command)
\t\ttimeout, timeoutDuration, err := checkedSandboxTimeout(a.Timeout, 30)
\t\tif err != nil {
\t\t\tmsg := formatErrorDiagnosticFromErr("Invalid bash timeout", err, fmt.Sprintf("Use a timeout from 1 to %d seconds and retry.", maxSandboxTimeoutSeconds))
\t\t\treturn llm.TextContent(msg), err
\t\t}
\t\tmeta := attachToolCallMeta(ctx, map[string]any{
'''
if text.count(old) != 1:
    raise SystemExit(f"bash timeout insertion anchor count={text.count(old)}")
text = text.replace(old, new)
old = '''\t\ttimeout := a.Timeout
\t\tif timeout <= 0 {
\t\t\ttimeout = 30
\t\t}
\t\tresolvedExecDir, err := s.RevalidateAccessPath(workdirAccessPath)
'''
new = '''\t\tresolvedExecDir, err := s.RevalidateAccessPath(workdirAccessPath)
'''
if text.count(old) != 1:
    raise SystemExit(f"bash old timeout block count={text.count(old)}")
text = text.replace(old, new)
text = text.replace('Timeout:        time.Duration(timeout) * time.Second,', 'Timeout:        timeoutDuration,')
bash.write_text(text)

web = Path("sdk/tools/sandbox/sandbox_webfetch.go")
text = web.read_text()
old = '''\t\ttimeout := a.Timeout
\t\tif timeout <= 0 {
\t\t\ttimeout = 30
\t\t}
\t\tmaxBytes := a.MaxBytes
'''
new = '''\t\ttimeout, timeoutDuration, err := checkedSandboxTimeout(a.Timeout, 30)
\t\tif err != nil {
\t\t\treturn "", fmt.Errorf("invalid webfetch timeout: %w; use a timeout from 1 to %d seconds", err, maxSandboxTimeoutSeconds)
\t\t}
\t\tmaxBytes := a.MaxBytes
'''
if text.count(old) != 1:
    raise SystemExit(f"webfetch timeout block count={text.count(old)}")
text = text.replace(old, new)
text = text.replace('Timeout: time.Duration(timeout) * time.Second,', 'Timeout: timeoutDuration,')
web.write_text(text)

Path("sdk/tools/sandbox/sandbox_timeout_test.go").write_text(r'''package sandbox

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
	confirmer := &countingTimeoutConfirmer{}
	deps := timeoutTestDeps(t, confirmer)
	args, _ := json.Marshal(map[string]any{"url": "https://example.com", "timeout": maxSandboxTimeoutSeconds + 1})
	_, err := webfetchTool().Execute(context.Background(), string(args), deps)
	if err == nil || !strings.Contains(err.Error(), "maximum allowed") {
		t.Fatalf("webfetch oversized timeout error = %v", err)
	}
	if confirmer.calls.Load() != 0 {
		t.Fatalf("webfetch requested confirmation %d time(s) before rejecting timeout", confirmer.calls.Load())
	}
}
''')
