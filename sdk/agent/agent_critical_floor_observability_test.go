package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

// REG (R4X-004): the critical-event floor silently replaced a host's configured
// EventSendTimeout, so a host that deliberately set 1ms or 13ms saw a 250ms wait
// with nothing in the logs explaining it. The floor itself must stay -- applying a
// delta-tuned budget to consistency-critical events is what ISS-129b prevents --
// but the substitution has to be observable rather than invisible.
func TestCriticalEventFloorOverrideOfHostConfigIsReported(t *testing.T) {
	for _, configured := range []time.Duration{time.Millisecond, 13 * time.Millisecond} {
		t.Run(configured.String(), func(t *testing.T) {
			var warnings lockedBuffer
			ag, err := New(Config{
				LLM:              &completionOnlyModel{},
				EventBufferSize:  1,
				EventSendTimeout: configured,
				Warningf: func(format string, args ...any) {
					_, _ = warnings.Write([]byte(fmt.Sprintf(format, args...) + "\n"))
				},
			})
			if err != nil {
				t.Fatalf("new agent: %v", err)
			}
			if ag.configuredEventSendTimeout() != configured {
				t.Fatalf("configured budget = %s, wanted the host value %s to survive normalization",
					ag.configuredEventSendTimeout(), configured)
			}

			out := make(chan Event, 1)
			out <- WarnEvent{Message: "filler"} // channel is now full
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			defer ag.registerTurnCancellation(out, ctx)()

			start := time.Now()
			if ag.emitEvent(out, ToolResultEvent{Tool: "read"}) {
				t.Fatal("expected the send into a full channel to fail")
			}
			// The floor still wins over the configured value: ISS-129b.
			if elapsed := time.Since(start); elapsed < criticalEventSendTimeoutFloor {
				t.Fatalf("critical event waited %v; the %v floor must not be weakened to the configured %s",
					elapsed, criticalEventSendTimeoutFloor, configured)
			}

			got := warnings.String()
			if !strings.Contains(got, "critical-event floor") {
				t.Fatalf("the floor overrode the configured EventSendTimeout without reporting it; warnings = %q", got)
			}
			if !strings.Contains(got, configured.String()) {
				t.Fatalf("the report does not name the configured budget %s; warnings = %q", configured, got)
			}
			if !strings.Contains(got, criticalEventSendTimeoutFloor.String()) {
				t.Fatalf("the report does not name the floor %s; warnings = %q", criticalEventSendTimeoutFloor, got)
			}

			// One line per turn, not one per event: a tool-heavy turn must not
			// flood the host's log with the same notice.
			for i := 0; i < 5; i++ {
				ag.emitEvent(out, ToolResultEvent{Tool: "read"})
			}
			lines := 0
			for _, line := range strings.Split(warnings.String(), "\n") {
				if strings.Contains(line, "critical-event floor") {
					lines++
				}
			}
			if lines != 1 {
				t.Fatalf("floor-override notice was emitted %d times; want exactly one per turn", lines)
			}
		})
	}
}

// REG (R4X-004): a host that configures a budget at or above the floor is not
// having anything overridden, so it must not be warned.
func TestCriticalEventFloorDoesNotReportWhenHostConfigMeetsTheFloor(t *testing.T) {
	var warnings lockedBuffer
	ag, err := New(Config{
		LLM:              &completionOnlyModel{},
		EventBufferSize:  1,
		EventSendTimeout: criticalEventSendTimeoutFloor,
		Warningf: func(format string, args ...any) {
			_, _ = warnings.Write([]byte(fmt.Sprintf(format, args...) + "\n"))
		},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	out := make(chan Event, 1)
	out <- WarnEvent{Message: "filler"}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	defer ag.registerTurnCancellation(out, ctx)()

	ag.emitEvent(out, ToolResultEvent{Tool: "read"})
	if got := warnings.String(); strings.Contains(got, "critical-event floor instead of") {
		t.Fatalf("warned about an override that did not happen; warnings = %q", got)
	}
}

// REG (R4X-004): a cancelled turn does not pay the floor, so there is no override
// to report either.
func TestCriticalEventFloorDoesNotReportForCanceledTurn(t *testing.T) {
	var warnings lockedBuffer
	ag, err := New(Config{
		LLM:              &completionOnlyModel{},
		EventBufferSize:  1,
		EventSendTimeout: time.Millisecond,
		Warningf: func(format string, args ...any) {
			_, _ = warnings.Write([]byte(fmt.Sprintf(format, args...) + "\n"))
		},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	out := make(chan Event, 1)
	out <- WarnEvent{Message: "filler"}
	ctx, cancel := context.WithCancel(context.Background())
	defer ag.registerTurnCancellation(out, ctx)()
	cancel()

	ag.emitEvent(out, ToolResultEvent{Tool: "read"})
	if got := warnings.String(); strings.Contains(got, "critical-event floor instead of") {
		t.Fatalf("a canceled turn reported a floor override it never paid; warnings = %q", got)
	}
}
