package agent

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// #242: every Tool Call announced with StepStart+ToolCall gets exactly one
// ToolResultEvent and one StepCompleteEvent, wherever root cancellation lands.
// The cancellation point walks every published event of the turn (EventClock
// runs synchronously once per event), so the window between Admit's
// announcement and the root gate before markRunning is hit deterministically.
func TestAnnouncedToolCallAlwaysGetsTerminalEvents(t *testing.T) {
	for _, tc := range []struct {
		name     string
		parallel *ToolParallelism
	}{
		{name: "sequential"},
		{name: "wave", parallel: &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			announcedNotRun := 0
			for k := int32(1); ; k++ {
				if k > 64 {
					t.Fatal("turn never completed without cancellation")
				}
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				var ticks atomic.Int32
				clock := func() time.Time {
					if ticks.Add(1) == k {
						cancel()
					}
					return time.Now()
				}
				var mu sync.Mutex
				ran := map[string]bool{}
				read := tools.Func[struct {
					FilePath string `json:"file_path"`
				}]("read", "read", func(ctx context.Context, _ struct {
					FilePath string `json:"file_path"`
				}, _ *tools.Container) (any, error) {
					mu.Lock()
					ran[tools.ToolCallID(ctx)] = true
					mu.Unlock()
					return "ok", nil
				})
				ag, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls("a", "b")}}, Tools: []tools.Tool{read},
					ToolParallelism: tc.parallel, EventClock: clock, Warningf: func(string, ...any) {}})
				if err != nil {
					t.Fatal(err)
				}
				called, results, completes := map[string]int{}, map[string]int{}, map[string]int{}
				canceled := false
				for env := range ag.QueryStreamEnveloped(ctx, llm.TextContent("go")) {
					switch e := env.Event.(type) {
					case ToolCallEvent:
						called[e.ToolCallID]++
					case ToolResultEvent:
						results[e.ToolCallID]++
						mu.Lock()
						started := ran[e.ToolCallID]
						mu.Unlock()
						if !started && (!e.IsError || e.ErrorOrigin != ToolErrorOriginCanceled) {
							t.Fatalf("k=%d: announced-but-unstarted %s result = %#v", k, e.ToolCallID, e)
						}
					case StepCompleteEvent:
						completes[e.StepID]++
					case ErrorEvent:
						canceled = canceled || e.Kind == "canceled"
					}
				}
				cancel()
				mu.Lock() // handlers have returned; the lock only orders the reads
				for id, n := range called {
					if n != 1 || results[id] != 1 || completes[id] != 1 {
						t.Fatalf("k=%d: call %s: ToolCall=%d ToolResult=%d StepComplete=%d, want 1/1/1", k, id, n, results[id], completes[id])
					}
					if !ran[id] {
						announcedNotRun++
					}
				}
				mu.Unlock()
				for id := range results {
					if called[id] == 0 {
						t.Fatalf("k=%d: unannounced call %s published a result", k, id)
					}
				}
				for _, m := range ag.Messages() {
					if m.Role == llm.RoleTool && m.ToolCallID == "" {
						t.Fatalf("k=%d: tool result without id: %#v", k, m)
					}
				}
				if !canceled {
					break
				}
			}
			if announcedNotRun == 0 {
				t.Fatal("no cancellation point landed between announcement and handler start")
			}
		})
	}
}
