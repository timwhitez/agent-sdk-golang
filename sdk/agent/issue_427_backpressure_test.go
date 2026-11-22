package agent

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type issue427BudgetModel struct{ calls int }

func TestIssue427BackgroundTurnRegistersFloorBudget(t *testing.T) {
	cancelable, cancel := context.WithCancel(context.Background())
	defer cancel()
	canceled, done := context.WithCancel(context.Background())
	done()
	for name, ctx := range map[string]context.Context{
		"background": context.Background(), "todo": context.TODO(),
		"without-cancel": context.WithoutCancel(canceled), "cancelable": cancelable, "canceled": canceled,
	} {
		t.Run(name, func(t *testing.T) {
			a := &Agent{}
			out := &eventOutput{}
			release := a.registerTurnCancellation(out, ctx)
			state := a.turnBackpressureState(out)
			if state == nil {
				t.Fatal("query was not assigned a turn budget state")
			}
			if state.done != ctx.Done() {
				t.Fatal("cancellation channel was changed")
			}
			state.chargeFloorSpent(criticalEventFloorTurnBudget)
			if a.criticalFloorBudgetAvailable(state) {
				t.Fatal("turn budget not exhausted")
			}
			release()
			if a.turnBackpressureState(out) != nil {
				t.Fatal("finished turn still registered")
			}
			defer a.registerTurnCancellation(out, ctx)()
			fresh := a.turnBackpressureState(out)
			if fresh == state || fresh.floorSpentNanos.Load() != 0 {
				t.Fatal("new turn reused old budget")
			}
		})
	}
}
func (*issue427BudgetModel) Provider() string { return "issue427" }
func (*issue427BudgetModel) Model() string    { return "issue427" }
func (m *issue427BudgetModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls > 1 {
		return &llm.Completion{Content: llm.TextContent("done"), StopReason: "stop"}, nil
	}
	calls := []llm.ToolCall{}
	for i := 0; i < 4; i++ {
		calls = append(calls, llm.ToolCall{ID: fmt.Sprintf("call_%d", i), Type: "function", Function: llm.FunctionCall{Name: "noop", Arguments: fmt.Sprintf(`{"n":%d}`, i)}})
	}
	return &llm.Completion{ToolCalls: calls, StopReason: "tool_calls"}, nil
}
func TestIssue427QueryBackgroundGetsCriticalFloorBudget(t *testing.T) {
	for _, cancelable := range []bool{false, true} {
		t.Run(fmt.Sprintf("cancelable=%v", cancelable), func(t *testing.T) {
			ctx := context.Background()
			if cancelable {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				defer cancel()
			}
			tool := tools.Func[struct {
				N int `json:"n"`
			}]("noop", "synthetic no-op", func(context.Context, struct {
				N int `json:"n"`
			}, *tools.Container) (any, error) {
				return "ok", nil
			})
			a, err := New(Config{LLM: &issue427BudgetModel{}, Tools: []tools.Tool{tool}, MaxIterations: 2, EventBufferSize: 1, EventSendTimeout: time.Millisecond, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			start := time.Now()
			events, receipt := a.QueryStreamEnvelopedWithReceipt(ctx, llm.TextContent("run synthetic no-ops"), nil)
			// Model a consumer that has stopped reading. Summary is safe to inspect
			// without draining the event stream and is filled before producer close.
			timer := time.NewTimer(10 * time.Second)
			defer timer.Stop()
			tick := time.NewTicker(10 * time.Millisecond)
			defer tick.Stop()
			for {
				if summary, ok := receipt.Summary(); ok {
					elapsed := time.Since(start)
					t.Logf("elapsed=%s drops=%d critical=%d", elapsed, summary.DroppedEvents, summary.DroppedCriticalEvents)
					for range events {
					}
					if elapsed > 3500*time.Millisecond {
						t.Fatalf("query paid an unbounded per-event floor instead of the 2s per-turn budget: %s", elapsed)
					}
					return
				}
				select {
				case <-tick.C:
				case <-timer.C:
					for range events {
					}
					t.Fatal("query did not finish")
				}
			}
		})
	}
}
