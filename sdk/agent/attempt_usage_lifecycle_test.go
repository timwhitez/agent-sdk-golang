package agent

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// attemptUsageStep scripts one InvokeStream attempt.
type attemptUsageStep struct {
	usage  []int // cumulative total-token snapshots reported by the attempt
	text   string
	failed bool
}

type attemptUsageModel struct {
	mu    sync.Mutex
	steps []attemptUsageStep
	calls int
}

func (m *attemptUsageModel) Provider() string { return "fixture" }
func (m *attemptUsageModel) Model() string    { return "attempt-usage" }
func (m *attemptUsageModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	panic("buffered path unused")
}
func (m *attemptUsageModel) InvokeStream(context.Context, llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.mu.Lock()
	step := m.steps[m.calls]
	m.calls++
	m.mu.Unlock()
	ch := make(chan llm.StreamEvent, len(step.usage)+3)
	for _, total := range step.usage {
		ch <- llm.StreamUsageEvent{Usage: llm.Usage{PromptTokens: total - 1, CompletionTokens: 1, TotalTokens: total}}
	}
	if step.failed {
		ch <- llm.StreamErrorEvent{Err: &llm.ProviderError{Provider: "fixture", StatusCode: 503, Message: "overloaded"}}
	} else {
		ch <- llm.StreamTextDeltaEvent{Delta: step.text}
		ch <- llm.StreamDoneEvent{StopReason: "stop"}
	}
	close(ch)
	return ch, nil
}

type observedUsage struct {
	total   int
	frameID string
	attempt uint64
}

func collectAttemptUsage(t *testing.T, ag *Agent, ctx context.Context) ([]observedUsage, []observedUsage) {
	t.Helper()
	var usage, accounting []observedUsage
	for envelope := range ag.QueryStreamEnveloped(ctx, llm.TextContent("hello")) {
		switch event := envelope.Event.(type) {
		case UsageEvent:
			usage = append(usage, observedUsage{total: event.Usage.TotalTokens, frameID: envelope.FrameID, attempt: envelope.InvokeAttempt})
		case AccountingEvent:
			if event.CorrelationKind == "response" {
				total := 0
				if event.Payload.Usage != nil && event.Payload.Usage.TotalTokens != nil {
					total = int(*event.Payload.Usage.TotalTokens)
				}
				accounting = append(accounting, observedUsage{total: total, frameID: envelope.FrameID, attempt: envelope.InvokeAttempt})
			}
		}
	}
	return usage, accounting
}

// A failed attempt that reported usage before any visible text and is then
// transparently retried keeps its billed usage: each attempt's usage is
// published once, under its own Frame/InvokeAttempt correlation.
func TestAttemptUsageIsReportedPerInvokeAttempt(t *testing.T) {
	for _, tt := range []struct {
		name  string
		steps []attemptUsageStep
		want  []observedUsage // frameID filled with "*" = any non-empty shared Frame
	}{
		{
			name:  "failure then success",
			steps: []attemptUsageStep{{usage: []int{107}, failed: true}, {usage: []int{103}, text: "ok"}},
			want:  []observedUsage{{total: 107, attempt: 1}, {total: 103, attempt: 2}},
		},
		{
			name:  "two failures exhaust retries",
			steps: []attemptUsageStep{{usage: []int{50}, failed: true}, {usage: []int{60}, failed: true}},
			want:  []observedUsage{{total: 50, attempt: 1}, {total: 60, attempt: 2}},
		},
		{
			name:  "cumulative snapshots settle once",
			steps: []attemptUsageStep{{usage: []int{100, 120}, failed: true}, {usage: []int{30, 40}, text: "ok"}},
			want:  []observedUsage{{total: 120, attempt: 1}, {total: 40, attempt: 2}},
		},
		{
			name:  "failed attempt without usage stays unreported",
			steps: []attemptUsageStep{{failed: true}, {usage: []int{103}, text: "ok"}},
			want:  []observedUsage{{total: 103, attempt: 2}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model := &attemptUsageModel{steps: tt.steps}
			ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			usage, accounting := collectAttemptUsage(t, ag, context.Background())
			if model.calls != len(tt.steps) {
				t.Fatalf("model calls=%d", model.calls)
			}
			for name, got := range map[string][]observedUsage{"usage": usage, "accounting": accounting} {
				if len(got) != len(tt.want) {
					t.Fatalf("%s events=%+v want %+v", name, got, tt.want)
				}
				for i, event := range got {
					if event.total != tt.want[i].total || event.attempt != tt.want[i].attempt || event.frameID == "" || event.frameID != got[0].frameID {
						t.Fatalf("%s[%d]=%+v want %+v in one Frame", name, i, event, tt.want[i])
					}
				}
			}
		})
	}
}

// Cancellation while waiting to retry reports the failed attempt's usage
// exactly once (on the terminal path) and starts no further attempt.
func TestAttemptUsageCancelledDuringBackoffIsReportedOnce(t *testing.T) {
	model := &attemptUsageModel{steps: []attemptUsageStep{{usage: []int{77}, failed: true}, {usage: []int{1}, text: "never"}}}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2, InvokeRetryBackoff: time.Hour, Warningf: func(string, ...any) {
		// The retry warning is logged immediately before the backoff wait.
		cancel()
	}})
	if err != nil {
		t.Fatal(err)
	}
	usage, accounting := collectAttemptUsage(t, ag, ctx)
	if model.calls != 1 {
		t.Fatalf("model calls=%d, want no retry after cancellation", model.calls)
	}
	if len(usage) != 1 || usage[0].total != 77 || usage[0].attempt != 1 || len(accounting) != 1 {
		t.Fatalf("usage=%+v accounting=%+v", usage, accounting)
	}
}
