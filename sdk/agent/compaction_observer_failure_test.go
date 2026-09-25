package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// historyHasSummary reports whether history starts from a published summary
// and still holds the user's input exactly once.
func historyHasSummary(messages []llm.Message) (summary bool, inputs int) {
	return countText(messages, "prior work summarized") > 0, countText(messages, "current request")
}

// G3: once a compaction is committed (checkpoint written, history
// published), observers cannot undo it. Here the event consumer stops reading,
// so the consistency-critical CompactionEvent waits out its floor and is
// dropped, and the Warningf that reports the drop blocks. While the warning is
// blocked the history is already the committed one, and afterwards the model
// call count is exactly that of the recovery: nothing is rolled back,
// retried or re-summarized.
func TestStalledConsumerAndBlockedWarningDoNotUndoCommittedCompaction(t *testing.T) {
	retried := make(chan struct{})
	model := newGatedSummaryModel(
		// A tool round first fills the one-slot event buffer, so the
		// CompactionEvent below meets a full channel.
		func() (*llm.Completion, error) {
			return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "noop", Arguments: "{}"}}}}, nil
		},
		func() (*llm.Completion, error) { return nil, typedOverflow() },
		func() (*llm.Completion, error) {
			close(retried)
			return &llm.Completion{Content: llm.TextContent("ok")}, nil
		},
	)
	close(model.gate)
	writer := &countingCheckpointWriter{}
	warnEntered, releaseWarn := make(chan struct{}), make(chan struct{})
	var warnOnce sync.Once
	noop := tools.Func[struct{}]("noop", "noop", func(context.Context, struct{}, *tools.Container) (any, error) {
		return "done", nil
	})
	ag, _ := interleaveAgent(t, model, writer, func(c *Config) {
		c.Tools = []tools.Tool{noop}
		c.EventBufferSize = 1
		c.EventSendTimeout = time.Millisecond
		c.EventDropLogEvery = 1
		c.Warningf = func(format string, args ...any) {
			msg := fmt.Sprintf(format, args...)
			if strings.Contains(msg, "dropping") && strings.Contains(msg, "CompactionEvent") {
				warnOnce.Do(func() {
					close(warnEntered)
					<-releaseWarn
				})
			}
		}
	})
	events := ag.QueryStreamEnveloped(context.Background(), llm.TextContent("current request"))

	select {
	case <-warnEntered:
	case <-time.After(30 * time.Second):
		t.Fatal("the CompactionEvent drop was never reported")
	}
	main, summaries := model.snapshot()
	summary, inputs := historyHasSummary(ag.Messages())
	if !summary || inputs != 1 || writer.writes.Load() != 1 || len(main) != 2 || summaries != 1 {
		t.Fatalf("while the warning blocks: summary=%v inputs=%d writes=%d main=%d summaries=%d", summary, inputs, writer.writes.Load(), len(main), summaries)
	}
	close(releaseWarn)
	// The retried request is made while the consumer still reads nothing.
	select {
	case <-retried:
	case <-time.After(30 * time.Second):
		t.Fatal("the recovered request was not sent")
	}
	compactionsSeen, finals, errs := 0, 0, 0
	for env := range events {
		switch env.Event.(type) {
		case CompactionEvent:
			compactionsSeen++
		case FinalResponseEvent:
			finals++
		case ErrorEvent:
			errs++
		}
	}
	main, summaries = model.snapshot()
	summary, inputs = historyHasSummary(ag.Messages())
	if len(main) != 3 || summaries != 1 || writer.writes.Load() != 1 || finals != 1 || errs != 0 {
		t.Fatalf("main=%d summaries=%d writes=%d finals=%d errors=%d", len(main), summaries, writer.writes.Load(), finals, errs)
	}
	if !summary || inputs != 1 {
		t.Fatalf("committed compaction was undone: summary=%v inputs=%d", summary, inputs)
	}
	if compactionsSeen != 0 {
		t.Fatalf("the dropped CompactionEvent was delivered %d times", compactionsSeen)
	}
	if countText(main[2].Messages, "prior work summarized") == 0 {
		t.Fatal("the recovered request does not carry the committed summary")
	}
}

// G3: a failing host checkpoint inspection (CheckpointProvider) degrades the
// summary material to UNKNOWN. With a summary that preserves UNKNOWN the
// compaction commits and the recovery makes exactly one summary and one
// retry. With a summary that drops it, the quality gate rejects the summary:
// the recovery fails within its summary budget, nothing is written and the
// history is unchanged, and the turn ends after the one rejected request.
func TestCheckpointInspectionFailureKeepsCompactionAndCallCount(t *testing.T) {
	for _, faithful := range []bool{true, false} {
		t.Run(map[bool]string{true: "summary preserves UNKNOWN", false: "summary drops UNKNOWN"}[faithful], func(t *testing.T) {
			model := newGatedSummaryModel(
				func() (*llm.Completion, error) { return nil, typedOverflow() },
			)
			model.keepUnknown = faithful
			close(model.gate)
			writer := &countingCheckpointWriter{}
			inspections := 0
			ag, _ := interleaveAgent(t, model, writer, func(c *Config) {
				c.Compaction.CheckpointProvider = func(context.Context, []llm.Message) (compaction.CheckpointContext, error) {
					inspections++
					return compaction.CheckpointContext{}, errors.New("host store unavailable")
				}
			})
			before := ag.Messages()
			_, plannedAttempts := ag.overflowSummaryPlan()
			compactions, errs, finals := drainQuery(ag, context.Background(), "current request", nil)
			main, summaries := model.snapshot()
			model.mu.Lock()
			material := append([]string(nil), model.summaryText...)
			model.mu.Unlock()
			for _, text := range material {
				if !strings.Contains(text, "Status: UNKNOWN") {
					t.Fatal("the failed inspection is not recorded as UNKNOWN in the summary material")
				}
			}
			summary, inputs := historyHasSummary(ag.Messages())
			if faithful {
				if len(main) != 2 || summaries != 1 || compactions != 1 || errs != 0 || finals != 1 || writer.writes.Load() != 1 || inspections != 1 {
					t.Fatalf("main=%d summaries=%d compactions=%d errors=%d finals=%d writes=%d inspections=%d", len(main), summaries, compactions, errs, finals, writer.writes.Load(), inspections)
				}
				if !summary || inputs != 1 {
					t.Fatalf("compaction not kept: summary=%v inputs=%d", summary, inputs)
				}
				return
			}
			if len(main) != 1 || summaries != plannedAttempts || compactions != 0 || errs != 1 || finals != 0 || writer.writes.Load() != 0 {
				t.Fatalf("main=%d summaries=%d (budget %d) compactions=%d errors=%d finals=%d writes=%d", len(main), summaries, plannedAttempts, compactions, errs, finals, writer.writes.Load())
			}
			want := append(llm.CloneMessages(before), llm.NewUserMessage("current request"))
			if summary || !messageJSONEqual(ag.Messages(), want) {
				t.Fatal("a rejected summary changed history")
			}
		})
	}
}
