package agent

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// settlingWriter records acknowledged checkpoints and their settled outcomes.
// onSave runs inside SaveCompactionCheckpoint before it acknowledges.
type settlingWriter struct {
	mu          sync.Mutex
	checkpoints []compaction.CompactionCheckpoint
	outcomes    []compaction.CompactionCheckpointOutcome
	onSave      func(int)
}

func (w *settlingWriter) SaveCompactionCheckpoint(_ context.Context, checkpoint compaction.CompactionCheckpoint) error {
	w.mu.Lock()
	w.checkpoints = append(w.checkpoints, checkpoint)
	n := len(w.checkpoints)
	w.mu.Unlock()
	if w.onSave != nil {
		w.onSave(n)
	}
	return nil
}

func (w *settlingWriter) SettleCompactionCheckpoint(_ context.Context, outcome compaction.CompactionCheckpointOutcome) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.outcomes = append(w.outcomes, outcome)
	return nil
}

func (w *settlingWriter) snapshot() ([]compaction.CompactionCheckpoint, []compaction.CompactionCheckpointOutcome) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]compaction.CompactionCheckpoint(nil), w.checkpoints...), append([]compaction.CompactionCheckpointOutcome(nil), w.outcomes...)
}

// #301 case 2: the host acknowledged a checkpoint, then history changed before
// the Agent could install it. The Agent reports that exact checkpoint as
// abandoned (and emits no CompactionEvent); the rebased retry is a new
// checkpoint that is reported published before its CompactionEvent.
func TestAcknowledgedCheckpointAbandonedAfterHistoryChangeIsSettled(t *testing.T) {
	model := newGatedSummaryModel(
		func() (*llm.Completion, error) { return nil, typedOverflow() },
	)
	close(model.gate)
	var ag *Agent
	updateErr := make(chan error, 1)
	writer := &settlingWriter{}
	writer.onSave = func(n int) {
		if n == 1 {
			current := ag.Messages()
			current[0] = llm.NewSystemMessage("system with refreshed host context")
			updateErr <- ag.ReplaceHistoryChecked(current)
		}
	}
	ag, _ = interleaveAgent(t, model, writer, nil)
	compactions, _, _ := drainQuery(ag, context.Background(), "current request", nil)
	if err := <-updateErr; err != nil {
		t.Fatalf("host system update rejected: %v", err)
	}
	checkpoints, outcomes := writer.snapshot()
	if compactions != 0 || len(checkpoints) != 1 || len(outcomes) != 1 {
		t.Fatalf("compactions=%d checkpoints=%d outcomes=%+v", compactions, len(checkpoints), outcomes)
	}
	abandoned := outcomes[0]
	if abandoned.Published || abandoned.CheckpointID != checkpoints[0].CheckpointID || abandoned.Messages != len(checkpoints[0].Messages) || abandoned.Reason == "" {
		t.Fatalf("abandoned outcome = %+v, checkpoint %s/%d", abandoned, checkpoints[0].CheckpointID, len(checkpoints[0].Messages))
	}

	// The retry: the published outcome precedes the CompactionEvent and names
	// exactly the installed prefix.
	var sawEvent bool
	for ev := range ag.QueryStream(context.Background(), llm.TextContent("next request")) {
		event, ok := ev.(CompactionEvent)
		if !ok {
			continue
		}
		sawEvent = true
		checkpoints, outcomes = writer.snapshot()
		if len(checkpoints) != 2 || len(outcomes) != 2 {
			t.Fatalf("at CompactionEvent: checkpoints=%d outcomes=%+v", len(checkpoints), outcomes)
		}
		published := outcomes[1]
		if !published.Published || published.CheckpointID != checkpoints[1].CheckpointID || published.CheckpointID != event.Result.CheckpointID || published.Messages != event.Result.CheckpointMessages || published.Reason != "" {
			t.Fatalf("published outcome = %+v, event result %s/%d", published, event.Result.CheckpointID, event.Result.CheckpointMessages)
		}
		if published.CheckpointID == abandoned.CheckpointID {
			t.Fatal("the retry reused the abandoned checkpoint identity")
		}
	}
	if !sawEvent {
		t.Fatal("the rebased retry was not published")
	}
}

// #301 case 1: an ephemeral tool result inside the checkpoint prefix that the
// next request's scan would recycle is recycled before the checkpoint is
// written, so the installed history equals the checkpoint and stays equal
// after the scan.
func TestPublishedCheckpointAlreadyCarriesEphemeralRelease(t *testing.T) {
	type args struct {
		ID string `json:"id"`
	}
	tasklist := tools.Func[args]("tasklist_get", "get", func(context.Context, args, *tools.Container) (any, error) {
		return "tasks", nil
	}).WithEphemeralKeep(1)
	writer := &settlingWriter{}
	ag, err := New(Config{
		LLM:        &ephemeralRetentionModel{},
		Tools:      []tools.Tool{tasklist},
		Compaction: &compaction.Config{Enabled: true, ContextWindow: 100000, ThresholdRatio: 0.85, CheckpointWriter: writer},
		Warningf:   func(string, ...any) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	call := func(id string) llm.Message {
		return llm.NewAssistantMessage("", []llm.ToolCall{{ID: id, Type: "function", Function: llm.FunctionCall{Name: "tasklist_get", Arguments: `{"id":"t1"}`}}})
	}
	result := func(id string) llm.Message {
		return llm.Message{Role: llm.RoleTool, ToolCallID: id, ToolName: "tasklist_get", Content: llm.TextContent("tasks " + id), Ephemeral: true}
	}
	source := []llm.Message{llm.NewSystemMessage("system"), llm.NewUserMessage("old"), llm.NewUserMessage("work"), call("r1"), result("r1")}
	// The candidate keeps the first result; the second arrives after the
	// candidate's source was sampled, so it is in the live tail.
	candidate := []llm.Message{llm.NewSystemMessage("system"), llm.NewUserMessage("summary"), llm.NewUserMessage("work"), call("r1"), result("r1")}
	live := append(llm.CloneMessages(source), call("r2"), result("r2"))
	ag.mu.Lock()
	ag.messages = llm.CloneMessages(live)
	ag.mu.Unlock()
	ag.pendingCompaction = &pendingCompaction{messages: candidate, snapshotLen: len(source), source: llm.CloneMessages(source), result: compaction.Result{Compacted: true}}
	if !ag.applyPendingCompaction(nil) {
		t.Fatal("candidate was not published")
	}
	checkpoints, outcomes := writer.snapshot()
	if len(checkpoints) != 1 || len(outcomes) != 1 || !outcomes[0].Published || outcomes[0].CheckpointID != checkpoints[0].CheckpointID {
		t.Fatalf("checkpoints=%d outcomes=%+v", len(checkpoints), outcomes)
	}
	n := outcomes[0].Messages
	installed := ag.Messages()
	if n != len(checkpoints[0].Messages) || n > len(installed) || !messageJSONEqual(installed[:n], checkpoints[0].Messages) {
		t.Fatalf("installed prefix differs from the checkpoint:\ninstalled=%+v\ncheckpoint=%+v", installed, checkpoints[0].Messages)
	}
	// The scan that precedes the next request must not rewrite the prefix.
	ag.destroyEphemeralMessages()
	scanned := ag.Messages()
	if !messageJSONEqual(scanned[:n], checkpoints[0].Messages) {
		t.Fatalf("ephemeral scan rewrote the published checkpoint prefix:\nscanned=%+v\ncheckpoint=%+v", scanned, checkpoints[0].Messages)
	}
	released := 0
	for _, m := range scanned {
		if m.Destroyed {
			released++
		}
	}
	if released != 1 {
		t.Fatalf("released %d ephemeral results, want the older one only", released)
	}
}

// Every Agent-owned publication path settles its acknowledged checkpoint:
// manual compaction (no event stream) and a host-computed candidate.
func TestManualAndHostCandidateCheckpointsAreSettledPublished(t *testing.T) {
	model := newGatedSummaryModel()
	close(model.gate)
	writer := &settlingWriter{}
	ag, _ := interleaveAgent(t, model, writer, nil)
	res, err := ag.CompactNow(context.Background())
	if err != nil || !res.Compacted {
		t.Fatalf("CompactNow: %+v %v", res, err)
	}
	checkpoints, outcomes := writer.snapshot()
	if len(outcomes) != 1 || !outcomes[0].Published || outcomes[0].CheckpointID != res.CheckpointID || outcomes[0].CheckpointID != checkpoints[0].CheckpointID {
		t.Fatalf("manual outcomes=%+v result=%s", outcomes, res.CheckpointID)
	}
	if !messageJSONEqual(ag.Messages(), checkpoints[0].Messages) {
		t.Fatal("manual installed history differs from its checkpoint")
	}

	expected := ag.Messages()
	candidate := []llm.Message{llm.NewSystemMessage("system"), llm.NewUserMessage("host candidate")}
	res, err = ag.CommitCompactionHistory(context.Background(), expected, candidate, compaction.Result{Compacted: true})
	if err != nil || !res.Compacted {
		t.Fatalf("CommitCompactionHistory: %+v %v", res, err)
	}
	_, outcomes = writer.snapshot()
	if len(outcomes) != 2 || !outcomes[1].Published || outcomes[1].CheckpointID != res.CheckpointID || outcomes[1].Messages != len(candidate) {
		t.Fatalf("host candidate outcomes=%+v result=%s", outcomes, res.CheckpointID)
	}
}

// A writer without the optional settler keeps its old contract.
func TestCheckpointWriterWithoutSettlerIsUnaffected(t *testing.T) {
	model := newGatedSummaryModel()
	close(model.gate)
	var writes atomic.Int32
	writer := compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
		writes.Add(1)
		return nil
	})
	ag, _ := interleaveAgent(t, model, writer, nil)
	if res, err := ag.CompactNow(context.Background()); err != nil || !res.Compacted || writes.Load() != 1 {
		t.Fatalf("CompactNow: %+v %v writes=%d", res, err, writes.Load())
	}
}
