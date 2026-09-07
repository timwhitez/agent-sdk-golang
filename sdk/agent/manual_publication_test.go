package agent

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// Measures admission plus existing runtime-use overhead with compaction disabled,
// not summary/provider latency or a claimed improvement over the old path.
func BenchmarkManualCompactionAdmission(b *testing.B) {
	ag, err := New(Config{LLM: &countingCompactionModel{}, Compaction: &compaction.Config{Enabled: false}})
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ag.CompactNow(ctx); err != nil {
			b.Fatal(err)
		}
	}
}

func TestManualCompactionOwnsHistoryThroughCheckpointPublication(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	original := []llm.Message{llm.NewSystemMessage("base"), llm.NewUserMessage("old request"), llm.NewAssistantMessage("old answer", nil)}
	var checkpoint []llm.Message
	var ag *Agent
	var callbackMutation error
	var err error
	ag, err = New(Config{
		LLM: &countingCompactionModel{}, InitialMessages: original,
		Compaction: &compaction.Config{Enabled: true, CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error { return nil })},
	})
	if err != nil {
		t.Fatal(err)
	}
	ag.compactor.Config.CheckpointWriter = compaction.CompactionCheckpointWriterFunc(func(_ context.Context, cp compaction.CompactionCheckpoint) error {
		checkpoint = llm.CloneMessages(cp.Messages)
		// Reentrant callbacks may read history and queue runtime updates, but
		// must not mutate the history whose checkpoint they are publishing.
		_ = ag.Messages()
		callbackMutation = ag.ClearHistoryChecked()
		ag.UpdateCompactionConfig(&compaction.Config{Enabled: false})
		close(entered)
		<-release
		return nil
	})
	done := make(chan error, 1)
	go func() { _, err := ag.CompactNow(context.Background()); done <- err }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		close(release)
		t.Fatal("checkpoint callback blocked")
	}
	for _, candidate := range [][]llm.Message{nil, {llm.NewUserMessage("new history")}, append([]llm.Message{llm.NewSystemMessage("new system")}, original[1:]...)} {
		if err := ag.ReplaceHistoryChecked(candidate); !errors.Is(err, ErrActiveHistoryMutation) {
			t.Errorf("replacement error=%v", err)
		}
	}
	if !errors.Is(callbackMutation, ErrActiveHistoryMutation) {
		t.Errorf("callback mutation=%v", callbackMutation)
	}
	for _, compact := range []func() (compaction.Result, error){
		func() (compaction.Result, error) { return ag.CompactNow(context.Background()) },
		func() (compaction.Result, error) { return ag.CompactLocalNow(context.Background(), 100) },
		func() (compaction.Result, error) {
			return ag.CompactPipelineNow(context.Background(), compaction.PipelineRequest{Trigger: "preflight"})
		},
	} {
		res, err := compact()
		if !errors.Is(err, ErrAgentBusy) || res.Compacted {
			t.Errorf("nested compaction=%v error=%v", res.Compacted, err)
		}
	}
	events := ag.QueryStream(context.Background(), llm.TextContent("must not enter history"))
	busy := 0
	for event := range events {
		if e, ok := event.(ErrorEvent); ok && e.Kind == "agent_busy" {
			busy++
		} else {
			t.Errorf("unexpected event %T", event)
		}
	}
	if busy != 1 || !reflect.DeepEqual(ag.Messages(), original) {
		t.Error("busy admission mutated history")
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ag.Messages(), checkpoint) {
		t.Fatal("published history differs from committed checkpoint")
	}
	if ag.hasCompactor {
		t.Fatal("queued runtime update was not released")
	}
	if err := ag.ReplaceHistoryChecked(original); err != nil {
		t.Fatalf("ownership leaked: %v", err)
	}
	drainCompactionUpdateTurn(t, ag.QueryStream(context.Background(), llm.TextContent("after release")))
}

func TestManualCompactionCannotEnterActiveQuery(t *testing.T) {
	model := &blockingCompactionUpdateModel{started: make(chan struct{}), release: make(chan struct{})}
	ag := newCompactionUpdateAgent(t, model, 100000)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream := ag.QueryStream(ctx, llm.TextContent("active"))
	select {
	case <-model.started:
	case <-time.After(5 * time.Second):
		t.Fatal("query not started")
	}
	before := ag.Messages()
	res, err := ag.CompactNow(context.Background())
	if !errors.Is(err, ErrAgentBusy) || res.Compacted || model.calls.Load() != 1 || !reflect.DeepEqual(before, ag.Messages()) {
		t.Errorf("active compaction=%v err=%v calls=%d", res.Compacted, err, model.calls.Load())
	}
	cancel()
	drainCompactionUpdateTurn(t, stream)
}

func TestManualCompactionClassifiesLingeringAutomaticWorkAsBusy(t *testing.T) {
	ag := newCompactionUpdateAgent(t, &countingCompactionModel{}, 100000)
	// A canceled turn can finish while a context-ignoring automatic summary
	// still holds compactionInFlight. The host must not enter fallback.
	ag.compactionInFlight.Store(true)
	res, err := ag.CompactNow(context.Background())
	if !errors.Is(err, ErrAgentBusy) || res.Compacted {
		t.Fatalf("result=%v error=%v", res.Compacted, err)
	}
	ag.compactionInFlight.Store(false)
	if err := ag.ClearHistoryChecked(); err != nil {
		t.Fatalf("manual ownership leaked: %v", err)
	}
}

func TestCanceledManualSummaryReleasesAdmission(t *testing.T) {
	model := &blockingCompactionUpdateModel{started: make(chan struct{}), release: make(chan struct{})}
	ag := newCompactionUpdateAgent(t, model, 100000)
	if err := ag.ReplaceHistoryChecked([]llm.Message{llm.NewUserMessage("compact")}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := ag.CompactNow(ctx); done <- err }()
	select {
	case <-model.started:
	case <-time.After(5 * time.Second):
		t.Fatal("summary not started")
	}
	if err := ag.ClearHistoryChecked(); !errors.Is(err, ErrActiveHistoryMutation) {
		t.Errorf("active manual clear=%v", err)
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("cancel error=%v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("summary did not cancel")
	}
	if err := ag.ClearHistoryChecked(); err != nil {
		t.Fatalf("ownership leaked: %v", err)
	}
	ag.UpdateCompactionConfig(&compaction.Config{Enabled: false})
	close(model.release)
	drainCompactionUpdateTurn(t, ag.QueryStream(context.Background(), llm.TextContent("next")))
}

func TestManualCompactionReleasesAdmissionAfterEarlyExit(t *testing.T) {
	for _, mode := range []string{"disabled", "canceled", "checkpoint_failure"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cfg := &compaction.Config{Enabled: mode != "disabled"}
			failure := errors.New("checkpoint failure")
			if mode == "checkpoint_failure" {
				cfg.CheckpointWriter = compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error { return failure })
			}
			ag, err := New(Config{LLM: &countingCompactionModel{}, Compaction: cfg, InitialMessages: []llm.Message{llm.NewUserMessage("original")}})
			if err != nil {
				t.Fatal(err)
			}
			if mode == "canceled" {
				cancel()
			}
			_, err = ag.CompactNow(ctx)
			if mode == "checkpoint_failure" && !errors.Is(err, failure) {
				t.Fatalf("error=%v", err)
			}
			if mode == "canceled" && !errors.Is(err, context.Canceled) {
				t.Fatalf("error=%v", err)
			}
			if mode == "disabled" && err != nil {
				t.Fatal(err)
			}
			if err := ag.ClearHistoryChecked(); err != nil {
				t.Fatalf("ownership leaked: %v", err)
			}
			ag.UpdateCompactionConfig(&compaction.Config{Enabled: false})
			drainCompactionUpdateTurn(t, ag.QueryStream(context.Background(), llm.TextContent("after exit")))
		})
	}
}
