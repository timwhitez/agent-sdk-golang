package agent

import (
	"context"
	"errors"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func BenchmarkCompactionHistoryPublication(b *testing.B) {
	for _, count := range []int{1, 32, 256} {
		b.Run(strconv.Itoa(count), func(b *testing.B) {
			messages := make([]llm.Message, count)
			for i := range messages {
				messages[i] = llm.NewUserMessage(strings.Repeat("x", 1024))
			}
			ag, err := New(Config{LLM: &countingCompactionModel{}, InitialMessages: messages})
			if err != nil {
				b.Fatal(err)
			}
			ctx := context.Background()
			res := compaction.Result{Compacted: true}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := ag.CommitCompactionHistory(ctx, messages, messages, res); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestCompactionHistoryRejectsBeforePersistence(t *testing.T) {
	for _, mode := range []string{"stale", "query", "inflight", "pending", "runtime_update", "canceled", "noop"} {
		t.Run(mode, func(t *testing.T) {
			writes := 0
			ag, err := New(Config{LLM: &countingCompactionModel{}, InitialMessages: []llm.Message{llm.NewUserMessage("source")}, Compaction: &compaction.Config{Enabled: true, CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error { writes++; return nil })}})
			if err != nil {
				t.Fatal(err)
			}
			source := ag.Messages()
			expected := llm.CloneMessages(source)
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			want := error(ErrAgentBusy)
			res := compaction.Result{Compacted: true, CheckpointID: "untrusted-old-id", CheckpointMessages: 99}
			var release func()
			switch mode {
			case "stale":
				expected[0].Name = "different identity, same text"
				want = ErrStaleCompactionHistory
			case "query":
				ag.turnActive.Store(true)
				defer ag.turnActive.Store(false)
			case "inflight":
				ag.compactionInFlight.Store(true)
				defer ag.compactionInFlight.Store(false)
			case "pending":
				ag.pendingCompaction = &pendingCompaction{messages: []llm.Message{llm.NewUserMessage("pending")}, result: compaction.Result{Compacted: true}}
			case "runtime_update":
				release, err = ag.beginCompactionRuntimeUse(ctx)
				if err != nil {
					t.Fatal(err)
				}
				defer release()
				ag.UpdateCompactionConfig(&compaction.Config{Enabled: false})
			case "canceled":
				cancel()
				want = context.Canceled
			case "noop":
				res.Compacted = false
				want = nil
			}
			pending := ag.pendingCompaction
			ag.ephemeralScanFrom = 7
			out, err := ag.CommitCompactionHistory(ctx, expected, []llm.Message{llm.NewUserMessage("candidate")}, res)
			if !errors.Is(err, want) || out.Compacted || out.CheckpointID != "" || out.CheckpointMessages != 0 || writes != 0 || !reflect.DeepEqual(source, ag.Messages()) || ag.compactionGeneration.Load() != 0 || ag.ephemeralScanFrom != 7 || ag.pendingCompaction != pending {
				t.Fatalf("mode=%s writes=%d compacted=%v id=%s error=%v", mode, writes, out.Compacted, out.CheckpointID, err)
			}
			if mode != "query" && ag.turnActive.Load() {
				t.Fatal("publication admission leaked")
			}
		})
	}
}

func TestEqualCompactionCandidateStillCommits(t *testing.T) {
	writes := 0
	ag, err := New(Config{LLM: &countingCompactionModel{}, InitialMessages: []llm.Message{llm.NewUserMessage("same content")}, Compaction: &compaction.Config{Enabled: true, CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error { writes++; return nil })}})
	if err != nil {
		t.Fatal(err)
	}
	source := ag.Messages()
	out, err := ag.CommitCompactionHistory(context.Background(), source, source, compaction.Result{Compacted: true})
	if err != nil || writes != 1 || !out.Compacted || out.CheckpointID == "" || ag.compactionGeneration.Load() != 1 {
		t.Fatalf("equal-content commit: writes=%d compacted=%v error=%v", writes, out.Compacted, err)
	}
}

func TestCompactionHistoryOwnsCandidateAndCompletesAcknowledgedCommit(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var ag *Agent
	var saved compaction.CompactionCheckpoint
	source := []llm.Message{llm.NewUserMessage("source")}
	candidate := []llm.Message{llm.NewUserMessage("candidate")}
	want := llm.CloneMessages(candidate)
	var err error
	ag, err = New(Config{LLM: &countingCompactionModel{}, InitialMessages: source, Compaction: &compaction.Config{Enabled: true, CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(_ context.Context, cp compaction.CompactionCheckpoint) error {
		if !reflect.DeepEqual(ag.Messages(), source) {
			t.Error("history published before checkpoint acknowledgement")
		}
		saved = cp
		saved.Messages = llm.CloneMessages(cp.Messages)
		cp.Messages[0].Content = llm.TextContent("writer mutation")
		candidate[0].Content = llm.TextContent("caller mutation")
		if !errors.Is(ag.ClearHistoryChecked(), ErrActiveHistoryMutation) {
			t.Error("callback mutation accepted")
		}
		if _, err := ag.CommitCompactionHistory(ctx, source, want, compaction.Result{Compacted: true}); !errors.Is(err, ErrAgentBusy) {
			t.Errorf("nested publication=%v", err)
		}
		busy := 0
		for event := range ag.QueryStream(context.Background(), llm.TextContent("not admitted")) {
			if e, ok := event.(ErrorEvent); ok && e.Kind == "agent_busy" {
				busy++
			} else {
				t.Errorf("query event=%T", event)
			}
		}
		if busy != 1 {
			t.Error("query was not rejected")
		}
		ag.UpdateCompactionConfig(&compaction.Config{Enabled: false})
		cancel()
		return nil
	})}})
	if err != nil {
		t.Fatal(err)
	}
	out, err := ag.CommitCompactionHistory(ctx, source, candidate, compaction.Result{Compacted: true, Trigger: "preflight", Watermark: "emergency_trim"})
	if err != nil || !out.Compacted || out.CheckpointID == "" || out.CheckpointID != saved.CheckpointID || ag.compactionGeneration.Load() != 1 || !reflect.DeepEqual(ag.Messages(), want) {
		t.Fatalf("acknowledged publication result=%+v error=%v", out, err)
	}
	if err := saved.Validate(); err != nil {
		t.Fatalf("saved checkpoint=%v", err)
	}
	if ag.hasCompactor || ag.turnActive.Load() {
		t.Fatal("runtime/admission release order failed")
	}
	if err := ag.ClearHistoryChecked(); err != nil {
		t.Fatalf("ownership leaked: %v", err)
	}
}

func TestCompactionHistoryMemoryOnlyAndLegacyPairParity(t *testing.T) {
	for _, writer := range []bool{false, true} {
		newAgent := func() *Agent {
			cfg := &compaction.Config{Enabled: writer}
			if writer {
				cfg.CheckpointWriter = compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error { return nil })
			}
			ag, err := New(Config{LLM: &countingCompactionModel{}, Compaction: cfg})
			if err != nil {
				t.Fatal(err)
			}
			return ag
		}
		old, combined := newAgent(), newAgent()
		candidate := []llm.Message{llm.NewUserMessage("candidate")}
		res := compaction.Result{Compacted: true, Trigger: "preflight", Watermark: "emergency_trim"}
		legacy, err := old.CommitCompactionCheckpoint(context.Background(), candidate, res)
		if err != nil {
			t.Fatal(err)
		}
		if err := old.ReplaceHistoryChecked(candidate); err != nil {
			t.Fatal(err)
		}
		got, err := combined.CommitCompactionHistory(context.Background(), []llm.Message{}, candidate, res)
		if err != nil || !reflect.DeepEqual(legacy, got) || !reflect.DeepEqual(old.Messages(), combined.Messages()) {
			t.Fatalf("writer=%v parity error=%v", writer, err)
		}
		if !writer && got.CheckpointID != "" {
			t.Fatal("memory-only publication invented a checkpoint")
		}
		if combined.compactionGeneration.Load() != 1 {
			t.Fatal("successful compaction was not counted")
		}
		if !writer {
			res.CheckpointID, res.CheckpointMessages = "caller-supplied-id", 99
			out, err := combined.CommitCompactionHistory(context.Background(), candidate, candidate, res)
			if err != nil || out.CheckpointID != "" || out.CheckpointMessages != 0 {
				t.Fatalf("memory-only commit retained stale identity: %+v error=%v", out, err)
			}
		}
	}
}

func TestCompactionHistoryPreservesLedgerAndHistoryOnFailure(t *testing.T) {
	for _, mode := range []string{"ledger", "ledger_busy", "writer", "writer_busy", "success"} {
		t.Run(mode, func(t *testing.T) {
			failure := errors.New("injected persistence error")
			if mode == "writer_busy" || mode == "ledger_busy" {
				failure = ErrAgentBusy
			}
			store := &checkpointLedgerStore{}
			if mode == "ledger" || mode == "ledger_busy" {
				store.failSaveAt = map[int]error{1: failure}
			}
			writes := 0
			ag, err := New(Config{LLM: &countingCompactionModel{}, InitialMessages: []llm.Message{llm.NewUserMessage("source"), llm.NewAssistantMessage("answer", nil)}, Compaction: &compaction.Config{Enabled: true, SessionID: "publication-ledger", LedgerStore: store, CheckpointWriter: compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
				writes++
				if mode == "writer" || mode == "writer_busy" {
					return failure
				}
				return nil
			})}})
			if err != nil {
				t.Fatal(err)
			}
			source := ag.Messages()
			candidate, res, err := ag.compactor.Compact(context.Background(), ag.llm, source)
			if err != nil {
				t.Fatal(err)
			}
			out, err := ag.CommitCompactionHistory(context.Background(), source, candidate, res)
			ledger, _ := store.snapshot()
			if mode == "success" {
				if err != nil || !out.Compacted || writes != 1 || ledger == nil || ledger.Summary == nil || !reflect.DeepEqual(candidate, ag.Messages()) {
					t.Fatalf("commit result=%v writes=%d error=%v", out.Compacted, writes, err)
				}
			} else {
				if !errors.Is(err, failure) || out.Compacted || !reflect.DeepEqual(source, ag.Messages()) || (ledger != nil && ledger.Summary != nil) {
					t.Fatalf("failure mode=%s result=%v error=%v", mode, out.Compacted, err)
				}
				if (mode == "ledger" || mode == "ledger_busy") && writes != 0 {
					t.Fatal("checkpoint attempted after ledger failure")
				}
				if err == ErrAgentBusy {
					t.Fatal("persistence cause escaped as a pre-I/O admission sentinel")
				}
			}
			if ag.turnActive.Load() || ag.compactionInFlight.Load() {
				t.Fatal("failed to release operation")
			}
		})
	}
}
