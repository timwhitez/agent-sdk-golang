package agent

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type compactionCallbackModel struct{}

func (compactionCallbackModel) Provider() string { return "stub" }
func (compactionCallbackModel) Model() string    { return "stub" }
func (compactionCallbackModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{Content: llm.TextContent("ok")}, nil
}

func installPendingCompactionForCallbackTest(t *testing.T, writer compaction.CompactionCheckpointWriter) *Agent {
	t.Helper()
	agent, err := New(Config{
		LLM: compactionCallbackModel{},
		Compaction: &compaction.Config{
			Enabled:          true,
			ContextWindow:    4096,
			CheckpointWriter: writer,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	agent.mu.Lock()
	agent.messages = []llm.Message{llm.NewUserMessage("before")}
	agent.mu.Unlock()
	agent.pendingCompactionMu.Lock()
	agent.pendingCompaction = &pendingCompaction{
		messages:    []llm.Message{llm.NewUserMessage("after")},
		snapshotLen: 1,
		result:      compaction.Result{Compacted: true},
	}
	agent.pendingCompactionMu.Unlock()
	return agent
}

func TestApplyPendingCompactionWriterCanReadMessages(t *testing.T) {
	var agent *Agent
	writerEntered := make(chan struct{})
	writer := compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
		close(writerEntered)
		_ = agent.Messages()
		return nil
	})
	agent = installPendingCompactionForCallbackTest(t, writer)

	done := make(chan struct{})
	go func() {
		agent.applyPendingCompaction(nil)
		close(done)
	}()

	select {
	case <-writerEntered:
	case <-time.After(time.Second):
		t.Fatal("checkpoint writer was not entered")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("applyPendingCompaction deadlocked when writer called Messages")
	}
	messages := agent.Messages()
	if len(messages) != 1 || messages[0].Content.PlainText() != "after" {
		t.Fatalf("published messages = %#v", messages)
	}
}

func TestBlockedCheckpointWriterDoesNotBlockHistoryReads(t *testing.T) {
	writerEntered := make(chan struct{})
	releaseWriter := make(chan struct{})
	writer := compaction.CompactionCheckpointWriterFunc(func(context.Context, compaction.CompactionCheckpoint) error {
		close(writerEntered)
		<-releaseWriter
		return nil
	})
	agent := installPendingCompactionForCallbackTest(t, writer)

	done := make(chan struct{})
	go func() {
		agent.applyPendingCompaction(nil)
		close(done)
	}()
	select {
	case <-writerEntered:
	case <-time.After(time.Second):
		t.Fatal("checkpoint writer was not entered")
	}

	readDone := make(chan struct{})
	go func() {
		_ = agent.Messages()
		close(readDone)
	}()
	select {
	case <-readDone:
	case <-time.After(250 * time.Millisecond):
		t.Fatal("Messages blocked behind checkpoint persistence")
	}
	close(releaseWriter)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("applyPendingCompaction did not finish")
	}
}

func TestApplyPendingCompactionWarningCallbackCanReadMessages(t *testing.T) {
	var agent *Agent
	warningEntered := make(chan struct{})
	var once sync.Once
	created, err := New(Config{
		LLM: compactionCallbackModel{},
		Warningf: func(string, ...any) {
			once.Do(func() { close(warningEntered) })
			_ = agent.Messages()
		},
		Compaction: &compaction.Config{Enabled: true, ContextWindow: 4096},
	})
	if err != nil {
		t.Fatal(err)
	}
	agent = created
	agent.mu.Lock()
	agent.messages = []llm.Message{llm.NewUserMessage("short")}
	agent.mu.Unlock()
	agent.pendingCompactionMu.Lock()
	agent.pendingCompaction = &pendingCompaction{
		messages:    []llm.Message{llm.NewUserMessage("after")},
		snapshotLen: 2,
		result:      compaction.Result{Compacted: true},
	}
	agent.pendingCompactionMu.Unlock()

	done := make(chan struct{})
	go func() {
		agent.applyPendingCompaction(nil)
		close(done)
	}()
	select {
	case <-warningEntered:
	case <-time.After(time.Second):
		t.Fatal("warning callback was not entered")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("warning callback deadlocked while reading Messages")
	}
}

func TestApplyPendingCompactionTokenEstimatorCanReadMessages(t *testing.T) {
	var agent *Agent
	estimatorEntered := make(chan struct{})
	var once sync.Once
	created, err := New(Config{
		LLM: compactionCallbackModel{},
		Compaction: &compaction.Config{
			Enabled:       true,
			ContextWindow: 4096,
			TokenEstimator: func(text string) int {
				once.Do(func() { close(estimatorEntered) })
				_ = agent.Messages()
				if text == "" {
					return 0
				}
				return 1
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	agent = created
	agent.mu.Lock()
	agent.messages = []llm.Message{llm.NewUserMessage("before")}
	agent.mu.Unlock()
	agent.pendingCompactionMu.Lock()
	agent.pendingCompaction = &pendingCompaction{
		messages:    []llm.Message{llm.NewUserMessage("after")},
		snapshotLen: 1,
		result:      compaction.Result{Compacted: true},
	}
	agent.pendingCompactionMu.Unlock()

	done := make(chan struct{})
	go func() {
		agent.applyPendingCompaction(nil)
		close(done)
	}()
	select {
	case <-estimatorEntered:
	case <-time.After(time.Second):
		t.Fatal("token estimator was not entered")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("token estimator deadlocked while reading Messages")
	}
}
