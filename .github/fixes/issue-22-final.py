from pathlib import Path

Path("sdk/agent/turn_concurrency.go").write_text(r'''package agent

import "errors"

// ErrAgentBusy reports that a second turn was submitted while this Agent still
// owns an active turn. Agent history, compaction, steering, and cancellation
// state form one turn-scoped state machine and are not shared concurrently.
var ErrAgentBusy = errors.New("agent: another query is already in progress")
''')

agent = Path("sdk/agent/agent.go")
text = agent.read_text()
field_old = '''\tturnCancelMu    sync.Mutex
\tturnCancelByOut map[chan Event]*turnBackpressure
}
'''
field_new = '''\t// turnActive is acquired synchronously before the turn goroutine starts,
\t// preventing overlapping submissions from mutating shared turn state.
\tturnActive      atomic.Bool
\tturnCancelMu    sync.Mutex
\tturnCancelByOut map[chan Event]*turnBackpressure
}
'''
if text.count(field_old) != 1:
    raise SystemExit(f"turn field anchor count={text.count(field_old)}")
text = text.replace(field_old, field_new)
start_old = '''\tout := make(chan Event, bufferSize)
\tunregisterTurn := a.registerTurnCancellation(out, ctx)
\tgo func() {
\t\tdefer close(out)
\t\tdefer unregisterTurn()
\t\ta.cleanupToolResultDumps(toolResultDumpNow(), false)
'''
start_new = '''\tout := make(chan Event, bufferSize)
\tif !a.turnActive.CompareAndSwap(false, true) {
\t\t// Admission is synchronous and out is buffered, so callers receive a
\t\t// deterministic terminal rejection without scheduling another goroutine.
\t\tout <- ErrorEvent{Kind: "agent_busy", Message: ErrAgentBusy.Error()}
\t\tclose(out)
\t\treturn out
\t}
\tunregisterTurn := a.registerTurnCancellation(out, ctx)
\tgo func() {
\t\t// Later defers execute first. By the time admission is released, all turn
\t\t// cleanup and cancellation bookkeeping have completed.
\t\tdefer close(out)
\t\tdefer a.turnActive.Store(false)
\t\tdefer unregisterTurn()
\t\ta.cleanupToolResultDumps(toolResultDumpNow(), false)
'''
if text.count(start_old) != 1:
    raise SystemExit(f"turn start anchor count={text.count(start_old)}")
agent.write_text(text.replace(start_old, start_new))

Path("sdk/agent/agent_turn_concurrency_test.go").write_text(r'''package agent

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type singleActiveTurnModel struct {
	calls       atomic.Int32
	firstStart  chan struct{}
	firstFinish chan struct{}
	startOnce   sync.Once
}

func (m *singleActiveTurnModel) Provider() string { return "stub" }
func (m *singleActiveTurnModel) Model() string    { return "stub" }
func (m *singleActiveTurnModel) Invoke(ctx context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	call := m.calls.Add(1)
	if call == 1 {
		m.startOnce.Do(func() { close(m.firstStart) })
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-m.firstFinish:
		}
	}
	return &llm.Completion{Content: llm.TextContent("ok")}, nil
}

func collectTurnEvents(t *testing.T, events <-chan Event) []Event {
	t.Helper()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	var out []Event
	for {
		select {
		case event, ok := <-events:
			if !ok {
				return out
			}
			out = append(out, event)
		case <-deadline.C:
			t.Fatal("timed out waiting for Agent event stream to close")
		}
	}
}

func isBusyTurn(events []Event) bool {
	for _, event := range events {
		if failure, ok := event.(ErrorEvent); ok && failure.Kind == "agent_busy" {
			return strings.Contains(failure.Message, ErrAgentBusy.Error())
		}
	}
	return false
}

func TestAgentRejectsOverlappingTurnWithoutMutatingHistory(t *testing.T) {
	model := &singleActiveTurnModel{firstStart: make(chan struct{}), firstFinish: make(chan struct{})}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatal(err)
	}
	first := ag.QueryStream(context.Background(), llm.TextContent("first"))
	select {
	case <-model.firstStart:
	case <-time.After(5 * time.Second):
		t.Fatal("first provider invocation did not start")
	}
	secondEvents := collectTurnEvents(t, ag.QueryStream(context.Background(), llm.TextContent("second")))
	if !isBusyTurn(secondEvents) {
		t.Fatalf("second turn events did not contain agent_busy: %#v", secondEvents)
	}
	if got := model.calls.Load(); got != 1 {
		t.Fatalf("provider calls while first turn active = %d, want 1", got)
	}
	for _, message := range ag.Messages() {
		if message.Role == llm.RoleUser && message.PlainText() == "second" {
			t.Fatal("rejected overlapping input was appended to history")
		}
	}
	close(model.firstFinish)
	firstEvents := collectTurnEvents(t, first)
	foundFinal := false
	for _, event := range firstEvents {
		if _, ok := event.(FinalResponseEvent); ok {
			foundFinal = true
		}
	}
	if !foundFinal {
		t.Fatalf("first turn did not complete normally: %#v", firstEvents)
	}
	thirdEvents := collectTurnEvents(t, ag.QueryStream(context.Background(), llm.TextContent("third")))
	if isBusyTurn(thirdEvents) {
		t.Fatalf("turn admission was not released: %#v", thirdEvents)
	}
	if got := model.calls.Load(); got != 2 {
		t.Fatalf("provider calls after third turn = %d, want 2", got)
	}
}

func TestAgentConcurrentSubmissionsAdmitAtMostOneTurn(t *testing.T) {
	model := &singleActiveTurnModel{firstStart: make(chan struct{}), firstFinish: make(chan struct{})}
	ag, err := New(Config{LLM: model})
	if err != nil {
		t.Fatal(err)
	}
	const callers = 16
	start := make(chan struct{})
	streams := make(chan (<-chan Event), callers)
	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			streams <- ag.QueryStream(context.Background(), llm.TextContent("parallel"))
		}()
	}
	close(start)
	wg.Wait()
	close(streams)
	select {
	case <-model.firstStart:
	case <-time.After(5 * time.Second):
		t.Fatal("admitted provider invocation did not start")
	}
	var active <-chan Event
	busyCount := 0
	for stream := range streams {
		select {
		case event, ok := <-stream:
			if !ok {
				continue
			}
			if failure, ok := event.(ErrorEvent); ok && failure.Kind == "agent_busy" {
				busyCount++
				for range stream {
				}
				continue
			}
			active = stream
		default:
			active = stream
		}
	}
	if got := model.calls.Load(); got != 1 {
		t.Fatalf("provider calls before release = %d, want 1", got)
	}
	if busyCount != callers-1 {
		t.Fatalf("busy rejections = %d, want %d", busyCount, callers-1)
	}
	if active == nil {
		t.Fatal("no active stream was admitted")
	}
	close(model.firstFinish)
	collectTurnEvents(t, active)
}
''')

docs = Path("agent_docs/architecture.md")
source = docs.read_text()
heading = "## Agent Turn Concurrency Contract"
if heading not in source:
    source = source.rstrip() + r'''

## Agent Turn Concurrency Contract

A single `Agent` owns one mutable conversation and permits exactly one active
`QueryStream` / `QueryStreamWithSteering` turn. Admission is acquired before the
turn goroutine starts. An overlapping submission receives an
`ErrorEvent{Kind: "agent_busy"}` and does not append input, invoke a provider,
or replace steering/cancellation state. Callers needing parallel turns must use
separate `Agent` instances.
'''
    docs.write_text(source)
