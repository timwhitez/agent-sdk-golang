package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// #84: a projection failure after out-of-order returns stops settlement
// without re-running effects, waits for every started worker and closes each
// call exactly once.
func TestToolBlockParallelProjectionFailureWaitsForWorkers(t *testing.T) {
	g := newGatedHandlers(3)
	trace := &parallelTrace{}
	calls, a := parallelFixture(3, g, trace, 3, allEligible)
	project := a.Project
	projected := make(chan int, 3)
	a.Project = func(i int, o BlockOutcome) (BlockTerminal, error) {
		projected <- i
		if i == 0 && o.NotStarted == "" {
			return BlockTerminal{}, errors.New("projection failed")
		}
		return project(i, o)
	}
	state, _ := newToolBlockState(calls)
	done := make(chan error, 1)
	go func() {
		_, err := runSequentialBlock(context.Background(), state, calls, a, false)
		done <- err
	}()
	g.waitStarted(t, 3)
	close(g.release[0])
	if got := wait(t, projected, "failing projection"); got != 0 {
		t.Fatalf("projected %d first", got)
	}
	// Calls 1 and 2 are still running: the owner must not return yet.
	select {
	case err := <-done:
		t.Fatalf("owner returned while workers were running: %v", err)
	case <-time.After(30 * time.Millisecond):
	}
	close(g.release[2])
	close(g.release[1])
	if err := <-done; err == nil {
		t.Fatal("expected projection failure")
	}
	if len(g.returnedSnapshot()) != 3 {
		t.Fatalf("owner returned before workers: %v", g.returnedSnapshot())
	}
	requireSingleTerminal(t, state)
}

// #84: a slow event consumer holds publication of the first call while later
// calls finish; the owner neither publishes them early nor runs anything
// twice, and publishes in model order once the consumer catches up.
func TestToolBlockParallelSlowPublisherKeepsModelOrder(t *testing.T) {
	g := newGatedHandlers(3)
	trace := &parallelTrace{}
	calls, a := parallelFixture(3, g, trace, 3, allEligible)
	publish := a.Publish
	consumerReady := make(chan struct{})
	entered := make(chan int, 3)
	a.Publish = func(i int, term BlockTerminal, d time.Duration) {
		entered <- i
		if i == 0 {
			<-consumerReady
		}
		publish(i, term, d)
	}
	state, _ := newToolBlockState(calls)
	done := make(chan error, 1)
	go func() {
		_, err := runSequentialBlock(context.Background(), state, calls, a, false)
		done <- err
	}()
	g.waitStarted(t, 3)
	close(g.release[0])
	if got := wait(t, entered, "first publication"); got != 0 {
		t.Fatalf("first publication for call %d", got)
	}
	close(g.release[2])
	close(g.release[1])
	// Both later calls return while call 0's consumer is blocked.
	for len(g.returnedSnapshot()) < 3 {
		select {
		case i := <-entered:
			t.Fatalf("call %d published while call 0's consumer was blocked", i)
		case <-time.After(10 * time.Millisecond):
		}
	}
	close(consumerReady)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	trace.mu.Lock()
	published := append([]int(nil), trace.published...)
	trace.mu.Unlock()
	if fmt.Sprint(published) != "[0 1 2]" {
		t.Fatalf("published %v", published)
	}
	requireSingleTerminal(t, state)
}

func (g *gatedHandlers) returnedSnapshot() []int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]int(nil), g.returnedOrder...)
}

// recordingTurns answers fixed completions and records every request.
type recordingTurns struct {
	mu       sync.Mutex
	turns    []*llm.Completion
	requests [][]llm.Message
}

func (*recordingTurns) Provider() string { return "fixture" }
func (*recordingTurns) Model() string    { return "turns" }
func (m *recordingTurns) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, llm.CloneMessages(req.Messages))
	if len(m.requests) > len(m.turns) {
		return &llm.Completion{Content: llm.TextContent("finished")}, nil
	}
	return m.turns[len(m.requests)-1], nil
}

func mixedCalls(names ...string) *llm.Completion {
	c := &llm.Completion{StopReason: "tool_calls"}
	for i, spec := range names {
		name, arg := spec, ""
		if n, a, ok := strings.Cut(spec, ":"); ok {
			name, arg = n, a
		}
		args, _ := json.Marshal(map[string]string{"file_path": arg, "message": arg})
		c.ToolCalls = append(c.ToolCalls, llm.ToolCall{ID: fmt.Sprintf("call_%d", i), Type: "function", Function: llm.FunctionCall{Name: name, Arguments: string(args)}})
	}
	return c
}

type waveParityRun struct {
	history  []llm.Message
	requests [][]llm.Message
	events   []string
	executed []string
}

func runWaveParity(t *testing.T, parallel *ToolParallelism, turns []*llm.Completion) waveParityRun {
	t.Helper()
	var mu sync.Mutex
	var executed []string
	type readArgs struct {
		FilePath string `json:"file_path"`
	}
	read := tools.Func[readArgs]("read", "read a file", func(_ context.Context, a readArgs, _ *tools.Container) (any, error) {
		mu.Lock()
		executed = append(executed, "read:"+a.FilePath)
		mu.Unlock()
		return "contents of " + a.FilePath, nil
	})
	type doneArgs struct {
		Message string `json:"message"`
	}
	done := tools.Func[doneArgs]("done", "finish", func(_ context.Context, a doneArgs, _ *tools.Container) (any, error) {
		mu.Lock()
		executed = append(executed, "done:"+a.Message)
		mu.Unlock()
		return nil, &tools.TaskCompleteError{Message: a.Message}
	})
	model := &recordingTurns{turns: turns}
	a, err := New(Config{LLM: model, Tools: []tools.Tool{read, done}, ToolParallelism: parallel, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	var run waveParityRun
	for ev := range a.QueryStream(context.Background(), llm.TextContent("go")) {
		switch e := ev.(type) {
		case ToolCallEvent:
			run.events = append(run.events, "call:"+e.ToolCallID+":"+e.Tool)
		case ToolResultEvent:
			run.events = append(run.events, fmt.Sprintf("result:%s:%v:%s", e.ToolCallID, e.IsError, e.Result))
		case FinalResponseEvent:
			run.events = append(run.events, "final")
		}
	}
	run.history = a.Messages()
	run.requests = model.requests
	run.executed = executed
	return run
}

// #84 native acceptance: waves change only handler concurrency. Across blocks
// that reuse provider call IDs, a done call in the middle of a wave-eligible
// block, and a following block, the model's requests, the committed history,
// the delivered events and the executed effects equal the Exclusive run.
func TestNativeWaveMatchesExclusiveHistoryAcrossReuseAndDone(t *testing.T) {
	turns := func() []*llm.Completion {
		return []*llm.Completion{
			mixedCalls("read:a", "read:b", "read:c"),
			mixedCalls("read:d", "read:e"), // reuses call_0/call_1
			mixedCalls("read:f", "read:g", "done:finished", "read:h"),
		}
	}
	exclusive := runWaveParity(t, nil, turns())
	wave := runWaveParity(t, &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}, turns())
	if !reflect.DeepEqual(exclusive.history, wave.history) {
		t.Fatalf("history differs:\nexclusive=%+v\nwave=%+v", exclusive.history, wave.history)
	}
	if !reflect.DeepEqual(exclusive.requests, wave.requests) {
		t.Fatal("model requests differ between Exclusive and wave runs")
	}
	// A wave announces its calls at admission, before their results, so only
	// the interleaving differs: calls and results each keep model order, and
	// every result follows its own call.
	split := func(events []string) (calls, results []string) {
		for _, e := range events {
			if strings.HasPrefix(e, "call:") {
				calls = append(calls, e)
			} else {
				results = append(results, e)
			}
		}
		return calls, results
	}
	exclusiveCalls, exclusiveResults := split(exclusive.events)
	waveCalls, waveResults := split(wave.events)
	if !reflect.DeepEqual(exclusiveCalls, waveCalls) || !reflect.DeepEqual(exclusiveResults, waveResults) {
		t.Fatalf("events differ:\nexclusive=%v\nwave=%v", exclusive.events, wave.events)
	}
	open := map[string]bool{}
	for _, e := range wave.events {
		parts := strings.SplitN(e, ":", 3)
		switch parts[0] {
		case "call":
			open[parts[1]] = true
		case "result":
			if !open[parts[1]] {
				t.Fatalf("result for %s before its call: %v", parts[1], wave.events)
			}
			delete(open, parts[1])
		}
	}
	sortedEqual := func(a, b []string) bool {
		count := map[string]int{}
		for _, s := range a {
			count[s]++
		}
		for _, s := range b {
			count[s]--
		}
		for _, n := range count {
			if n != 0 {
				return false
			}
		}
		return len(a) == len(b)
	}
	if !sortedEqual(exclusive.executed, wave.executed) {
		t.Fatalf("executed effects differ: exclusive=%v wave=%v", exclusive.executed, wave.executed)
	}
	for _, e := range wave.executed {
		if e == "read:h" {
			t.Fatal("a call after done was executed")
		}
	}
}
