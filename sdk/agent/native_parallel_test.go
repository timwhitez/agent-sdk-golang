package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// turnModel answers with fixed completions, one per request.
type turnModel struct {
	mu    sync.Mutex
	turns []*llm.Completion
	next  int
}

func (*turnModel) Provider() string { return "fixture" }
func (*turnModel) Model() string    { return "turns" }
func (m *turnModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.next >= len(m.turns) {
		return &llm.Completion{Content: llm.TextContent("done")}, nil
	}
	c := m.turns[m.next]
	m.next++
	return c, nil
}

func readCalls(paths ...string) *llm.Completion {
	c := &llm.Completion{StopReason: "tool_calls"}
	for i, p := range paths {
		args, _ := json.Marshal(map[string]any{"file_path": p})
		c.ToolCalls = append(c.ToolCalls, llm.ToolCall{ID: fmt.Sprintf("call-%d", i), Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: string(args)}})
	}
	return c
}

// gatedRead is a read tool whose handler reports its start and waits for an
// explicit release, then acknowledges its completion.
type gatedRead struct {
	started, returned chan string
	gates             map[string]chan struct{}
	active, peak      atomic.Int32
	calls             atomic.Int32
	mu                sync.Mutex
}

func newGatedRead(paths ...string) *gatedRead {
	g := &gatedRead{started: make(chan string, 16), returned: make(chan string, 16), gates: map[string]chan struct{}{}}
	for _, p := range paths {
		g.gates[p] = make(chan struct{})
	}
	return g
}

func (g *gatedRead) tool() tools.Tool {
	type args struct {
		FilePath string `json:"file_path"`
	}
	return tools.Func[args]("read", "read a file", func(ctx context.Context, a args, _ *tools.Container) (any, error) {
		g.calls.Add(1)
		now := g.active.Add(1)
		for {
			peak := g.peak.Load()
			if now <= peak || g.peak.CompareAndSwap(peak, now) {
				break
			}
		}
		defer g.active.Add(-1)
		g.started <- a.FilePath
		g.mu.Lock()
		gate := g.gates[a.FilePath]
		g.mu.Unlock()
		var err error
		select {
		case <-gate:
		case <-ctx.Done():
			err = ctx.Err()
		}
		defer func() { g.returned <- a.FilePath }()
		if err != nil {
			return nil, err
		}
		return "contents of " + a.FilePath, nil
	})
}

func (g *gatedRead) release(path string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	select {
	case <-g.gates[path]:
	default:
		close(g.gates[path])
	}
}

func (g *gatedRead) releaseAll() {
	for p := range g.gates {
		g.release(p)
	}
}

func wait[T any](t *testing.T, ch <-chan T, what string) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
		var zero T
		return zero
	}
}

func readOnlyPlan(in ToolCallPlanInput) BlockCallPlan {
	return BlockCallPlan{Concurrent: in.Tool == "read"}
}

// runQuery runs a Query in the background and returns its events; cleanup
// always releases the gates and waits for the Query to finish.
func runQuery(t *testing.T, a *Agent, g *gatedRead, ctx context.Context) <-chan []Event {
	t.Helper()
	out := make(chan []Event, 1)
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		var events []Event
		for e := range a.QueryStream(ctx, llm.TextContent("go")) {
			events = append(events, e)
		}
		out <- events
	}()
	t.Cleanup(func() { g.releaseAll(); <-exited })
	return out
}

// #85: with the host's plan, declared reads on distinct targets run in one
// native wave: handlers start together and finish 3→1→2, while tool results
// are committed and published once each in model order.
func TestNativeWaveRunsDeclaredReadsConcurrently(t *testing.T) {
	g := newGatedRead("a.txt", "b.txt", "c.txt")
	a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls("a.txt", "b.txt", "c.txt")}}, Tools: []tools.Tool{g.tool()},
		ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	events := runQuery(t, a, g, context.Background())
	for i := 0; i < 3; i++ {
		wait(t, g.started, "wave start")
	}
	for _, p := range []string{"c.txt", "a.txt", "b.txt"} {
		g.release(p)
		if got := wait(t, g.returned, p); got != p {
			t.Fatalf("completed %s, want %s", got, p)
		}
	}
	var results []string
	for _, e := range wait(t, events, "query end") {
		if r, ok := e.(ToolResultEvent); ok {
			results = append(results, r.ToolCallID)
		}
	}
	if strings.Join(results, ",") != "call-0,call-1,call-2" || g.peak.Load() != 3 {
		t.Fatalf("results=%v peak=%d", results, g.peak.Load())
	}
	var history []string
	for _, m := range a.Messages() {
		if m.Role == llm.RoleTool {
			history = append(history, m.ToolCallID+"="+m.Content.PlainText())
		}
	}
	if strings.Join(history, ";") != "call-0=contents of a.txt;call-1=contents of b.txt;call-2=contents of c.txt" {
		t.Fatalf("history=%v", history)
	}
}

// Without the option, and for calls the SDK keeps Exclusive (a target shared
// with an earlier call, an undeclared tool, a plan that panics), handlers run
// one at a time in model order.
func TestNativeWaveNarrowsToExclusive(t *testing.T) {
	cases := []struct {
		name  string
		par   *ToolParallelism
		paths []string
	}{
		{"option unset", nil, []string{"a.txt", "b.txt", "c.txt"}},
		// Three spellings of one evidence target (distinct gates).
		{"same target", &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}, []string{"a.txt", "./a.txt", "sub/../a.txt"}},
		{"plan declines", &ToolParallelism{MaxWorkers: 4, Plan: func(ToolCallPlanInput) BlockCallPlan { return BlockCallPlan{} }}, []string{"a.txt", "b.txt", "c.txt"}},
		{"plan panics", &ToolParallelism{MaxWorkers: 4, Plan: func(ToolCallPlanInput) BlockCallPlan { panic("plan") }}, []string{"a.txt", "b.txt", "c.txt"}},
		{"one worker", &ToolParallelism{MaxWorkers: 1, Plan: readOnlyPlan}, []string{"a.txt", "b.txt", "c.txt"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := newGatedRead(tc.paths...)
			a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls(tc.paths...)}}, Tools: []tools.Tool{g.tool()}, ToolParallelism: tc.par})
			if err != nil {
				t.Fatal(err)
			}
			events := runQuery(t, a, g, context.Background())
			for _, p := range tc.paths {
				if got := wait(t, g.started, "call start"); got != p {
					t.Fatalf("started %s, want %s alone and in order", got, p)
				}
				g.release(p)
				wait(t, g.returned, p)
			}
			wait(t, events, "query end")
			if g.peak.Load() != 1 {
				t.Fatalf("peak=%d, want Exclusive", g.peak.Load())
			}
		})
	}
}

// A tool that is not evidence-family stays Exclusive even when the host
// declares it Concurrent, and the plan never sees an unknown tool.
func TestNativeWaveKeepsNonEvidenceAndUnknownToolsExclusive(t *testing.T) {
	var planned []string
	var mu sync.Mutex
	g := newGatedRead("a.txt", "b.txt")
	note := tools.Func[struct{}]("note", "writes a note", func(context.Context, struct{}, *tools.Container) (any, error) { return "noted", nil })
	c := readCalls("a.txt", "b.txt")
	c.ToolCalls = append([]llm.ToolCall{
		{ID: "note-0", Type: "function", Function: llm.FunctionCall{Name: "note", Arguments: `{}`}},
		{ID: "ghost-0", Type: "function", Function: llm.FunctionCall{Name: "ghost", Arguments: `{}`}},
	}, c.ToolCalls...)
	a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{c}}, Tools: []tools.Tool{g.tool(), note},
		ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: func(in ToolCallPlanInput) BlockCallPlan {
			mu.Lock()
			planned = append(planned, in.Tool)
			mu.Unlock()
			return BlockCallPlan{Concurrent: true}
		}}})
	if err != nil {
		t.Fatal(err)
	}
	events := runQuery(t, a, g, context.Background())
	wait(t, g.started, "read a")
	wait(t, g.started, "read b") // the two reads share a wave after the Exclusive calls
	g.releaseAll()
	wait(t, g.returned, "a")
	wait(t, g.returned, "b")
	wait(t, events, "query end")
	mu.Lock()
	defer mu.Unlock()
	for _, name := range planned {
		if name != "read" {
			t.Fatalf("plan consulted for %q: %v", name, planned)
		}
	}
	if g.peak.Load() != 2 {
		t.Fatalf("peak=%d", g.peak.Load())
	}
}

// Steering interrupts every handler of a running wave, not only the last
// admitted one.
func TestNativeWaveSteeringInterruptsEveryCall(t *testing.T) {
	g := newGatedRead("a.txt", "b.txt", "c.txt")
	a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls("a.txt", "b.txt", "c.txt")}}, Tools: []tools.Tool{g.tool()},
		ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	steering := make(chan SteeringMsg, 1)
	out := make(chan struct{})
	go func() {
		defer close(out)
		for range a.QueryStreamWithSteering(context.Background(), llm.TextContent("go"), steering) {
		}
	}()
	t.Cleanup(func() { g.releaseAll(); <-out })
	for i := 0; i < 3; i++ {
		wait(t, g.started, "wave start")
	}
	steering <- SteeringMsg{Content: "change of plan"}
	if !a.InterruptActiveStageForSteering() {
		t.Fatal("no stage interrupted")
	}
	for i := 0; i < 3; i++ {
		wait(t, g.returned, "interrupted call")
	}
	<-out
	interrupted := 0
	for _, m := range a.Messages() {
		if m.Role == llm.RoleTool && strings.Contains(m.Content.PlainText(), context.Canceled.Error()) {
			interrupted++
		}
	}
	if interrupted != 3 {
		t.Fatalf("interrupted results=%d", interrupted)
	}
}

// Deterministic planning check: calls on one evidence target carry the same
// SDK resource, so no wave can hold two of them; distinct targets do not.
func TestNativePlanAddsEvidenceTargetResource(t *testing.T) {
	a, err := New(Config{LLM: &turnModel{}, ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	read := newGatedRead().tool()
	exact := map[string]tools.Tool{"read": read}
	plan := func(path string) BlockCallPlan {
		args, _ := json.Marshal(map[string]any{"file_path": path})
		return a.planNativeCall(0, prepareNativeCall(llm.ToolCall{ID: "x", Function: llm.FunctionCall{Name: "read", Arguments: string(args)}}, exact, exact))
	}
	resource := func(p BlockCallPlan) string {
		if !p.Concurrent || len(p.Resources) != 1 {
			t.Fatalf("plan=%+v", p)
		}
		return p.Resources[0]
	}
	if resource(plan("a.txt")) != resource(plan("sub/../a.txt")) || resource(plan("a.txt")) == resource(plan("b.txt")) {
		t.Fatal("evidence target resource does not identify the target")
	}
	if p := a.planNativeCall(0, prepareNativeCall(llm.ToolCall{ID: "x", Function: llm.FunctionCall{Name: "read", Arguments: `{"file_path":`}}, exact, exact)); p.Concurrent {
		t.Fatal("invalid arguments planned concurrent")
	}
	_ = errors.New
}

// Each call of a wave keeps its own state: a read and an ls in one wave,
// finishing in reverse order, each commit their own tool name and result.
func TestNativeWaveKeepsEachCallsOwnState(t *testing.T) {
	g := newGatedRead("a.txt")
	lsStarted, lsGate := make(chan struct{}, 1), make(chan struct{})
	ls := tools.Func[struct {
		Path string `json:"path"`
	}]("ls", "list", func(ctx context.Context, _ struct {
		Path string `json:"path"`
	}, _ *tools.Container) (any, error) {
		lsStarted <- struct{}{}
		<-lsGate
		return "listing", nil
	})
	c := readCalls("a.txt")
	c.ToolCalls = append(c.ToolCalls, llm.ToolCall{ID: "call-ls", Type: "function", Function: llm.FunctionCall{Name: "ls", Arguments: `{"path":"dir"}`}})
	a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{c}}, Tools: []tools.Tool{g.tool(), ls},
		ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: func(ToolCallPlanInput) BlockCallPlan { return BlockCallPlan{Concurrent: true} }}})
	if err != nil {
		t.Fatal(err)
	}
	events := runQuery(t, a, g, context.Background())
	wait(t, g.started, "read start")
	wait(t, lsStarted, "ls start")
	close(lsGate)
	g.release("a.txt")
	wait(t, g.returned, "read end")
	wait(t, events, "query end")
	var rows []string
	for _, m := range a.Messages() {
		if m.Role == llm.RoleTool {
			rows = append(rows, m.ToolCallID+":"+m.ToolName+"="+m.Content.PlainText())
		}
	}
	if strings.Join(rows, ";") != "call-0:read=contents of a.txt;call-ls:ls=listing" {
		t.Fatalf("history=%v", rows)
	}
}
