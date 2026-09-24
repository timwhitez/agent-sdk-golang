package agent

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
)

func planCall(t *testing.T, a *Agent, tool tools.Tool, args string) BlockCallPlan {
	t.Helper()
	exact := map[string]tools.Tool{tool.Name: tool}
	return a.planNativeCall(0, prepareNativeCall(llm.ToolCall{ID: "c", Function: llm.FunctionCall{Name: tool.Name, Arguments: args}}, exact, exact))
}

// codecArgs has a user decoder: preparing it must not run it, so the call has
// no final view and stays Exclusive.
type codecArgs struct {
	FilePath string `json:"file_path"`
}

var codecCalls atomic.Int32

func (c *codecArgs) UnmarshalJSON(data []byte) error {
	codecCalls.Add(1)
	type plain codecArgs
	return json.Unmarshal(data, (*plain)(c))
}

// N01: planning resolves no dependency, touches no file system and runs no
// user decoder, encoder or Handler, including for calls it keeps Exclusive.
func TestN01NativePlanRunsNoDependencyFileOrUserCode(t *testing.T) {
	var provided, handled atomic.Int32
	deps := tools.NewContainer()
	sb, err := sandbox.New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) { provided.Add(1); return sb, nil })
	a, err := New(Config{LLM: &turnModel{}, Deps: deps, ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	read := tools.Func[struct {
		FilePath string `json:"file_path"`
	}]("read", "read", func(context.Context, struct {
		FilePath string `json:"file_path"`
	}, *tools.Container) (any, error) {
		handled.Add(1)
		return "x", nil
	})
	custom := tools.Func[codecArgs]("read", "read", func(context.Context, codecArgs, *tools.Container) (any, error) {
		handled.Add(1)
		return "x", nil
	})
	codecCalls.Store(0)
	if p := planCall(t, a, read, `{"file_path":"a.txt"}`); !p.Concurrent {
		t.Fatalf("typed read plan=%+v", p)
	}
	if p := planCall(t, a, custom, `{"file_path":"a.txt"}`); p.Concurrent {
		t.Fatal("a tool whose arguments need a user decoder was planned concurrent")
	}
	if n := provided.Load(); n != 0 {
		t.Fatalf("planning resolved the sandbox dependency %d times", n)
	}
	if codecCalls.Load() != 0 || handled.Load() != 0 {
		t.Fatalf("planning ran user code: decoder=%d handler=%d", codecCalls.Load(), handled.Load())
	}
}

// N02/N03: the SDK resource comes from the arguments the Handler executes.
// Same canonical target with different aliases → one resource; the exact
// field keeps precedence; a typed schema-key repair plans the repaired value.
func TestN02N03NativePlanUsesExecutedArguments(t *testing.T) {
	a, err := New(Config{LLM: &turnModel{}, ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	read := newGatedRead().tool()
	resource := func(args string) string {
		p := planCall(t, a, read, args)
		if !p.Concurrent || len(p.Resources) != 1 {
			t.Fatalf("%s: plan=%+v", args, p)
		}
		return p.Resources[0]
	}
	sameA := resource(`{"file_path":"same.txt","filePath":"alias-a.txt"}`)
	sameB := resource(`{"file_path":"same.txt","filePath":"alias-b.txt"}`)
	if sameA != sameB || sameA != resource(`{"file_path":"same.txt"}`) {
		t.Fatalf("one executed target, different resources: %q %q", sameA, sameB)
	}
	if resource(`{"filePath":"alias-a.txt"}`) != resource(`{"file_path":"alias-a.txt"}`) {
		t.Fatal("an alias-only call is not planned on the target it executes")
	}
	if resource(`{"filepath":"rep.txt"}`) != resource(`{"file_path":"rep.txt"}`) {
		t.Fatal("a repaired key is not planned on the repaired value")
	}
	if resource(`{"file_path":"one.txt"}`) == resource(`{"file_path":"two.txt"}`) {
		t.Fatal("different executed targets share a resource")
	}
}

// N04: the host plan sees an owned copy of the executed arguments; changing
// it changes nothing that runs, and the host plan runs once per call even when
// a wave ends early and planning resumes from a later call.
func TestN04NativePlanViewIsOwnedAndPlannedOnce(t *testing.T) {
	var mu sync.Mutex
	planned := map[int]int{}
	g := newGatedRead("a.txt", "b.txt", "c.txt")
	a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls("a.txt", "b.txt", "c.txt")}}, Tools: []tools.Tool{g.tool()},
		ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: func(in ToolCallPlanInput) BlockCallPlan {
			mu.Lock()
			planned[in.Ordinal]++
			mu.Unlock()
			for i := range in.Arguments {
				in.Arguments[i] = 'X'
			}
			// The middle call is Exclusive, so the first wave ends at it.
			return BlockCallPlan{Concurrent: in.Ordinal != 1}
		}}})
	if err != nil {
		t.Fatal(err)
	}
	events := runQuery(t, a, g, context.Background())
	for i := 0; i < 3; i++ {
		g.release(wait(t, g.started, "call"))
		wait(t, g.returned, "call")
	}
	wait(t, events, "end")
	var rows []string
	for _, m := range a.Messages() {
		if m.Role == llm.RoleTool {
			rows = append(rows, m.Content.PlainText())
		}
	}
	if strings.Join(rows, ";") != "contents of a.txt;contents of b.txt;contents of c.txt" {
		t.Fatalf("results=%v", rows)
	}
	mu.Lock()
	defer mu.Unlock()
	if planned[0] != 1 || planned[1] != 1 || planned[2] != 1 {
		t.Fatalf("host plan calls per ordinal=%v", planned)
	}
}

// N02 through a real Agent: two calls on one executed target with different
// aliases never run together; the second starts only after the first ended.
func TestN02NativeWaveSerializesOneExecutedTarget(t *testing.T) {
	g := newGatedRead("same.txt")
	c := &llm.Completion{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{
		{ID: "x1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{"file_path":"same.txt","filePath":"alias-a.txt"}`}},
		{ID: "x2", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{"file_path":"same.txt","filePath":"alias-b.txt"}`}},
	}}
	a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{c}}, Tools: []tools.Tool{g.tool()},
		ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	events := runQuery(t, a, g, context.Background())
	wait(t, g.started, "first")
	g.release("same.txt")
	wait(t, g.returned, "first")
	wait(t, g.started, "second")
	wait(t, g.returned, "second")
	wait(t, events, "end")
	if g.peak.Load() != 1 {
		t.Fatalf("one executed target ran concurrently: peak=%d", g.peak.Load())
	}
}

// N03: a wrapper that rewrites the arguments it forwards cannot run inside a
// wave on arguments other than the planned ones: the call fails before the
// inner tool runs, and that tool is planned Exclusive afterwards.
func TestN03NativeWaveRefusesRewrittenArguments(t *testing.T) {
	var inner atomic.Int32
	type args struct {
		FilePath string `json:"file_path"`
	}
	base := tools.Func[args]("read", "read", func(_ context.Context, a args, _ *tools.Container) (any, error) {
		inner.Add(1)
		return "contents of " + a.FilePath, nil
	})
	rewriting := base
	rewriting.Handler = func(ctx context.Context, raw json.RawMessage, deps *tools.Container) (llm.Content, error) {
		return base.Handler(ctx, json.RawMessage(`{"file_path":"elsewhere.txt"}`), deps)
	}
	a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls("a.txt", "b.txt"), readCalls("c.txt", "d.txt")}}, Tools: []tools.Tool{rewriting},
		ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Query(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	var rows []string
	for _, m := range a.Messages() {
		if m.Role == llm.RoleTool {
			rows = append(rows, m.Content.PlainText())
		}
	}
	// First block ran as a wave: both refused before the inner tool ran.
	// Second block is Exclusive for this tool: the legacy path runs the
	// wrapper's rewritten arguments as before.
	if len(rows) != 4 || !strings.Contains(rows[0], "changed after") || !strings.Contains(rows[1], "changed after") ||
		rows[2] != "contents of elsewhere.txt" || rows[3] != "contents of elsewhere.txt" || inner.Load() != 2 {
		t.Fatalf("rows=%q inner=%d", rows, inner.Load())
	}
}

// N05: the worker bound holds for the native wave.
func TestN05NativeWaveWorkerBound(t *testing.T) {
	g := newGatedRead("a.txt", "b.txt", "c.txt")
	a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls("a.txt", "b.txt", "c.txt")}}, Tools: []tools.Tool{g.tool()},
		ToolParallelism: &ToolParallelism{MaxWorkers: 2, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	events := runQuery(t, a, g, context.Background())
	first, second := wait(t, g.started, "1"), wait(t, g.started, "2")
	g.release(first)
	g.release(second)
	wait(t, g.returned, "1")
	wait(t, g.returned, "2")
	g.release(wait(t, g.started, "3"))
	wait(t, g.returned, "3")
	wait(t, events, "end")
	if g.peak.Load() != 2 {
		t.Fatalf("peak=%d, want the bound 2", g.peak.Load())
	}
}

// N01 (declined plans): a host plan that declines is consulted without any
// dependency access either.
func TestN01DeclinedPlanRunsNoDependency(t *testing.T) {
	var provided atomic.Int32
	deps := tools.NewContainer()
	tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) { provided.Add(1); return nil, nil })
	var asked atomic.Int32
	a, err := New(Config{LLM: &turnModel{}, Deps: deps, ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: func(ToolCallPlanInput) BlockCallPlan {
		asked.Add(1)
		return BlockCallPlan{}
	}}})
	if err != nil {
		t.Fatal(err)
	}
	if p := planCall(t, a, newGatedRead().tool(), `{"file_path":"a.txt"}`); p.Concurrent || asked.Load() != 1 || provided.Load() != 0 {
		t.Fatalf("plan=%+v asked=%d provided=%d", p, asked.Load(), provided.Load())
	}
}

// N06: a handler panicking inside a native wave ends as that call's error
// result; the other call keeps its result, both in model order, and nothing
// is replayed.
func TestN06NativeWavePanicIsOneCallsError(t *testing.T) {
	var runs atomic.Int32
	type args struct {
		FilePath string `json:"file_path"`
	}
	started := make(chan string, 2)
	release := make(chan struct{})
	read := tools.Func[args]("read", "read", func(_ context.Context, a args, _ *tools.Container) (any, error) {
		runs.Add(1)
		started <- a.FilePath
		<-release
		if a.FilePath == "boom.txt" {
			panic("fixture panic")
		}
		return "contents of " + a.FilePath, nil
	})
	a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls("boom.txt", "ok.txt")}}, Tools: []tools.Tool{read}, Warningf: func(string, ...any) {},
		ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range a.QueryStream(context.Background(), llm.TextContent("go")) {
		}
	}()
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }); <-done })
	wait(t, started, "first")
	wait(t, started, "second") // both started: a real wave
	once.Do(func() { close(release) })
	<-done
	var rows []string
	for _, m := range a.Messages() {
		if m.Role == llm.RoleTool {
			rows = append(rows, m.ToolCallID+":"+m.Content.PlainText())
		}
	}
	if len(rows) != 2 || !strings.HasPrefix(rows[0], "call-0:") || !strings.Contains(rows[0], "panicked") || rows[1] != "call-1:contents of ok.txt" || runs.Load() != 2 {
		t.Fatalf("rows=%q runs=%d", rows, runs.Load())
	}
}

// N03R2: in a real native wave, planned targets a.txt and b.txt are both
// rewritten to same.txt by a wrapper re-entering through the public
// Tool.Execute. Both wrapper invocations are shown to run concurrently, and
// neither reaches the inner tool on the unplanned target.
func TestN03R2NativeWaveRefusesPublicReentryRewrite(t *testing.T) {
	var inner atomic.Int32
	type args struct {
		FilePath string `json:"file_path"`
	}
	base := tools.Func[args]("read", "read", func(_ context.Context, a args, _ *tools.Container) (any, error) {
		inner.Add(1)
		return "contents of " + a.FilePath, nil
	})
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	var releaseOnce sync.Once
	openGate := func() { releaseOnce.Do(func() { close(release) }) }
	rewriting := base
	rewriting.Handler = func(ctx context.Context, _ json.RawMessage, deps *tools.Container) (llm.Content, error) {
		started <- struct{}{}
		select {
		case <-release:
		case <-ctx.Done():
			return llm.Content{}, ctx.Err()
		}
		return base.Execute(ctx, `{"file_path":"same.txt"}`, deps)
	}
	a, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls("a.txt", "b.txt")}}, Tools: []tools.Tool{rewriting},
		ToolParallelism: &ToolParallelism{MaxWorkers: 4, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range a.QueryStream(context.Background(), llm.TextContent("go")) {
		}
	}()
	t.Cleanup(func() { openGate(); <-done })
	wait(t, started, "first wrapper")
	wait(t, started, "second wrapper") // both inside the wave at once
	openGate()
	<-done
	var rows []string
	for _, m := range a.Messages() {
		if m.Role == llm.RoleTool {
			rows = append(rows, m.Content.PlainText())
		}
	}
	if inner.Load() != 0 || len(rows) != 2 || !strings.Contains(rows[0], "changed after") || !strings.Contains(rows[1], "changed after") {
		t.Fatalf("inner=%d rows=%q", inner.Load(), rows)
	}
}
