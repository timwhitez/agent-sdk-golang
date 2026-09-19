package agent

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func sequentialFixture(n int, handler func(context.Context, int) (any, error)) ([]llm.ToolCall, SequentialBlockAdapter) {
	calls := make([]llm.ToolCall, n)
	for i := range calls {
		calls[i] = llm.ToolCall{ID: fmt.Sprint("call-", i), Function: llm.FunctionCall{Name: "fixture", Arguments: "{}"}}
	}
	return calls, SequentialBlockAdapter{
		Admit: func(ctx context.Context, i int) (BlockAdmission, error) {
			tool := tools.Func[struct{}]("fixture", "fixture", func(ctx context.Context, _ struct{}, _ *tools.Container) (any, error) { return handler(ctx, i) })
			p, _ := tool.PrepareCall("{}")
			return BlockAdmission{Call: &p, Context: tools.WithToolResultMetadata(ctx)}, nil
		},
		Project: func(i int, o BlockOutcome) (BlockTerminal, error) {
			content := o.Content
			failed := o.Err != nil
			if o.NotStarted != "" {
				content = llm.TextContent("not started")
				failed = true
			}
			return BlockTerminal{History: llm.NewToolMessage(calls[i].ID, "fixture", content, failed), Visible: content.PlainText(), Original: content.PlainText(), Publish: o.NotStarted == "", Reason: "handler_return"}, nil
		},
		Commit: func([]BlockTerminal) error { return nil },
	}
}

func TestSequentialOwnerOrdersExecutionCommitAndPublication(t *testing.T) {
	var trace []string
	calls, a := sequentialFixture(2, func(_ context.Context, i int) (any, error) {
		trace = append(trace, fmt.Sprint("execute", i))
		return "ok", nil
	})
	admit := a.Admit
	a.Admit = func(ctx context.Context, i int) (BlockAdmission, error) {
		trace = append(trace, fmt.Sprint("admit", i))
		return admit(ctx, i)
	}
	project := a.Project
	a.Project = func(i int, o BlockOutcome) (BlockTerminal, error) {
		trace = append(trace, fmt.Sprint("project", i))
		return project(i, o)
	}
	a.Commit = func(ts []BlockTerminal) error {
		for _, v := range ts {
			trace = append(trace, "commit"+v.History.ToolCallID)
		}
		return nil
	}
	a.Publish = func(i int, _ BlockTerminal, _ time.Duration) { trace = append(trace, fmt.Sprint("publish", i)) }
	a.Boundary = func(i int, _ BlockOutcome) BlockStop {
		trace = append(trace, fmt.Sprint("boundary", i))
		return BlockContinue
	}
	state, _ := newToolBlockState(calls)
	stop, err := runSequentialBlock(context.Background(), state, calls, a, false)
	want := []string{"admit0", "execute0", "project0", "commitcall-0", "publish0", "boundary0", "admit1", "execute1", "project1", "commitcall-1", "publish1", "boundary1"}
	if err != nil || stop != BlockContinue || !reflect.DeepEqual(trace, want) || state.validateClosed() != nil {
		t.Fatalf("stop=%s err=%v trace=%v state=%+v", stop, err, trace, state)
	}
	for _, c := range state.calls {
		if c.terminalCount != 1 || c.executionKnowledge != toolExecutionOutcomeObserved {
			t.Fatalf("state=%+v", c)
		}
	}
}

func TestSequentialOwnerDoesNotReplayFailedProjectionCommitOrPublication(t *testing.T) {
	for _, stage := range []string{"project", "commit", "publish"} {
		t.Run(stage, func(t *testing.T) {
			effects, publications := 0, 0
			commits := map[string]int{}
			calls, a := sequentialFixture(2, func(context.Context, int) (any, error) { effects++; return "effect", nil })
			project := a.Project
			a.Project = func(i int, o BlockOutcome) (BlockTerminal, error) {
				if stage == "project" && i == 0 {
					return BlockTerminal{}, errors.New("projection failed")
				}
				return project(i, o)
			}
			a.Commit = func(ts []BlockTerminal) error {
				for _, v := range ts {
					commits[v.History.ToolCallID]++
				}
				if stage == "commit" && ts[0].History.ToolCallID == calls[0].ID {
					return errors.New("commit outcome uncertain")
				}
				return nil
			}
			a.Publish = func(int, BlockTerminal, time.Duration) {
				publications++
				if stage == "publish" {
					panic("publication uncertain")
				}
			}
			state, _ := newToolBlockState(calls)
			_, err := runSequentialBlock(context.Background(), state, calls, a, false)
			if err == nil || effects != 1 || commits[calls[0].ID] != 1 || commits[calls[1].ID] != 1 || state.validateClosed() != nil {
				t.Fatalf("err=%v effects=%d commits=%v state=%+v", err, effects, commits, state)
			}
			if state.calls[0].executionKnowledge != toolExecutionOutcomeObserved || state.calls[1].executionKnowledge != toolExecutionNotStarted {
				t.Fatalf("knowledge=%+v", state.calls)
			}
			if stage != "publish" && publications != 0 || publications > 1 {
				t.Fatalf("publications=%d", publications)
			}
		})
	}
}

func TestSequentialOwnerFinalRootGateReleasesStageAndClosesTail(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	effects, finishes := 0, 0
	calls, a := sequentialFixture(2, func(context.Context, int) (any, error) { effects++; return "unsafe", nil })
	admit := a.Admit
	a.Admit = func(ctx context.Context, i int) (BlockAdmission, error) {
		v, e := admit(ctx, i)
		v.Finish = func() bool { finishes++; return true }
		cancel()
		return v, e
	}
	state, _ := newToolBlockState(calls)
	stop, err := runSequentialBlock(ctx, state, calls, a, false)
	if err != nil || stop != BlockRootBeforeStart || effects != 0 || finishes != 1 || state.validateClosed() != nil {
		t.Fatalf("stop=%s err=%v effects=%d finishes=%d", stop, err, effects, finishes)
	}
	for _, c := range state.calls {
		if c.executionKnowledge != toolExecutionNotStarted || c.terminalCount != 1 {
			t.Fatal(c)
		}
	}
}

func TestSequentialChildScopeRejectsConcurrentNestedAndExpiredUse(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	entered, release := make(chan struct{}), make(chan struct{})
	var retained context.Context
	rootCalls, a := sequentialFixture(1, func(parent context.Context, _ int) (any, error) {
		retained = parent
		childCalls, child := sequentialFixture(1, func(childCtx context.Context, _ int) (any, error) {
			nestedCalls, nested := sequentialFixture(1, func(context.Context, int) (any, error) { t.Error("nested executed"); return nil, nil })
			if _, err := RunSequentialChildBlock(childCtx, nestedCalls, nested); err == nil {
				t.Error("nested accepted")
			}
			close(entered)
			select {
			case <-release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			return "child", nil
		})
		done := make(chan error, 1)
		go func() { _, e := RunSequentialChildBlock(parent, childCalls, child); done <- e }()
		<-entered
		if _, err := RunSequentialChildBlock(parent, childCalls, child); err == nil {
			t.Error("concurrent accepted")
		}
		if _, err := RunSequentialBlock(parent, childCalls, child); err == nil {
			t.Error("root escaped active scope")
		}
		close(release)
		if err := <-done; err != nil {
			t.Error(err)
		}
		return "parent", nil
	})
	if _, err := RunSequentialBlock(ctx, rootCalls, a); err != nil {
		t.Fatal(err)
	}
	calls, next := sequentialFixture(1, func(context.Context, int) (any, error) { t.Error("expired executed"); return nil, nil })
	if _, err := RunSequentialChildBlock(retained, calls, next); err == nil {
		t.Fatal("expired accepted")
	}
}

func TestSequentialParentWaitsForAcceptedChildAfterHandlerReturns(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	entered, release, returned := make(chan struct{}), make(chan struct{}), make(chan struct{})
	calls, a := sequentialFixture(1, func(parent context.Context, _ int) (any, error) {
		childCalls, child := sequentialFixture(1, func(context.Context, int) (any, error) { close(entered); <-release; return "child", nil })
		go func() {
			if _, err := RunSequentialChildBlock(parent, childCalls, child); err != nil {
				t.Error(err)
			}
		}()
		<-entered
		close(returned)
		return "parent", nil
	})
	done := make(chan error, 1)
	go func() { _, err := RunSequentialBlock(ctx, calls, a); done <- err }()
	select {
	case <-returned:
	case <-ctx.Done():
		t.Fatal("parent did not return")
	}
	select {
	case <-done:
		t.Fatal("owner returned before child closed")
	default:
	}
	close(release)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal("owner failed to join child")
	}
}

func TestSequentialAdmissionErrorStillReleasesAllocatedStage(t *testing.T) {
	calls, a := sequentialFixture(1, func(context.Context, int) (any, error) { t.Fatal("handler ran"); return nil, nil })
	finishes := 0
	a.Admit = func(context.Context, int) (BlockAdmission, error) {
		return BlockAdmission{Finish: func() bool { finishes++; return false }}, errors.New("admission failed")
	}
	state, _ := newToolBlockState(calls)
	_, err := runSequentialBlock(context.Background(), state, calls, a, false)
	if err == nil || finishes != 1 || state.validateClosed() != nil || state.calls[0].executionKnowledge != toolExecutionNotStarted {
		t.Fatalf("err=%v finishes=%d state=%+v", err, finishes, state)
	}
}

func TestSequentialChildAdmissionRacesParentReturnWithoutEscapingLifetime(t *testing.T) {
	for attempt := 0; attempt < 30; attempt++ {
		var started, committed atomic.Int32
		childDone := make(chan error, 1)
		calls, a := sequentialFixture(1, func(parent context.Context, _ int) (any, error) {
			childCalls, child := sequentialFixture(1, func(context.Context, int) (any, error) { started.Add(1); return "child", nil })
			child.Commit = func([]BlockTerminal) error { committed.Add(1); return nil }
			go func() { _, err := RunSequentialChildBlock(parent, childCalls, child); childDone <- err }()
			return "parent", nil
		})
		a.Publish = func(int, BlockTerminal, time.Duration) {
			if started.Load() != committed.Load() {
				t.Error("parent published before accepted child committed")
			}
		}
		if _, err := RunSequentialBlock(context.Background(), calls, a); err != nil {
			t.Fatal(err)
		}
		err := <-childDone
		if err == nil && (started.Load() != 1 || committed.Load() != 1) || err != nil && started.Load() != 0 {
			t.Fatalf("err=%v started=%d committed=%d", err, started.Load(), committed.Load())
		}
	}
}

type delayedProjectionCodec struct{ artifact.JSONEnvelopeCodec }

func (c delayedProjectionCodec) Decode(text string) (artifact.Envelope, bool, error) {
	time.Sleep(20 * time.Millisecond)
	return c.JSONEnvelopeCodec.Decode(text)
}

func TestSequentialNativeDurationStillIncludesProjection(t *testing.T) {
	tool := tools.Func[struct{}]("fixture", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { return "ok", nil })
	model := &toolPlanScriptModel{toolCalls: []llm.ToolCall{{ID: "timed", Function: llm.FunctionCall{Name: "fixture", Arguments: "{}"}}}}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{tool}, ArtifactEnvelopeCodec: delayedProjectionCodec{}})
	if err != nil {
		t.Fatal(err)
	}
	var accounting, step int64
	for ev := range ag.QueryStream(context.Background(), llm.TextContent("run")) {
		switch ev := ev.(type) {
		case AccountingEvent:
			if ev.ToolCallID == "timed" {
				accounting = ev.DurationMS
			}
		case StepCompleteEvent:
			if ev.StepID == "timed" {
				step = ev.DurationMS
			}
		}
	}
	if accounting < 20 || step < accounting {
		t.Fatalf("accounting=%d step=%d; projection excluded", accounting, step)
	}
}

func TestSequentialNativeSuppressionEventOrder(t *testing.T) {
	for _, which := range []string{"repeat", "evidence"} {
		t.Run(which, func(t *testing.T) {
			done := tools.Func[evidenceDoneArgs]("done", "done", func(_ context.Context, a evidenceDoneArgs, _ *tools.Container) (any, error) {
				return nil, tools.TaskComplete(a.Message)
			})
			cfg := Config{EventBufferSize: 256, Warningf: func(string, ...any) {}, Tools: []tools.Tool{done}}
			var want []string
			if which == "repeat" {
				cfg.LLM = &repeatedInterventionBoundaryModel{}
				cfg.Tools = append(cfg.Tools, tools.Func[struct {
					Text string `json:"text"`
				}]("echo", "echo", func(context.Context, struct {
					Text string `json:"text"`
				}, *tools.Container) (any, error) { return "ok", nil }))
				cfg.RepeatToolSignatureThreshold = 2
				cfg.RepeatToolSignatureWindow = 4
				cfg.LoopGuardStrikeThreshold = 1
				cfg.LoopGuardUserMessage = "stop repeating"
				want = []string{"hidden", "warn:loop_guard", "warn:loop_guard", "start:echo-2", "call:echo-2", "result:echo-2", "step:echo-2"}
			} else {
				cfg.LLM = &evidenceFixtureModel{}
				read := tools.Func[evidenceReadArgs]("read", "read", func(_ context.Context, a evidenceReadArgs, _ *tools.Container) (any, error) {
					if a.Offset >= 101 {
						return "101: new block", nil
					}
					return "1: same block", nil
				})
				alias := read
				alias.Name = "read_file"
				cfg.Tools = append(cfg.Tools, read, alias)
				want = []string{"start:read-3", "call:read-3", "result:read-3", "step:read-3", "hidden", "warn:no_progress_recovery"}
			}
			ag, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			for ev := range ag.QueryStream(context.Background(), llm.TextContent("run")) {
				switch ev := ev.(type) {
				case HiddenUserMessageEvent:
					got = append(got, "hidden")
				case WarnEvent:
					got = append(got, "warn:"+ev.Kind)
				case StepStartEvent:
					got = append(got, "start:"+ev.StepID)
				case ToolCallEvent:
					got = append(got, "call:"+ev.ToolCallID)
				case ToolResultEvent:
					got = append(got, "result:"+ev.ToolCallID)
				case StepCompleteEvent:
					got = append(got, "step:"+ev.StepID)
				}
			}
			found := false
			for i := 0; i+len(want) <= len(got); i++ {
				if reflect.DeepEqual(got[i:i+len(want)], want) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("missing legacy sequence %v in %v", want, got)
			}
		})
	}
}
