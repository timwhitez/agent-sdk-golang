package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// BlockStop describes a boundary after accepted calls have been terminalized.
type BlockStop string

const (
	BlockContinue         BlockStop = ""
	BlockDone             BlockStop = "task_complete_tail"
	BlockSteering         BlockStop = "steering"
	BlockRootBeforeStart  BlockStop = "root_cancel_before_start"
	BlockRootAfterHandler BlockStop = "root_cancel_after_handler"
	BlockRootAfterDone    BlockStop = "root_cancel_after_task_complete"
)

// BlockTerminal is a projection, not a second lifecycle authority. History is
// the scoped result record: a host child adapter must not append it to a parent
// model's history without a corresponding assistant call. Publish=false keeps
// native tails history-only.
type BlockTerminal struct {
	// ExecutionKnowledge is populated by the owner from its existing state.
	// Adapter-supplied values are ignored.
	ExecutionKnowledge string
	History            llm.Message
	Visible, Original  string
	Metadata           map[string]any
	Publish            bool
	Reason             string
}

func (t BlockTerminal) projection() toolResultProjection {
	return toolResultProjection{history: t.History, visible: t.Visible, original: t.Original, metadata: t.Metadata, publish: t.Publish}
}
func blockTerminal(p toolResultProjection, reason string) BlockTerminal {
	return BlockTerminal{History: p.history, Visible: p.visible, Original: p.original, Metadata: p.metadata, Publish: p.publish, Reason: reason}
}

// BlockAdmission contains the complete prepared Handler. Finish releases the
// host's attempt context, including when the final root gate rejects execution.
// Skipped supplies a pre-execution terminal and must not also supply a Call.
type BlockAdmission struct {
	Call    *tools.PreparedCall
	Context context.Context
	Deps    *tools.Container
	Finish  func() bool
	Skipped *BlockTerminal
}

// BlockOutcome retains the observed result separately from cancellation and
// panic. NotStarted is set only for tail projection; no handler ran for it.
type BlockOutcome struct {
	Content        llm.Content
	Err, RootError error
	Metadata       map[string]any
	Duration       time.Duration
	Interrupted    bool
	Panic          any
	NotStarted     BlockStop
}

// SequentialBlockAdapter keeps host policy and presentation at their existing
// boundaries. Callbacks are synchronous; Commit/Publish are never retried.
// Project must not execute tools. Commit failures can be indeterminate: a
// callback error cannot prove its write or the tool effect did not happen.
type SequentialBlockAdapter struct {
	Admit    func(context.Context, int) (BlockAdmission, error)
	Project  func(int, BlockOutcome) (BlockTerminal, error)
	Commit   func([]BlockTerminal) error
	Publish  func(int, BlockTerminal, time.Duration)
	Boundary func(int, BlockOutcome) BlockStop
	OnPanic  func(int, context.Context, any) (llm.Content, error)

	// parallel is an in-package opt-in for bounded waves. No production
	// adapter sets it yet: nil keeps every call Exclusive and sequential.
	parallel *blockParallelism
}

// blockParallelism lets consecutive eligible calls run in one bounded wave.
// Workers only execute an owned PreparedCall and return its outcome; the
// single owner keeps per-ordinal result slots and runs the existing
// terminal/projection/commit/publication path strictly in model order.
type blockParallelism struct {
	// maxWorkers bounds concurrently running handlers and the wave's result
	// slots; values below 2 disable waves.
	maxWorkers int
	// eligible reports, before admission, whether call i may share a wave.
	// It must come from a trusted capability; false keeps the call Exclusive
	// and ends the wave, acting as a barrier.
	eligible func(int) bool
}

type sequentialScopeKey struct{}
type sequentialScope struct {
	mu            sync.Mutex
	active, child bool
	childDone     chan struct{}
}

func (s *sequentialScope) finish() {
	s.mu.Lock()
	s.active = false
	done := s.childDone
	s.mu.Unlock()
	if done != nil {
		<-done
	}
}

// RunSequentialBlock runs a whole accepted block, with all calls Exclusive.
// It creates no goroutines and cannot be used to escape an active child scope.
func RunSequentialBlock(ctx context.Context, calls []llm.ToolCall, adapter SequentialBlockAdapter) (BlockStop, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if ctx.Value(sequentialScopeKey{}) != nil {
		return BlockContinue, errors.New("root tool block cannot replace an active execution scope")
	}
	state, err := newToolBlockState(calls)
	if err != nil {
		return BlockContinue, err
	}
	return runSequentialBlock(ctx, state, calls, adapter, false)
}

// RunSequentialChildBlock explicitly delegates from the currently executing
// Handler. Concurrent, nested and expired scope use is rejected before accepting
// calls. Without a parent scope it runs a standalone host block using the same
// owner. Child records do not acquire native Frame identity or parent history.
func RunSequentialChildBlock(ctx context.Context, calls []llm.ToolCall, adapter SequentialBlockAdapter) (BlockStop, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	scope, _ := ctx.Value(sequentialScopeKey{}).(*sequentialScope)
	if scope != nil {
		scope.mu.Lock()
		if !scope.active || scope.child || scope.childDone != nil {
			scope.mu.Unlock()
			return BlockContinue, errors.New("child tool block requires an active, unoccupied parent scope")
		}
		done := make(chan struct{})
		scope.childDone = done
		scope.mu.Unlock()
		defer func() { scope.mu.Lock(); scope.childDone = nil; close(done); scope.mu.Unlock() }()
	}
	state, err := newToolBlockState(calls)
	if err != nil {
		return BlockContinue, err
	}
	return runSequentialBlock(ctx, state, calls, adapter, true)
}

func runSequentialBlock(root context.Context, state *toolBlockState, calls []llm.ToolCall, a SequentialBlockAdapter, child bool) (stop BlockStop, err error) {
	if a.Admit == nil || a.Project == nil || a.Commit == nil {
		return BlockContinue, errors.New("tool block adapter is incomplete")
	}
	var finish func() bool
	defer func() {
		if recovered := recover(); recovered != nil {
			err = errors.New("tool block adapter failed; effects or publication may have occurred")
		}
		if finish != nil {
			pending := finish
			finish = nil
			func() {
				defer func() {
					if recover() != nil {
						err = errors.New("tool stage cleanup failed; effects may have occurred")
					}
				}()
				pending()
			}()
		}
		if err == nil {
			err = state.validateClosed()
		}
		if err != nil {
			rows := state.abortOpen()
			if len(rows) > 0 {
				terminals := make([]BlockTerminal, 0, len(rows))
				for _, call := range state.calls {
					i := len(terminals)
					if i < len(rows) && call.id == rows[i].ToolCallID {
						terminals = append(terminals, BlockTerminal{History: rows[i], Reason: "lifecycle_failure", ExecutionKnowledge: call.executionKnowledge.String()})
					}
				}
				// Only newly unclosed records are committed. Never retry the terminal
				// whose commit/publication returned an error or panicked.
				func() {
					defer func() {
						if recover() != nil {
							err = errors.Join(err, errors.New("tool block recovery commit failed"))
						}
					}()
					err = errors.Join(err, a.Commit(terminals))
				}()
			}
		}
	}()
	commit := func(index int, phase toolCallPhase, terminals []BlockTerminal) error {
		projections := make([]toolResultProjection, len(terminals))
		for i, t := range terminals {
			projections[i] = t.projection()
		}
		reason := "handler_return"
		if len(terminals) > 0 {
			reason = terminals[0].Reason
		}
		history, e := state.acceptResults(index, phase, reason, projections)
		if e != nil {
			return e
		}
		owned := make([]BlockTerminal, len(terminals))
		for i, t := range terminals {
			owned[i] = t
			owned[i].History = history[i]
			owned[i].ExecutionKnowledge = state.calls[index+i].executionKnowledge.String()
			owned[i].Metadata = cloneToolResultMetadata(t.Metadata)
		}
		return a.Commit(owned)
	}
	publish := func(index int, t BlockTerminal, duration time.Duration) error {
		if !t.Publish {
			return nil
		}
		p, e := state.takePublication(index)
		if e != nil {
			return e
		}
		if a.Publish != nil {
			terminal := blockTerminal(p, t.Reason)
			terminal.ExecutionKnowledge = state.calls[index].executionKnowledge.String()
			a.Publish(index, terminal, duration)
		}
		return nil
	}
	tail := func(start int, reason BlockStop) error {
		terminals := make([]BlockTerminal, 0, len(calls)-start)
		for i := start; i < len(calls); i++ {
			t, e := a.Project(i, BlockOutcome{NotStarted: reason, RootError: root.Err()})
			if e != nil {
				return e
			}
			t.Reason = string(reason)
			terminals = append(terminals, t)
		}
		if len(terminals) > 0 {
			if e := commit(start, toolCallAccepted, terminals); e != nil {
				return e
			}
			for i, t := range terminals {
				if e := publish(start+i, t, 0); e != nil {
					return e
				}
			}
		}
		return nil
	}
	// execute runs one owned PreparedCall. It is the only work a wave worker
	// performs: it never touches state, history, events or the adapter.
	execute := func(admission BlockAdmission, ctx context.Context, scope *sequentialScope) (o BlockOutcome) {
		started := time.Now()
		defer func() { o.Duration = time.Since(started) }()
		defer scope.finish()
		defer func() { o.Panic = recover() }()
		o.Content, o.Err = admission.Call.Execute(ctx, admission.Deps)
		return o
	}
	begin := func(i int, admission BlockAdmission) (context.Context, *sequentialScope, error) {
		if e := state.markRunning(i); e != nil {
			return nil, nil, e
		}
		ctx := admission.Context
		if ctx == nil {
			ctx = root
		}
		scope := &sequentialScope{active: true, child: child}
		return context.WithValue(ctx, sequentialScopeKey{}, scope), scope, nil
	}
	// settle is the owner's in-order completion of one started call.
	// release stays owned by the caller's cleanup until it is invoked here, so
	// a panicking OnPanic still leaves the stage to the deferred cleanup.
	settle := func(i int, ctx context.Context, release *func() bool, outcome BlockOutcome) (BlockTerminal, BlockOutcome, error) {
		if outcome.Panic != nil {
			outcome.Err = fmt.Errorf("tool handler panicked; effects may have occurred")
			if a.OnPanic != nil {
				outcome.Content, outcome.Err = a.OnPanic(i, ctx, outcome.Panic)
			}
		}
		if release != nil && *release != nil {
			f := *release
			*release = nil
			outcome.Interrupted = f()
		}
		outcome.RootError = root.Err()
		if e := state.markAttemptReturned(i, outcome.RootError != nil); e != nil {
			return BlockTerminal{}, outcome, e
		}
		outcome.Metadata = tools.TakeToolResultMetadataSnapshot(ctx)
		terminal, e := a.Project(i, outcome)
		if e != nil {
			return terminal, outcome, e
		}
		if e := commit(i, toolCallRunning, []BlockTerminal{terminal}); e != nil {
			return terminal, outcome, e
		}
		if e := publish(i, terminal, outcome.Duration); e != nil {
			return terminal, outcome, e
		}
		return terminal, outcome, nil
	}
	// after applies the post-commit root gate and Boundary for call i.
	after := func(i int, terminal BlockTerminal, outcome BlockOutcome) BlockStop {
		if root.Err() != nil {
			if terminal.Reason == "task_complete" {
				return BlockRootAfterDone
			}
			return BlockRootAfterHandler
		}
		if a.Boundary != nil {
			return a.Boundary(i, outcome)
		}
		return BlockContinue
	}
	waveSize := func(i int) int {
		p := a.parallel
		if p == nil || p.maxWorkers < 2 || p.eligible == nil {
			return 1
		}
		n := 0
		for i+n < len(calls) && n < p.maxWorkers && p.eligible(i+n) {
			n++
		}
		if n < 2 {
			return 1
		}
		return n
	}
	for i := 0; i < len(calls); {
		if n := waveSize(i); n > 1 {
			var next int
			var stop BlockStop
			var e error
			next, stop, e, finish = runOrderedWave(root, i, n, a, state, execute, begin, settle, after, commit, publish)
			if e != nil {
				return BlockContinue, e
			}
			if stop != BlockContinue {
				return stop, tail(next, stop)
			}
			i = next
			continue
		}
		if root.Err() != nil {
			return BlockRootBeforeStart, tail(i, BlockRootBeforeStart)
		}
		admission, e := a.Admit(root, i)
		// A failed host admission may still have allocated an attempt stage.
		finish = admission.Finish
		if e != nil {
			return BlockContinue, e
		}
		if admission.Skipped != nil {
			if admission.Call != nil || finish != nil {
				return BlockContinue, errors.New("invalid skipped tool admission")
			}
			if e := commit(i, toolCallAccepted, []BlockTerminal{*admission.Skipped}); e != nil {
				return BlockContinue, e
			}
			if e := publish(i, *admission.Skipped, 0); e != nil {
				return BlockContinue, e
			}
			i++
			continue
		}
		if admission.Call == nil {
			return BlockContinue, errors.New("tool admission is missing prepared call")
		}
		// Root authority, not steering's child context: preserve the final gate
		// after host events and stage creation, immediately before markRunning.
		if root.Err() != nil {
			if finish != nil {
				f := finish
				finish = nil
				f()
			}
			return BlockRootBeforeStart, tail(i, BlockRootBeforeStart)
		}
		ctx, scope, e := begin(i, admission)
		if e != nil {
			return BlockContinue, e
		}
		outcome := execute(admission, ctx, scope)
		terminal, outcome, e := settle(i, ctx, &finish, outcome)
		if e != nil {
			return BlockContinue, e
		}
		if reason := after(i, terminal, outcome); reason != BlockContinue {
			return reason, tail(i+1, reason)
		}
		i++
	}
	return BlockContinue, nil
}

// waveCompletionSettled is a test-only observation point, called after the
// owner has processed one worker completion (settling whatever became ready).
var waveCompletionSettled func(index int)

type waveSlot struct {
	index     int
	admission BlockAdmission
	ctx       context.Context
	release   func() bool
	skipped   bool
	started   bool
	done      bool
	outcome   BlockOutcome
}

type waveCompletion struct {
	slot    int
	outcome BlockOutcome
}

// runOrderedWave admits calls [start, start+n) in order, runs the started
// handlers concurrently (at most n), and settles each call in model order as
// soon as it and every earlier call have returned. It always waits for every
// started worker before returning, so no handler outlives the owner. It
// returns the first index that was neither started nor skipped, a stop reason
// and any lifecycle error; a pending host stage (finish) is returned for the
// caller's cleanup when admission failed.
func runOrderedWave(
	root context.Context, start, n int, a SequentialBlockAdapter, state *toolBlockState,
	execute func(BlockAdmission, context.Context, *sequentialScope) BlockOutcome,
	begin func(int, BlockAdmission) (context.Context, *sequentialScope, error),
	settle func(int, context.Context, *func() bool, BlockOutcome) (BlockTerminal, BlockOutcome, error),
	after func(int, BlockTerminal, BlockOutcome) BlockStop,
	commit func(int, toolCallPhase, []BlockTerminal) error,
	publish func(int, BlockTerminal, time.Duration) error,
) (next int, stop BlockStop, err error, pendingFinish func() bool) {
	slots := make([]waveSlot, 0, n)
	completions := make(chan waveCompletion, n)
	var workers sync.WaitGroup
	running := 0
	defer func() {
		// Never abandon a started handler: wait, then release any stage the
		// in-order settle did not reach.
		workers.Wait()
		for _, slot := range slots {
			if slot.release != nil {
				func() {
					defer func() { _ = recover() }()
					slot.release()
				}()
			}
		}
	}()

	stop = BlockContinue
	admitted := 0
	for k := 0; k < n; k++ {
		i := start + k
		if root.Err() != nil {
			stop = BlockRootBeforeStart
			break
		}
		admission, e := a.Admit(root, i)
		if e != nil {
			return start + admitted, BlockContinue, e, admission.Finish
		}
		slot := waveSlot{index: i, admission: admission, release: admission.Finish}
		if admission.Skipped != nil {
			if admission.Call != nil || admission.Finish != nil {
				return start + admitted, BlockContinue, errors.New("invalid skipped tool admission"), admission.Finish
			}
			slot.skipped, slot.done = true, true
			slots = append(slots, slot)
			admitted++
			continue
		}
		if admission.Call == nil {
			return start + admitted, BlockContinue, errors.New("tool admission is missing prepared call"), admission.Finish
		}
		if root.Err() != nil {
			if admission.Finish != nil {
				admission.Finish()
			}
			stop = BlockRootBeforeStart
			break
		}
		ctx, scope, e := begin(i, admission)
		if e != nil {
			return start + admitted, BlockContinue, e, admission.Finish
		}
		slot.ctx, slot.started = ctx, true
		slots = append(slots, slot)
		position := len(slots) - 1
		running++
		workers.Add(1)
		go func() {
			defer workers.Done()
			completions <- waveCompletion{slot: position, outcome: execute(admission, ctx, scope)}
		}()
		admitted++
	}

	settled := 0
	settleReady := func() error {
		for settled < len(slots) && slots[settled].done {
			slot := &slots[settled]
			var terminal BlockTerminal
			outcome := slot.outcome
			if slot.skipped {
				terminal = *slot.admission.Skipped
				if e := commit(slot.index, toolCallAccepted, []BlockTerminal{terminal}); e != nil {
					return e
				}
				if e := publish(slot.index, terminal, 0); e != nil {
					return e
				}
			} else {
				var e error
				terminal, outcome, e = settle(slot.index, slot.ctx, &slot.release, outcome)
				if e != nil {
					return e
				}
			}
			settled++
			// Started calls always settle with their real outcome; a stop
			// only prevents later calls from being admitted.
			if stop == BlockContinue {
				stop = after(slot.index, terminal, outcome)
			}
		}
		return nil
	}
	if e := settleReady(); e != nil {
		return start + admitted, BlockContinue, e, nil
	}
	for running > 0 {
		completion := <-completions
		running--
		slot := &slots[completion.slot]
		if slot.done {
			return start + admitted, BlockContinue, &toolBlockTransitionError{slot.index, "duplicate_completion"}, nil
		}
		slot.outcome, slot.done = completion.outcome, true
		if e := settleReady(); e != nil {
			return start + admitted, BlockContinue, e, nil
		}
		if waveCompletionSettled != nil {
			waveCompletionSettled(slot.index)
		}
	}
	return start + admitted, stop, nil, nil
}
