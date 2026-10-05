package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"testing"
	"time"
)

// The owner has actually received ordinal1's error before cancellation; it
// cannot settle that ordinal while ordinal0 is still executing.
func TestOriginalHandlerFailureBeforeOrderedCancellationSettlement(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	waveCompletionSettled = func(i int) {
		if i == 1 {
			cancel()
		}
	}
	defer func() { waveCompletionSettled = nil }()
	tool := tools.Func("read", "owned", func(ctx context.Context, a struct {
		FilePath string `json:"file_path"`
	}, _ *tools.Container) (any, error) {
		if a.FilePath == "slow.txt" {
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return nil, errors.New("owned independent failure")
	})
	ag, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls("slow.txt", "fast.txt")}}, Tools: []tools.Tool{tool}, Warningf: func(string, ...any) {}, ToolParallelism: &ToolParallelism{MaxWorkers: 2, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	events, receipt := ag.QueryStreamEnvelopedWithReceipt(ctx, llm.TextContent("owned"), nil)
	count := 0
	for envelope := range events {
		if r, ok := envelope.Event.(ToolResultEvent); ok {
			count++
			if r.ErrorOrigin != ToolErrorOriginCanceled {
				t.Fatalf("changed cancellation projection: %+v", r)
			}
			raw, _ := json.Marshal(r)
			var fields map[string]any
			_ = json.Unmarshal(raw, &fields)
			want := r.ToolCallID == "call-1"
			if got, _ := fields["HandlerFailed"].(bool); got != want {
				t.Fatalf("failure=%v want=%v result=%s", got, want, raw)
			}
		}
	}
	summary, ok := receipt.Summary()
	raw, _ := json.Marshal(summary)
	var fields map[string]any
	_ = json.Unmarshal(raw, &fields)
	if !ok || count != 2 || fields["HandlerFailures"] != float64(1) {
		t.Fatalf("summary=%s results=%d ok=%v", raw, count, ok)
	}
	assertContiguousToolResults(t, ag.Messages())
}

func TestOriginalHandlerEvidenceIndependentOfProjection(t *testing.T) {
	for _, tc := range []struct {
		name       string
		err        error
		panicValue any
		want       bool
	}{
		{name: "success"}, {name: "cancel", err: context.Canceled}, {name: "deadline", err: context.DeadlineExceeded},
		{name: "wrapped cancel", err: fmt.Errorf("owned: %w", context.Canceled)},
		{name: "wrapped deadline", err: fmt.Errorf("owned: %w", context.DeadlineExceeded)},
		{name: "done", err: &tools.TaskCompleteError{Message: "owned"}},
		{name: "joined controls", err: errors.Join(context.Canceled, context.DeadlineExceeded, &tools.TaskCompleteError{Message: "owned"})},
		{name: "failure", err: errors.New("dummy-private"), want: true},
		{name: "joined", err: errors.Join(errors.New("dummy-private"), context.Canceled), want: true},
		{name: "done plus failure", err: errors.Join(&tools.TaskCompleteError{Message: "owned"}, errors.New("dummy-private")), want: true},
		{name: "panic canceled", panicValue: context.Canceled, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tool := tools.Tool{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				if tc.panicValue != nil {
					panic(tc.panicValue)
				}
				return llm.Content{}, tc.err
			}}
			ag, err := New(Config{LLM: &stubModel{toolName: "work", toolArgs: `{}`, toolID: "call-1"}, Tools: []tools.Tool{tool}, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			stream, receipt := ag.QueryStreamEnvelopedWithReceipt(t.Context(), llm.TextContent("owned"), nil)
			var found *ToolResultEvent
			for e := range stream {
				if r, ok := e.Event.(ToolResultEvent); ok {
					found = &r
				}
			}
			summary, _ := receipt.Summary()
			wantCount := uint64(0)
			if tc.want {
				wantCount = 1
			}
			if found == nil || found.HandlerFailed != tc.want || summary.HandlerFailures != wantCount {
				t.Fatalf("result=%+v summary=%+v want=%v", found, summary, tc.want)
			}
		})
	}
}

func TestDroppedOriginalHandlerFailureStillInCloseReceipt(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	started := make(chan struct{})
	release := make(chan struct{})
	tool := tools.Func("work", "owned", func(context.Context, struct{}, *tools.Container) (any, error) {
		close(started)
		<-release
		cancel()
		return nil, errors.New("dummy-private")
	})
	ag, err := New(Config{LLM: &stubModel{toolName: "work", toolArgs: `{}`, toolID: "call-1"}, Tools: []tools.Tool{tool}, EventBufferSize: 1, EventSendTimeout: time.Millisecond, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	stream, receipt := ag.QueryStreamEnvelopedWithReceipt(ctx, llm.TextContent("owned"), nil)
	// Drain through the real handler's start, then leave the one-element channel
	// full. Cancellation permits bounded publication drops, not lost evidence.
	for envelope := range stream {
		if _, ok := envelope.Event.(ToolCallEvent); ok {
			break
		}
	}
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatal("handler did not start")
	}
	close(release)
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	for {
		if _, ok := receipt.Summary(); ok {
			break
		}
		select {
		case <-deadline.C:
			t.Fatal("stream did not close")
		case <-time.After(time.Millisecond):
		}
	}
	deliveredFailures := 0
	for e := range stream {
		if r, ok := e.Event.(ToolResultEvent); ok && r.HandlerFailed {
			deliveredFailures++
		}
	}
	summary, _ := receipt.Summary()
	if deliveredFailures != 0 || summary.HandlerFailures != 1 || summary.DroppedCriticalEvents == 0 {
		t.Fatalf("delivered failures=%d summary=%+v", deliveredFailures, summary)
	}
	clean, cleanReceipt := ag.QueryStreamEnvelopedWithReceipt(t.Context(), llm.TextContent("next"), nil)
	collectEnvelopes(clean)
	cleanSummary, _ := cleanReceipt.Summary()
	if cleanSummary.HandlerFailures != 0 {
		t.Fatalf("cross-query evidence contamination: %+v", cleanSummary)
	}
}

type originalMultiLeaf struct{ children []error }

func (originalMultiLeaf) Error() string     { return "owned non-control leaf" }
func (e originalMultiLeaf) Unwrap() []error { return e.children }

type originalCycle struct{}

func (*originalCycle) Error() string   { return "owned cycle" }
func (e *originalCycle) Unwrap() error { return e }
func TestOriginalHandlerMalformedErrorTreesStayBounded(t *testing.T) {
	for _, err := range []error{originalMultiLeaf{}, originalMultiLeaf{children: []error{nil, nil}}, &originalCycle{}} {
		if !independentHandlerError(err) {
			t.Fatalf("non-control/malformed original error erased: %T", err)
		}
	}
	deep := error(context.Canceled)
	for i := 0; i < 300; i++ {
		deep = fmt.Errorf("owned: %w", deep)
	}
	if !independentHandlerError(deep) {
		t.Fatal("budget exhaustion was not conservative")
	}
}
