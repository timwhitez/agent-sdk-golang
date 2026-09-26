package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// The SDK labels each error tool result with the path that produced it, and
// leaves successful results unlabeled. The label is a path, not a cause.
func TestToolResultErrorOriginNamesTheProducingPath(t *testing.T) {
	handler := func(content llm.Content, err error) tools.Tool {
		return tools.Tool{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
			return content, err
		}}
	}
	cases := []struct {
		name    string
		tool    tools.Tool
		call    string
		want    string
		isError bool
	}{
		{name: "success", tool: handler(llm.TextContent("ok"), nil), call: "work", want: "", isError: false},
		{name: "handler error", tool: handler(llm.TextContent("boom"), errors.New("boom")), call: "work", want: ToolErrorOriginHandler, isError: true},
		{name: "task complete", tool: handler(llm.Content{}, &tools.TaskCompleteError{Message: "done"}), call: "work", want: "", isError: false},
		{name: "handler panic", tool: tools.Tool{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
			panic("tool bug")
		}}, call: "work", want: ToolErrorOriginHandler, isError: true},
		// The invalid-tool fallback answers a name the Agent does not offer,
		// even though the fallback's own handler succeeds.
		{name: "unknown tool", tool: handler(llm.TextContent("ok"), nil), call: "invented_tool", want: ToolErrorOriginUnknownTool, isError: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ag, err := New(Config{LLM: &stubModel{toolName: tc.call, toolArgs: `{}`, toolID: "call-1"}, Tools: []tools.Tool{tc.tool}, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			result := onlyToolResult(t, collectEvents(ag.QueryStream(context.Background(), llm.TextContent("go"))))
			if result.IsError != tc.isError || result.ErrorOrigin != tc.want {
				t.Fatalf("tool result is_error=%v origin=%q, want %v/%q", result.IsError, result.ErrorOrigin, tc.isError, tc.want)
			}
		})
	}
}

// A root cancellation after the handler started replaces its result; the
// origin says so even though the handler also returned an error.
func TestToolResultErrorOriginRootCancellation(t *testing.T) {
	started := make(chan struct{})
	tool := tools.Tool{Name: "work", Handler: func(ctx context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
		close(started)
		<-ctx.Done()
		return llm.TextContent("stopped"), ctx.Err()
	}}
	ag, err := New(Config{LLM: &stubModel{toolName: "work", toolArgs: `{}`, toolID: "call-1"}, Tools: []tools.Tool{tool}, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream := ag.QueryStream(ctx, llm.TextContent("go"))
	go func() {
		<-started
		cancel()
	}()
	result := onlyToolResult(t, collectEvents(stream))
	if !result.IsError || result.ErrorOrigin != ToolErrorOriginCanceled {
		t.Fatalf("tool result is_error=%v origin=%q, want canceled", result.IsError, result.ErrorOrigin)
	}
}

// Accepted steering interrupts a running handler; its error is labeled as
// interrupted, not as the handler's own failure.
func TestToolResultErrorOriginSteeringInterrupt(t *testing.T) {
	started := make(chan struct{})
	tool := tools.Tool{Name: "work", Handler: func(ctx context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
		close(started)
		<-ctx.Done()
		return llm.TextContent("stopped"), ctx.Err()
	}}
	ag, err := New(Config{LLM: &stubModel{toolName: "work", toolArgs: `{}`, toolID: "call-1"}, Tools: []tools.Tool{tool}, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	steering := make(chan SteeringMsg, 1)
	stream := ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("go"), steering)
	go func() {
		<-started
		steering <- SteeringMsg{Content: "change of plan"}
		ag.InterruptActiveStageForSteering()
	}()
	var events []Event
	for envelope := range stream {
		events = append(events, envelope.Event)
	}
	result := onlyToolResult(t, events)
	if !result.IsError || result.ErrorOrigin != ToolErrorOriginInterrupted {
		t.Fatalf("tool result is_error=%v origin=%q, want interrupted", result.IsError, result.ErrorOrigin)
	}
}

func onlyToolResult(t *testing.T, events []Event) ToolResultEvent {
	t.Helper()
	var results []ToolResultEvent
	for _, event := range events {
		if result, ok := event.(ToolResultEvent); ok {
			results = append(results, result)
		}
	}
	if len(results) != 1 {
		t.Fatalf("tool results=%d want 1 (%#v)", len(results), results)
	}
	return results[0]
}

// Steering that arrives after a wave call returned its own error must not
// relabel it: the call waits to settle behind an earlier, still running call,
// and only that earlier call was interrupted.
func TestToolResultErrorOriginLateSteeringKeepsHandler(t *testing.T) {
	type args struct {
		FilePath string `json:"file_path"`
	}
	slowStarted, fastReturned := make(chan struct{}), make(chan struct{})
	read := tools.Func[args]("read", "read", func(ctx context.Context, a args, _ *tools.Container) (any, error) {
		if a.FilePath == "slow.txt" {
			close(slowStarted)
			<-ctx.Done()
			return nil, ctx.Err()
		}
		defer close(fastReturned)
		return nil, errors.New("invalid argument")
	})
	ag, err := New(Config{LLM: &turnModel{turns: []*llm.Completion{readCalls("slow.txt", "fast.txt")}}, Tools: []tools.Tool{read}, Warningf: func(string, ...any) {},
		ToolParallelism: &ToolParallelism{MaxWorkers: 2, Plan: readOnlyPlan}})
	if err != nil {
		t.Fatal(err)
	}
	steering := make(chan SteeringMsg, 1)
	stream := ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("go"), steering)
	go func() {
		<-slowStarted
		<-fastReturned
		// The fast call's handler has returned; give its worker time to
		// leave Execute. A slow scheduler can only make this test fail
		// (label interrupted), never pass wrongly.
		time.Sleep(50 * time.Millisecond)
		steering <- SteeringMsg{Content: "change of plan"}
		ag.InterruptActiveStageForSteering()
	}()
	origins := map[string]string{}
	for envelope := range stream {
		if result, ok := envelope.Event.(ToolResultEvent); ok {
			origins[result.ToolCallID] = result.ErrorOrigin
		}
	}
	if origins["call-0"] != ToolErrorOriginInterrupted || origins["call-1"] != ToolErrorOriginHandler {
		t.Fatalf("origins=%v, want call-0 interrupted and call-1 handler", origins)
	}
}

// The precedence between the paths: a root cancellation outranks an unknown
// tool, and an unknown tool outranks a steering interruption. A host-provided
// "invalid" fallback that waits on its context puts both under test.
func TestToolResultErrorOriginPrecedence(t *testing.T) {
	for _, tc := range []struct {
		name  string
		steer bool
		want  string
	}{
		{name: "root cancel over unknown tool", want: ToolErrorOriginCanceled},
		{name: "unknown tool over steering", steer: true, want: ToolErrorOriginUnknownTool},
	} {
		t.Run(tc.name, func(t *testing.T) {
			started := make(chan struct{})
			invalid := tools.Tool{Name: "invalid", Handler: func(ctx context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
				close(started)
				<-ctx.Done()
				return llm.TextContent("stopped"), ctx.Err()
			}}
			ag, err := New(Config{LLM: &stubModel{toolName: "invented_tool", toolArgs: `{}`, toolID: "call-1"}, Tools: []tools.Tool{invalid}, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			steering := make(chan SteeringMsg, 1)
			stream := ag.QueryStreamEnvelopedWithSteering(ctx, llm.TextContent("go"), steering)
			go func() {
				<-started
				if tc.steer {
					steering <- SteeringMsg{Content: "change of plan"}
					ag.InterruptActiveStageForSteering()
				} else {
					cancel()
				}
			}()
			var events []Event
			for envelope := range stream {
				events = append(events, envelope.Event)
			}
			result := onlyToolResult(t, events)
			if !result.IsError || result.ErrorOrigin != tc.want {
				t.Fatalf("tool result is_error=%v origin=%q, want %q", result.IsError, result.ErrorOrigin, tc.want)
			}
		})
	}
}

// A handler that panics after steering canceled its context is labeled
// interrupted, like one that returns an error then.
func TestToolResultErrorOriginPanicAfterSteering(t *testing.T) {
	started := make(chan struct{})
	tool := tools.Tool{Name: "work", Handler: func(ctx context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
		close(started)
		<-ctx.Done()
		panic("handler bug on cancellation")
	}}
	ag, err := New(Config{LLM: &stubModel{toolName: "work", toolArgs: `{}`, toolID: "call-1"}, Tools: []tools.Tool{tool}, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	steering := make(chan SteeringMsg, 1)
	stream := ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("go"), steering)
	go func() {
		<-started
		steering <- SteeringMsg{Content: "change of plan"}
		ag.InterruptActiveStageForSteering()
	}()
	var events []Event
	for envelope := range stream {
		events = append(events, envelope.Event)
	}
	result := onlyToolResult(t, events)
	if !result.IsError || result.ErrorOrigin != ToolErrorOriginInterrupted {
		t.Fatalf("tool result is_error=%v origin=%q, want interrupted", result.IsError, result.ErrorOrigin)
	}
}
