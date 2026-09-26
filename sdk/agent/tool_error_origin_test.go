package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

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
