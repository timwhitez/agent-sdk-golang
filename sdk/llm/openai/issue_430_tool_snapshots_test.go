package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func issue430ToolSnapshotSSE(delta, final string, terminalOnly bool) string {
	item := map[string]any{"id": "fc_issue430", "type": "function_call", "call_id": "call_issue430", "name": "inspect", "arguments": final}
	var events []any
	if !terminalOnly {
		events = append(events,
			map[string]any{"type": "response.output_item.added", "output_index": 0, "item": map[string]any{"id": "fc_issue430", "type": "function_call", "call_id": "call_issue430", "name": "inspect", "arguments": ""}},
			map[string]any{"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_issue430", "delta": delta},
			map[string]any{"type": "response.function_call_arguments.done", "output_index": 0, "item_id": "fc_issue430", "arguments": final},
			map[string]any{"type": "response.output_item.done", "output_index": 0, "item": item},
		)
	}
	events = append(events, map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_issue430", "status": "completed", "output": []any{item}}})
	var body strings.Builder
	for _, event := range events {
		b, _ := json.Marshal(event)
		body.WriteString("data: " + string(b) + "\n\n")
	}
	return body.String()
}

type issue430SnapshotTransport struct {
	mu       sync.Mutex
	first    string
	requests [][]byte
}

func (r *issue430SnapshotTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	b, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	r.requests = append(r.requests, b)
	n := len(r.requests)
	r.mu.Unlock()
	body := r.first
	if n > 1 {
		body = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"after\",\"status\":\"completed\",\"output_text\":\"done\"}}\n\n"
	}
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(body)), Request: req}, nil
}

func TestIssue430ResponsesToolFinalSnapshots(t *testing.T) {
	for _, tc := range []struct {
		name, delta, final      string
		terminalOnly, wantError bool
	}{
		{"matching-control", `{"value":1}`, `{"value":1}`, false, false},
		{"partial-prefix", `{"value":`, `{"value":1}`, false, false},
		{"conflicting-arguments", `{"value":1}`, `{"value":2}`, false, true},
		{"empty-final-conflict", `{"value":1}`, "", false, true},
		{"terminal-only", "", `{"value":1}`, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rt := &issue430SnapshotTransport{first: issue430ToolSnapshotSSE(tc.delta, tc.final, tc.terminalOnly)}
			c := &ResponsesClient{ModelName: "issue430", MaxRetries: 1, HTTPClient: &http.Client{Transport: rt}}
			stream, err := c.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("inspect")}})
			if err != nil {
				t.Fatal(err)
			}
			args, name := "", ""
			done := 0
			var streamErr error
			var state []llm.ProviderState
			for event := range stream {
				switch e := event.(type) {
				case llm.StreamToolCallDeltaEvent:
					args += e.ArgumentsDelta
					name += e.NameDelta
				case llm.StreamErrorEvent:
					streamErr = e.AsError()
				case llm.StreamDoneEvent:
					done++
				case llm.StreamProviderStateEvent:
					state = append(state, e.State...)
				}
			}
			if tc.wantError {
				if streamErr == nil || done != 0 {
					t.Fatalf("conflicting final accepted: args=%q done=%d error=%v state=%s", args, done, streamErr, state[0].Data)
				}
				return
			}
			if streamErr != nil || done != 1 || name != "inspect" || args != tc.final {
				t.Fatalf("final tool snapshot not reconciled: name=%q args=%q want=%q done=%d error=%v", name, args, tc.final, done, streamErr)
			}
		})
	}
}

func TestIssue430ResponsesToolArgsExecutionMatchesReplay(t *testing.T) {
	rt := &issue430SnapshotTransport{first: issue430ToolSnapshotSSE(`{"value":1}`, `{"value":2}`, false)}
	c := &ResponsesClient{ModelName: "issue430", MaxRetries: 1, HTTPClient: &http.Client{Transport: rt}}
	type input struct {
		Value int `json:"value"`
	}
	executed := -1
	tool := tools.Func[input]("inspect", "noop fixture", func(_ context.Context, in input, _ *tools.Container) (any, error) {
		executed = in.Value
		return "observed", nil
	})
	a, err := agent.New(agent.Config{LLM: c, Tools: []tools.Tool{tool}, MaxIterations: 2, InvokeRetryMaxAttempts: 1, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	_, queryErr := a.Query(context.Background(), "inspect")
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if executed < 0 {
		if queryErr == nil {
			t.Fatal("no execution and no protocol error")
		}
		return
	}
	if len(rt.requests) != 2 {
		t.Fatalf("requests=%d error=%v", len(rt.requests), queryErr)
	}
	var body struct {
		Input []struct{ Type, Arguments string } `json:"input"`
	}
	if err := json.Unmarshal(rt.requests[1], &body); err != nil {
		t.Fatal(err)
	}
	for _, item := range body.Input {
		if item.Type == "function_call" {
			var replay input
			if err := json.Unmarshal([]byte(item.Arguments), &replay); err != nil {
				t.Fatal(err)
			}
			if replay.Value != executed {
				t.Fatalf("same SSE disagrees: handler executed value=%d but next request replays value=%d", executed, replay.Value)
			}
			return
		}
	}
	t.Fatal("missing replay")
}

func issue430Wire(t *testing.T, events ...any) string {
	t.Helper()
	var body strings.Builder
	for _, event := range events {
		data, err := json.Marshal(event)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&body, "data: %s\n\n", data)
	}
	return body.String()
}

func TestIssue430SnapshotSourcesAndLosslessArguments(t *testing.T) {
	for _, source := range []string{"arguments.done", "output_item.done", "terminal"} {
		for _, object := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/object=%v", source, object), func(t *testing.T) {
				const final = `{"value":9007199254740993}`
				var args any = final
				if object {
					args = map[string]any{"value": json.Number("9007199254740993")}
				}
				item := map[string]any{"type": "function_call", "id": "fc_precise", "call_id": "call_precise", "name": "inspect", "arguments": args}
				events := []any{
					map[string]any{"type": "response.output_item.added", "output_index": 0, "item": map[string]any{"type": "function_call", "id": "fc_precise", "call_id": "call_precise", "name": "inspect", "arguments": ""}},
					map[string]any{"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_precise", "delta": `{"value":`},
				}
				if source == "arguments.done" {
					events = append(events, map[string]any{"type": "response.function_call_arguments.done", "output_index": 0, "item_id": "fc_precise", "arguments": args})
				}
				if source != "terminal" {
					events = append(events, map[string]any{"type": "response.output_item.done", "output_index": 0, "item": item})
				}
				events = append(events, map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_precise", "status": "completed", "output": []any{item}}})
				rt := &issue430SnapshotTransport{first: issue430Wire(t, events...)}
				c := &ResponsesClient{ModelName: "issue430", MaxRetries: 1, HTTPClient: &http.Client{Transport: rt}}
				stream, err := c.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
				if err != nil {
					t.Fatal(err)
				}
				got, name := "", ""
				done := 0
				for event := range stream {
					switch e := event.(type) {
					case llm.StreamToolCallDeltaEvent:
						got += e.ArgumentsDelta
						name += e.NameDelta
					case llm.StreamErrorEvent:
						t.Fatal(e.AsError())
					case llm.StreamDoneEvent:
						done++
					}
				}
				if got != final || name != "inspect" || done != 1 {
					t.Fatalf("name=%q args=%q done=%d", name, got, done)
				}
			})
		}
	}
}

func TestIssue430ConflictingIdentityAndFinalization(t *testing.T) {
	for _, changed := range []string{"name", "id", "call_id", "arguments", "late-delta"} {
		t.Run(changed, func(t *testing.T) {
			first := map[string]any{"type": "function_call", "id": "fc_fixed", "call_id": "call_fixed", "name": "inspect", "arguments": `{"value":1}`}
			last := map[string]any{"type": "function_call", "id": "fc_fixed", "call_id": "call_fixed", "name": "inspect", "arguments": `{"value":1}`}
			if changed != "late-delta" {
				last[changed] = "changed"
			}
			events := []any{map[string]any{"type": "response.output_item.added", "output_index": 0, "item": first}}
			if changed == "late-delta" {
				events = append(events,
					map[string]any{"type": "response.function_call_arguments.done", "output_index": 0, "item_id": "fc_fixed", "arguments": `{"value":1}`},
					map[string]any{"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_fixed", "delta": " "},
				)
			}
			events = append(events, map[string]any{"type": "response.completed", "response": map[string]any{"id": "r", "status": "completed", "output": []any{last}}})
			c := &ResponsesClient{ModelName: "issue430", MaxRetries: 1, HTTPClient: &http.Client{Transport: &issue430SnapshotTransport{first: issue430Wire(t, events...)}}}
			stream, err := c.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
			if err != nil {
				t.Fatal(err)
			}
			var observed error
			for event := range stream {
				switch e := event.(type) {
				case llm.StreamErrorEvent:
					observed = e.AsError()
				case llm.StreamDoneEvent:
					t.Fatal("conflicting tool snapshot reported success")
				}
			}
			var protocol *llm.ProviderError
			if !errors.As(observed, &protocol) {
				t.Fatalf("expected typed protocol failure, got %T %v", observed, observed)
			}
		})
	}
}

func TestIssue430CompletedPrefixExecutesOnce(t *testing.T) {
	rt := &issue430SnapshotTransport{first: issue430ToolSnapshotSSE(`{"value":`, `{"value":1}`, false)}
	c := &ResponsesClient{ModelName: "issue430", MaxRetries: 1, HTTPClient: &http.Client{Transport: rt}}
	type input struct {
		Value int `json:"value"`
	}
	calls := 0
	tool := tools.Func[input]("inspect", "noop", func(_ context.Context, in input, _ *tools.Container) (any, error) {
		calls++
		if in.Value != 1 {
			t.Errorf("value=%d", in.Value)
		}
		return "ok", nil
	})
	a, err := agent.New(agent.Config{LLM: c, Tools: []tools.Tool{tool}, MaxIterations: 2, InvokeRetryMaxAttempts: 1, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Query(context.Background(), "hello"); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("executions=%d want=1", calls)
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if len(rt.requests) != 2 {
		t.Fatalf("requests=%d", len(rt.requests))
	}
	var request struct {
		Input []struct{ Type, Arguments string } `json:"input"`
	}
	if err := json.Unmarshal(rt.requests[1], &request); err != nil {
		t.Fatal(err)
	}
	for _, item := range request.Input {
		if item.Type == "function_call" {
			if item.Arguments != `{"value":1}` {
				t.Fatalf("replayed arguments=%s", item.Arguments)
			}
			return
		}
	}
	t.Fatal("missing replayed function call")
}
