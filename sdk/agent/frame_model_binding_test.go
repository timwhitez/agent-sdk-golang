package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type frameBindingSource struct {
	current  llm.ChatModel
	bind     func(context.Context) (llm.ChatModel, bool, error)
	bindings int
	label    string
}

func (m *frameBindingSource) Provider() string {
	if m.label != "" {
		return m.label
	}
	return "outer"
}
func (*frameBindingSource) Model() string { return "outer" }
func (m *frameBindingSource) Invoke(ctx context.Context, request llm.InvokeRequest) (*llm.Completion, error) {
	return m.current.Invoke(ctx, request)
}
func (m *frameBindingSource) BindFrameModel(ctx context.Context) (llm.ChatModel, bool, error) {
	m.bindings++
	return m.bind(ctx)
}

func TestFrameModelBindingRetryAndNextFrame(t *testing.T) {
	for _, known := range []bool{false, true} {
		var calls []string
		source := &frameBindingSource{}
		newModel := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
			calls = append(calls, "new")
			return &llm.Completion{Content: llm.TextContent("done")}, nil
		}}
		oldModel := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
			calls = append(calls, "old")
			if len(calls) == 1 {
				source.current = newModel
				return nil, &llm.ProviderError{StatusCode: 500, Message: "fixture"}
			}
			return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "work", Function: llm.FunctionCall{Name: "work", Arguments: `{}`}}}}, nil
		}}
		source.current = oldModel
		source.bind = func(context.Context) (llm.ChatModel, bool, error) {
			return &frameBindingSource{current: source.current}, known, nil
		}
		handlers := 0
		ag, err := New(Config{LLM: source, QueryIDGenerator: func() string { return "binding" }, InvokeRetryMaxAttempts: 2, InvokeRetryBackoff: time.Millisecond, Warningf: func(string, ...any) {}, Tools: []tools.Tool{tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { handlers++; return "ok", nil })}})
		if err != nil {
			t.Fatal(err)
		}
		var final EventEnvelope
		for e := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("run")) {
			if _, ok := e.Event.(ToolCallEvent); ok && (e.FrameID != "binding/frame/1" || e.InvokeAttempt != 2) {
				t.Error("binding altered retry correlation", e.FrameID, e.InvokeAttempt)
			}
			if _, ok := e.Event.(FinalResponseEvent); ok {
				final = e
			}
		}
		want, bindings, toolCalls, frame := []string{"old", "new"}, 1, 0, "binding/frame/1"
		if known {
			want, bindings, toolCalls, frame = []string{"old", "old", "new"}, 2, 1, "binding/frame/2"
		}
		if !reflect.DeepEqual(calls, want) || source.bindings != bindings || handlers != toolCalls || final.FrameID != frame {
			t.Fatalf("known=%v calls=%v bindings=%d handlers=%d final=%s", known, calls, source.bindings, handlers, final.FrameID)
		}
	}
}

func TestFrameBindingFailureCleansUnacceptedContinuation(t *testing.T) {
	for _, canceled := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		invocations := 0
		partial := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
			invocations++
			return &llm.Completion{StopReason: "max_tokens", ToolCalls: []llm.ToolCall{{ID: "unfinished", Function: llm.FunctionCall{Name: "work", Arguments: `{"x":`}}}}, nil
		}}
		source := &frameBindingSource{current: partial}
		source.bind = func(context.Context) (llm.ChatModel, bool, error) {
			if source.bindings == 1 {
				return &frameBindingSource{current: partial}, true, nil
			}
			if canceled {
				cancel()
			}
			return nil, false, errors.New("PRIVATE_BINDING_SECRET")
		}
		ag, err := New(Config{LLM: source, Warningf: func(format string, args ...any) {
			if strings.Contains(fmt.Sprintf(format, args...), "PRIVATE_") {
				t.Error("raw binding error logged")
			}
		}})
		if err != nil {
			t.Fatal(err)
		}
		errorsSeen := 0
		for e := range ag.QueryStreamEnveloped(ctx, llm.TextContent("run")) {
			if failure, ok := e.Event.(ErrorEvent); ok {
				errorsSeen++
				kind := "invalid_request"
				if canceled {
					kind = "canceled"
				}
				if failure.Kind != kind || strings.Contains(failure.Message, "PRIVATE_") || e.InvokeAttempt != 0 || e.Origin != EventOriginSDKDriver {
					t.Fatal("wrong binding failure outcome", failure, e.InvokeAttempt, e.Origin)
				}
			}
			if _, ok := e.Event.(ToolResultEvent); ok {
				t.Error("unaccepted fragment got result")
			}
		}
		if invocations != 1 || source.bindings != 2 || errorsSeen != 1 {
			t.Fatal(invocations, source.bindings, errorsSeen)
		}
		for _, m := range ag.Messages() {
			if len(m.ToolCalls) != 0 || m.Role == llm.RoleTool {
				t.Fatal("unaccepted continuation survived")
			}
		}
	}
}

func TestFrameBindingTerminalErrorUsesBoundProvider(t *testing.T) {
	target := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) { return nil, errors.New("fixture rejection") }}
	source := &frameBindingSource{current: target, bind: func(context.Context) (llm.ChatModel, bool, error) {
		return &frameBindingSource{current: target, label: "fixture"}, true, nil
	}}
	ag, err := New(Config{LLM: source, InvokeRetryMaxAttempts: 1, Warningf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	errorsSeen := 0
	for event := range ag.QueryStream(context.Background(), llm.TextContent("run")) {
		if e, ok := event.(ErrorEvent); ok {
			errorsSeen++
			if e.Provider != "fixture" {
				t.Fatalf("error provider=%q, used outer instead of bound model", e.Provider)
			}
		}
	}
	if errorsSeen != 1 {
		t.Fatalf("terminal errors=%d", errorsSeen)
	}
}

type embeddedFrameClient struct {
	*anthropic.Client
	calls int
}

func (*embeddedFrameClient) Provider() string { return "wrapper" }
func (*embeddedFrameClient) Model() string    { return "wrapper" }
func (m *embeddedFrameClient) InvokeStream(context.Context, llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.calls++
	out := make(chan llm.StreamEvent, 2)
	out <- llm.StreamTextDeltaEvent{Delta: "wrapper guard"}
	out <- llm.StreamDoneEvent{}
	close(out)
	return out, nil
}

type frameBindingTransport func(*http.Request) (*http.Response, error)

func (f frameBindingTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestPromotedFrameBinderCannotBypassWrapper(t *testing.T) {
	for _, nilClient := range []bool{false, true} {
		httpCalls := 0
		wrapper := &embeddedFrameClient{}
		if !nilClient {
			wrapper.Client = &anthropic.Client{ModelName: "fixture", MaxRetries: 1, HTTPClient: &http.Client{Transport: frameBindingTransport(func(r *http.Request) (*http.Response, error) {
				httpCalls++
				return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"unexpected"}}`)), Request: r}, nil
			})}}
		}
		ag, err := New(Config{LLM: wrapper, InvokeRetryMaxAttempts: 1, Warningf: func(string, ...any) {}})
		if err != nil {
			t.Fatal(err)
		}
		out, err := ag.Query(context.Background(), "run")
		if err != nil || out != "wrapper guard" || wrapper.calls != 1 || httpCalls != 0 {
			t.Fatalf("promoted binder bypassed wrapper: %q %v calls=%d http=%d", out, err, wrapper.calls, httpCalls)
		}
	}
}

func TestAnthropicFrameBindingOwnsActualRetryWire(t *testing.T) {
	temperature := 0.2
	client := &anthropic.Client{BaseURL: "https://fixture.invalid", ModelName: "old", Temperature: &temperature, Beta: []string{"old-beta"}, MaxRetries: 1}
	var bodies, betas []string
	client.HTTPClient = &http.Client{Transport: frameBindingTransport(func(r *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(body))
		betas = append(betas, r.Header.Get("anthropic-beta"))
		status := 200
		payload := ""
		switch len(bodies) {
		case 1:
			// Binding is complete before this callback. Source changes must not
			// alter this frame's retry, but must be visible to the next frame.
			client.ModelName, client.Beta[0], temperature = "new", "new-beta", 0.9
			status, payload = 500, `{"error":{"message":"fixture"}}`
		case 2:
			payload = "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"usage\":{\"input_tokens\":7}}}\n\n" +
				"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"work\",\"name\":\"work\",\"input\":{}}}\n\n" +
				"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"partial_json\":\"{}\"}}\n\n" +
				"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":1}}\n\ndata: {\"type\":\"message_stop\"}\n\n"
		default:
			payload = "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m2\",\"usage\":{\"input_tokens\":7}}}\n\n" +
				"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"text\":\"done\"}}\n\n" +
				"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\ndata: {\"type\":\"message_stop\"}\n\n"
		}
		return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(payload)), Request: r}, nil
	})}
	handlers := 0
	ag, err := New(Config{LLM: client, InvokeRetryMaxAttempts: 2, InvokeRetryBackoff: time.Millisecond, Warningf: func(string, ...any) {}, Tools: []tools.Tool{tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { handlers++; return "ok", nil })}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := ag.Query(ctx, "run")
	if err != nil || out != "done" || len(bodies) != 3 || handlers != 1 {
		t.Fatalf("wire run: %q %v requests=%d tools=%d", out, err, len(bodies), handlers)
	}
	if bodies[0] != bodies[1] || betas[0] != betas[1] || !strings.Contains(betas[2], "new-beta") {
		t.Fatal("same-frame retry wire changed")
	}
	for i, body := range bodies {
		var payload map[string]any
		if err := json.Unmarshal([]byte(body), &payload); err != nil {
			t.Fatal(err)
		}
		model, temp := "old", 0.2
		if i == 2 {
			model, temp = "new", 0.9
		}
		if payload["model"] != model || payload["temperature"] != temp {
			t.Fatalf("request %d did not use bound configuration: %v %v", i, payload["model"], payload["temperature"])
		}
	}
}
