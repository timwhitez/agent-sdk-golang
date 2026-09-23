package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// All tests use local HTTP fixtures; no request leaves the machine.

const secretKey = "sk-canary-must-not-leak"

type protocolFixture struct {
	protocol string
	path     string   // required endpoint suffix
	first    string   // first text delta event (data payload)
	rest     []string // remaining events, ending with the terminal
}

var fixtures = []protocolFixture{
	{
		protocol: "anthropic",
		path:     "/messages",
		first:    `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
		rest: []string{
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
			`{"type":"message_stop"}`,
		},
	},
	{
		protocol: "chat",
		path:     "/chat/completions",
		first:    `{"choices":[{"delta":{"content":"Hel"}}]}`,
		rest: []string{
			`{"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}`,
			`[DONE]`,
		},
	},
	{
		protocol: "responses",
		path:     "/responses",
		first:    `{"type":"response.output_text.delta","delta":"Hel"}`,
		rest: []string{
			`{"type":"response.output_text.delta","delta":"lo"}`,
			`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}]}}`,
		},
	},
}

// anthropicPrelude opens the Anthropic message and text block.
var anthropicPrelude = []string{
	`{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":3}}}`,
	`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
}

func writeEvents(w http.ResponseWriter, events ...string) {
	flusher := w.(http.Flusher)
	for _, event := range events {
		fmt.Fprintf(w, "data: %s\n\n", event)
		flusher.Flush()
	}
}

type capturedRequest struct {
	path   string
	stream any
	auth   bool
}

// gatedServer sends the prelude and the first delta, flushes, then waits for
// gate before sending the rest (or, when rest is nil, returns without a
// terminal). onClose is closed if the client goes away while waiting.
type gatedServer struct {
	*httptest.Server
	gate     chan struct{}
	release  func()
	requests chan capturedRequest
	onClose  chan struct{}
	timedOut chan struct{}
}

func newGatedServer(t *testing.T, fx protocolFixture, rest []string, waitForClose bool) *gatedServer {
	t.Helper()
	s := &gatedServer{gate: make(chan struct{}), requests: make(chan capturedRequest, 4), onClose: make(chan struct{}), timedOut: make(chan struct{})}
	var once sync.Once
	s.release = func() { once.Do(func() { close(s.gate) }) }
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		select { // never block a retried request on the capture buffer
		case s.requests <- capturedRequest{path: r.URL.Path, stream: body["stream"], auth: r.Header.Get("Authorization") != "" || r.Header.Get("x-api-key") != ""}:
		default:
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if fx.protocol == "anthropic" {
			writeEvents(w, anthropicPrelude...)
		}
		writeEvents(w, fx.first)
		if waitForClose {
			select {
			case <-r.Context().Done():
				close(s.onClose)
			case <-time.After(5 * time.Second):
				close(s.timedOut)
			}
			return
		}
		select {
		case <-s.gate:
		case <-r.Context().Done():
			return
		case <-time.After(5 * time.Second):
			close(s.timedOut)
			return
		}
		writeEvents(w, rest...)
	}))
	// Release before Close so a waiting handler can finish on every path.
	t.Cleanup(func() { s.release(); s.Server.Close() })
	return s
}

type streamingModel = interface {
	llm.StreamingChatModel
}

func clientFor(t *testing.T, protocol, baseURL string) streamingModel {
	t.Helper()
	model, err := newModel(config{Protocol: protocol, Model: "fixture-model", APIKey: secretKey, BaseURL: baseURL})
	if err != nil {
		t.Fatal(err)
	}
	return model
}

// ackWriter records output and closes ack on the first non-empty write.
type ackWriter struct {
	mu      sync.Mutex
	buf     bytes.Buffer
	once    sync.Once
	ack     chan struct{}
	onFirst func()
}

func newAckWriter() *ackWriter { return &ackWriter{ack: make(chan struct{})} }

func (w *ackWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	n, err := w.buf.Write(p)
	w.mu.Unlock()
	if len(p) > 0 {
		w.once.Do(func() {
			close(w.ack)
			if w.onFirst != nil {
				w.onFirst()
			}
		})
	}
	return n, err
}

func (w *ackWriter) String() string { w.mu.Lock(); defer w.mu.Unlock(); return w.buf.String() }

// D01: missing configuration fails before any network call.
func TestD01MissingConfigFailsBeforeNetwork(t *testing.T) {
	for _, cfg := range []config{
		{Protocol: "chat", APIKey: "k"},
		{Protocol: "chat", Model: "m"},
		{Protocol: "grpc", Model: "m", APIKey: "k"},
	} {
		if _, err := newModel(cfg); err == nil {
			t.Fatalf("config %+v accepted", cfg)
		}
	}
}

// D02 + D03: each client posts stream:true to its real endpoint, and the first
// text delta reaches the consumer while the server is still holding back the
// rest of the response, including the terminal event.
func TestD02D03FirstDeltaArrivesBeforeTerminal(t *testing.T) {
	for _, fx := range fixtures {
		t.Run(fx.protocol, func(t *testing.T) {
			server := newGatedServer(t, fx, fx.rest, false)
			out := newAckWriter()
			result := make(chan error, 1)
			go func() {
				result <- streamInvoke(context.Background(), clientFor(t, fx.protocol, server.URL), "hi", out, io.Discard)
			}()
			select {
			case <-out.ack:
			case <-server.timedOut:
				t.Fatal("server timed out waiting; the delta was not delivered before the terminal")
			case err := <-result:
				t.Fatalf("stream ended before the first delta was consumed: %v", err)
			case <-time.After(5 * time.Second):
				t.Fatal("first delta never reached the consumer")
			}
			if got := out.String(); got != "Hel" {
				t.Fatalf("before the terminal the consumer saw %q, want only the first delta", got)
			}
			server.release()
			if err := <-result; err != nil {
				t.Fatal(err)
			}
			if got := out.String(); got != "Hello\n" {
				t.Fatalf("output=%q, want each delta exactly once", got)
			}
			req := <-server.requests
			if !strings.HasSuffix(req.path, fx.path) || req.stream != true || !req.auth {
				t.Fatalf("request path=%s stream=%v auth=%v", req.path, req.stream, req.auth)
			}
		})
	}
}

// D04: cancelling after the first delta stops the request (the server sees
// the connection close) and is not reported as success; a stream that ends
// without its terminal event is an error; neither leaks the API key.
func TestD04CancelAndTruncationAreFailures(t *testing.T) {
	for _, fx := range fixtures {
		t.Run(fx.protocol+"/cancel", func(t *testing.T) {
			server := newGatedServer(t, fx, nil, true)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			out := newAckWriter()
			out.onFirst = cancel
			err := streamInvoke(ctx, clientFor(t, fx.protocol, server.URL), "hi", out, io.Discard)
			if err == nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("cancelled stream returned %v", err)
			}
			select {
			case <-server.onClose:
			case <-server.timedOut:
				t.Fatal("request was not closed after cancel")
			case <-time.After(5 * time.Second):
				t.Fatal("request was not closed after cancel")
			}
		})
		t.Run(fx.protocol+"/truncated", func(t *testing.T) {
			server := newGatedServer(t, fx, []string{}, false)
			server.release() // send nothing after the first delta, then end
			var diag bytes.Buffer
			out := newAckWriter()
			err := streamInvoke(context.Background(), clientFor(t, fx.protocol, server.URL), "hi", out, &diag)
			if err == nil {
				t.Fatalf("truncated stream reported success; output %q", out.String())
			}
			if strings.Contains(err.Error()+out.String()+diag.String(), secretKey) {
				t.Fatal("API key leaked into output or error")
			}
		})
	}
}

// D06: through the Agent, streamed deltas are printed once and the final
// answer is not appended again.
func TestD06AgentPrintsDeltasOnce(t *testing.T) {
	for _, fx := range fixtures {
		t.Run(fx.protocol, func(t *testing.T) {
			server := newGatedServer(t, fx, fx.rest, false)
			server.release()
			out := newAckWriter()
			var diag bytes.Buffer
			if err := streamAgent(context.Background(), clientFor(t, fx.protocol, server.URL), "hi", out, &diag); err != nil {
				t.Fatal(err)
			}
			if got := out.String(); got != "Hello\n" {
				t.Fatalf("agent output=%q", got)
			}
		})
	}
}

// D06: event consumer semantics independent of any provider.
func TestD06ConsumerStatusHandling(t *testing.T) {
	run := func(events ...agent.Event) (string, error) {
		ch := make(chan agent.Event, len(events))
		for _, e := range events {
			ch <- e
		}
		close(ch)
		var out, diag bytes.Buffer
		err := consumeAgentEvents(ch, &out, &diag)
		return out.String(), err
	}
	if out, err := run(agent.FinalResponseEvent{Content: "whole"}); err != nil || out != "whole\n" {
		t.Fatalf("non-streamed final: %q %v", out, err)
	}
	if out, err := run(agent.TextDeltaEvent{Delta: "a"}, agent.TextEvent{Content: "a"}, agent.FinalResponseEvent{Content: "a"}); err != nil || out != "a\n" {
		t.Fatalf("streamed final duplicated: %q %v", out, err)
	}
	if _, err := run(agent.FinalResponseEvent{Content: "x", Status: "partial", Reason: "require_done_safety"}); !errors.Is(err, errPartialResponse) {
		t.Fatalf("partial accepted: %v", err)
	}
	if _, err := run(agent.FinalResponseEvent{Content: "x", DroppedEvents: 2, DroppedCriticalEvents: 1}); err == nil {
		t.Fatal("critical drop accepted")
	}
	if _, err := run(agent.FinalResponseEvent{Content: "x", DroppedEvents: 2}); err != nil {
		t.Fatalf("non-critical drop failed the run: %v", err)
	}
	if _, err := run(agent.TextDeltaEvent{Delta: "a"}, agent.ErrorEvent{Kind: "provider", Message: "boom"}); err == nil {
		t.Fatal("error event accepted")
	}
	if _, err := run(agent.TextDeltaEvent{Delta: "a"}); !errors.Is(err, errIncompleteStream) {
		t.Fatalf("stream without final accepted: %v", err)
	}
}

// D05: tool-call arguments split across SSE chunks are executed once, with
// the complete arguments, only after the stream's terminal event; the partial
// fragment held back by the server never reaches the Handler.
func TestD05FragmentedToolArgumentsExecuteOnceComplete(t *testing.T) {
	gate := make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(gate) }) }
	firstSent := make(chan struct{})
	var mu sync.Mutex
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []json.RawMessage `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		if len(body.Messages) > 2 { // the follow-up request after the tool result
			writeEvents(w, `{"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}`, `[DONE]`)
			return
		}
		writeEvents(w, `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":"{\"te"}}]}}]}`)
		close(firstSent)
		select {
		case <-gate:
		case <-r.Context().Done():
			return
		}
		writeEvents(w, `{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"xt\":\"hi\"}"}}]},"finish_reason":"tool_calls"}]}`, `[DONE]`)
	}))
	t.Cleanup(func() { release(); server.Close() })

	var got []string
	echo := tools.Func[struct {
		Text string `json:"text"`
	}]("echo", "echo text", func(_ context.Context, args struct {
		Text string `json:"text"`
	}, _ *tools.Container) (any, error) {
		mu.Lock()
		defer mu.Unlock()
		calls++
		got = append(got, args.Text)
		return "echoed", nil
	})
	a, err := agent.New(agent.Config{LLM: clientFor(t, "chat", server.URL), Tools: []tools.Tool{echo}})
	if err != nil {
		t.Fatal(err)
	}
	result := make(chan error, 1)
	var out, diag bytes.Buffer
	go func() {
		result <- consumeAgentEvents(a.QueryStream(context.Background(), llm.TextContent("hi")), &out, &diag)
	}()
	select {
	case <-firstSent:
	case <-time.After(5 * time.Second):
		t.Fatal("first fragment never sent")
	}
	mu.Lock()
	early := calls
	mu.Unlock()
	if early != 0 {
		t.Fatalf("handler ran %d times before the arguments were complete", early)
	}
	release()
	if err := <-result; err != nil {
		t.Fatal(err)
	}
	if calls != 1 || len(got) != 1 || got[0] != "hi" || out.String() != "done\n" {
		t.Fatalf("calls=%d args=%v out=%q", calls, got, out.String())
	}
}

// scriptedStreamer is a streaming model whose turns are fixed event lists.
type scriptedStreamer struct {
	mu    sync.Mutex
	turns [][]llm.StreamEvent
	next  int
}

func (m *scriptedStreamer) Provider() string { return "fixture" }
func (m *scriptedStreamer) Model() string    { return "scripted" }
func (m *scriptedStreamer) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return nil, errors.New("buffered call not expected")
}
func (m *scriptedStreamer) InvokeStream(context.Context, llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.next >= len(m.turns) {
		return nil, errors.New("no scripted turn left")
	}
	turn := m.turns[m.next]
	m.next++
	ch := make(chan llm.StreamEvent, len(turn))
	for _, e := range turn {
		ch <- e
	}
	close(ch)
	return ch, nil
}

func toolCallTurn(text, id, name, args string) []llm.StreamEvent {
	var events []llm.StreamEvent
	if text != "" {
		events = append(events, llm.StreamTextDeltaEvent{Delta: text})
	}
	return append(events,
		llm.StreamToolCallDeltaEvent{Index: 0, ID: id, NameDelta: name, ArgumentsDelta: args},
		llm.StreamDoneEvent{StopReason: "tool_calls"})
}

// D06 through a real Agent: progress text streamed before a tool is not the
// answer; a different final answer delivered by the done tool is printed.
func TestD06AgentPrintsDifferentFinalAfterProgress(t *testing.T) {
	type noArgs struct{}
	check := tools.Func[noArgs]("check", "inspect files", func(context.Context, noArgs, *tools.Container) (any, error) {
		return "ok", nil
	})
	type doneArgs struct {
		Message string `json:"message"`
	}
	done := tools.Func[doneArgs]("done", "finish with the answer", func(_ context.Context, args doneArgs, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})
	for _, tt := range []struct {
		name  string
		tools []tools.Tool
		turns [][]llm.StreamEvent
		want  string
	}{
		{
			name:  "progress, tool, different final from done",
			tools: []tools.Tool{check, done},
			turns: [][]llm.StreamEvent{
				toolCallTurn("Inspecting files...", "c1", "check", `{}`),
				toolCallTurn("", "c2", "done", `{"message":"Verified answer."}`),
			},
			want: "Inspecting files...\n" + finalMarker + "Verified answer.\n",
		},
		{
			name:  "streamed final answer is not repeated",
			tools: []tools.Tool{check},
			turns: [][]llm.StreamEvent{
				toolCallTurn("Inspecting files...", "c1", "check", `{}`),
				{llm.StreamTextDeltaEvent{Delta: "Verified "}, llm.StreamTextDeltaEvent{Delta: ""}, llm.StreamTextDeltaEvent{Delta: "answer."}, llm.StreamDoneEvent{StopReason: "stop"}},
			},
			want: "Inspecting files...\nVerified answer.\n",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a, err := agent.New(agent.Config{LLM: &scriptedStreamer{turns: tt.turns}, Tools: tt.tools})
			if err != nil {
				t.Fatal(err)
			}
			var out, diag bytes.Buffer
			if err := consumeAgentEvents(a.QueryStream(context.Background(), llm.TextContent("hi")), &out, &diag); err != nil {
				t.Fatal(err)
			}
			if got := out.String(); got != tt.want {
				t.Fatalf("output=%q want %q", got, tt.want)
			}
		})
	}
}

// D06: the final answer is compared with what the current model turn
// actually showed, not with whether any delta ever arrived.
func TestD06FinalAnswerAgainstShownTurn(t *testing.T) {
	run := func(events ...agent.Event) (string, error) {
		ch := make(chan agent.Event, len(events))
		for _, e := range events {
			ch <- e
		}
		close(ch)
		var out, diag bytes.Buffer
		err := consumeAgentEvents(ch, &out, &diag)
		return out.String(), err
	}
	for _, tt := range []struct {
		name   string
		events []agent.Event
		want   string
	}{
		{"done final without deltas", []agent.Event{agent.ToolCallEvent{Tool: "done"}, agent.FinalResponseEvent{Content: "Answer."}}, "Answer.\n"},
		{"same text once", []agent.Event{agent.TextDeltaEvent{Delta: "Hel"}, agent.TextDeltaEvent{Delta: "lo"}, agent.FinalResponseEvent{Content: "Hello"}}, "Hello\n"},
		{"empty deltas", []agent.Event{agent.TextDeltaEvent{Delta: ""}, agent.FinalResponseEvent{Content: "Hello"}}, "Hello\n"},
		{"lost delta recovered from final", []agent.Event{agent.TextDeltaEvent{Delta: "Hel"}, agent.FinalResponseEvent{Content: "Hello", DroppedEvents: 1}}, "Hel\n" + finalMarker + "Hello\n"},
		{"earlier turn text is not the answer", []agent.Event{
			agent.TextDeltaEvent{Delta: "Hello"}, agent.ToolCallEvent{Tool: "check"}, agent.ToolResultEvent{Tool: "check"},
			agent.FinalResponseEvent{Content: "Hello"},
		}, "Hello\n" + finalMarker + "Hello\n"},
		{"multiple turns, last turn streamed the answer", []agent.Event{
			agent.TextDeltaEvent{Delta: "Step 1."}, agent.ToolCallEvent{Tool: "a"}, agent.ToolResultEvent{Tool: "a"},
			agent.TextDeltaEvent{Delta: "Step 2."}, agent.ToolCallEvent{Tool: "b"}, agent.ToolResultEvent{Tool: "b"},
			agent.TextDeltaEvent{Delta: "Done."}, agent.FinalResponseEvent{Content: "Done."},
		}, "Step 1.\nStep 2.\nDone.\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := run(tt.events...)
			if err != nil || got != tt.want {
				t.Fatalf("output=%q err=%v want %q", got, err, tt.want)
			}
		})
	}
	if out, err := run(agent.TextDeltaEvent{Delta: "partial text"}, agent.FinalResponseEvent{Content: "fallback", Status: "partial", Reason: "require_done_safety"}); !errors.Is(err, errPartialResponse) || !strings.Contains(out, "fallback") {
		t.Fatalf("partial: out=%q err=%v", out, err)
	}
	if _, err := run(agent.TextDeltaEvent{Delta: "x"}, agent.FinalResponseEvent{Content: "x", DroppedEvents: 3, DroppedCriticalEvents: 1}); err == nil {
		t.Fatal("critical drop accepted")
	}
}
