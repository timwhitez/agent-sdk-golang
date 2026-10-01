package anthropic

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestSSELogicalEventBudgets(t *testing.T) {
	for _, tt := range []struct {
		name, wire string
		bytes      int
		reason     string
		want       int
	}{
		{"empty heartbeat", "data: \n\n", 3, "", 0},
		{"below", "data: {}\n\n", 3, "", 1},
		{"exact", "data: {}\n\n", 2, "", 1},
		{"above", "data: {}\n\n", 1, sseLimitEventBytes, 0},
		{"multiline exact", "data: {\ndata: }\n\n", 3, "", 1},
		{"multiline join counted", "data: {\ndata: }\n\n", 2, sseLimitEventBytes, 0},
		{"unicode bytes exact", "data: \"界\"\n\n", 5, "", 1},
		{"unicode bytes above", "data: \"界\"\n\n", 4, sseLimitEventBytes, 0},
		{"premature boundary exact", "data: {\n\n: comment\n\ndata: }\n\n", 3, "", 1},
		{"pending join counted", "data: {\n\ndata: }\n\n", 2, sseLimitEventBytes, 0},
		{"independent events reset", strings.Repeat("data: {}\n\n", 20), 2, "", 20},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			err := consumeSSEWithLimits(strings.NewReader(tt.wire), func(string) error { calls++; return nil }, sseLimits{tt.bytes, 3})
			if tt.reason == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				var limit *sseResourceLimitError
				if !errors.As(err, &limit) || limit.Reason != tt.reason || limit.Limit != tt.bytes {
					t.Fatalf("error=%v", err)
				}
			}
			if calls != tt.want {
				t.Fatalf("callbacks=%d want %d", calls, tt.want)
			}
		})
	}
	// Two failed candidates followed by a valid reconstruction consume exactly
	// three validation attempts, and the next event gets a fresh budget.
	calls := 0
	err := consumeSSEWithLimits(strings.NewReader("data: {\n\ndata: \"a\":\n\ndata: 1}\n\ndata: {}\n\n"), func(string) error { calls++; return nil }, sseLimits{30, 3})
	if err != nil || calls != 2 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}

// A reader with no EOF provides one physical line at a time, so the byte count
// proves the parser stops before pulling an unbounded stream into memory.
type repeatingSSEReader struct {
	line  string
	reads int
}

func (r *repeatingSSEReader) Read(p []byte) (int, error) { r.reads++; return copy(p, r.line), nil }

func TestSSEUnendingInputStopsAtBudget(t *testing.T) {
	for _, tt := range []struct {
		name, line, reason string
		limits             sseLimits
		reads              int
	}{
		{"short lines", "data: x\n", sseLimitEventBytes, sseLimits{7, 10}, 5},
		{"invalid boundaries", "data: {\n\n", sseLimitDecodeAttempts, sseLimits{100, 3}, 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := &repeatingSSEReader{line: tt.line}
			calls := 0
			err := consumeSSEWithLimits(r, func(string) error { calls++; return nil }, tt.limits)
			var limit *sseResourceLimitError
			if !errors.As(err, &limit) || limit.Reason != tt.reason || r.reads != tt.reads || calls != 0 {
				t.Fatalf("error=%v reads=%d callbacks=%d", err, r.reads, calls)
			}
		})
	}
	for _, wire := range []string{"data: {\n\n", "data: {\n"} {
		err := consumeSSEWithLimits(strings.NewReader(wire), func(string) error { t.Fatal("callback on invalid JSON"); return nil }, sseLimits{10, 1})
		var limit *sseResourceLimitError
		if !errors.As(err, &limit) || limit.Reason != sseLimitDecodeAttempts {
			t.Fatalf("error=%v", err)
		}
	}
}

type streamBudgetBody struct {
	io.Reader
	closes atomic.Int32
}

func (b *streamBudgetBody) Close() error { b.closes.Add(1); return nil }

func TestInvokeStreamBudgetClosesBodyAndKeepsPartialText(t *testing.T) {
	prefix := "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"partial\"}}\n\n"
	for _, tt := range []struct{ name, tail, reason string }{
		{"attempts", strings.Repeat("data: {\n\n", defaultSSEMaxDecodeAttempts+1), sseLimitDecodeAttempts},
		{"bytes", strings.Repeat("data: "+strings.Repeat("x", 8192)+"\n", defaultSSEMaxEventBytes/8192+1), sseLimitEventBytes},
	} {
		t.Run(tt.name, func(t *testing.T) {
			body := &streamBudgetBody{Reader: strings.NewReader(prefix + tt.tail)}
			client := &Client{BaseURL: "https://fixture.invalid", ModelName: "fixture", MaxRetries: 1, HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				resp := httpResponse(200, "", r)
				resp.Body = body
				return resp, nil
			})}}
			events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{})
			if err != nil {
				t.Fatal(err)
			}
			text, failures, done := "", 0, 0
			for event := range events {
				switch e := event.(type) {
				case llm.StreamTextDeltaEvent:
					text += e.Delta
				case llm.StreamDoneEvent:
					done++
				case llm.StreamErrorEvent:
					failures++
					var limit *sseResourceLimitError
					if !errors.As(e.Err, &limit) || limit.Reason != tt.reason {
						t.Fatalf("error=%v", e.Err)
					}
				}
			}
			if text != "partial" || failures != 1 || done != 0 || body.closes.Load() != 1 {
				t.Fatalf("text=%q errors=%d done=%d closes=%d", text, failures, done, body.closes.Load())
			}
		})
	}
}

func TestInvokeStreamObservedUsageSurvivesTermination(t *testing.T) {
	start := `data: {"type":"message_start","message":{"id":"msg_partial","usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":0,"cache_creation_input_tokens":5}}}` + "\n\n"
	for _, tt := range []struct {
		name, tail string
		want       int
		done       bool
	}{
		{"error", `data: {"type":"error","error":{"type":"overloaded_error"}}` + "\n\n", 1, false},
		{"EOF", "", 1, false},
		{"cumulative error", `data: {"type":"message_delta","usage":{"output_tokens":2}}` + "\n\n" + `data: {"type":"message_delta","usage":{"output_tokens":5}}` + "\n\n" + `data: {"type":"message_delta","usage":{"output_tokens":7}}` + "\n\n" + `data: {"type":"error","error":{"type":"overloaded_error"}}` + "\n\n", 7, false},
		{"complete", `data: {"type":"message_stop"}` + "\n\n", 1, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{BaseURL: "https://fixture.invalid", ModelName: "fixture", HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) { return httpResponse(200, start+tt.tail, r), nil })}}
			events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{})
			if err != nil {
				t.Fatal(err)
			}
			id := ""
			var usage *llm.Usage
			done, failures := 0, 0
			for event := range events {
				switch e := event.(type) {
				case llm.StreamResponseEvent:
					id = e.ResponseID
				case llm.StreamUsageEvent:
					if id != "msg_partial" {
						t.Fatal("usage before response ID")
					}
					u := e.Usage
					usage = &u
				case llm.StreamDoneEvent:
					done++
				case llm.StreamErrorEvent:
					failures++
				}
			}
			if usage == nil || usage.PromptTokens != 105 || usage.CompletionTokens != tt.want || usage.PromptCachedTokens == nil || *usage.PromptCachedTokens != 0 || usage.PromptCacheCreationTokens == nil || *usage.PromptCacheCreationTokens != 5 {
				t.Fatalf("usage=%+v", usage)
			}
			if (done == 1) != tt.done || (failures == 1) == tt.done {
				t.Fatalf("done=%d errors=%d", done, failures)
			}
		})
	}
	for _, usageJSON := range []string{"", `,"usage":{}`, `,"usage":{"input_tokens":null,"output_tokens":"bad"}`, `,"usage":{"input_tokens":0,"output_tokens":0}`} {
		client := &Client{BaseURL: "https://fixture.invalid", ModelName: "fixture", HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return httpResponse(200, `data: {"type":"message_start","message":{"id":"msg_zero"`+usageJSON+`}}`+"\n\n"+`data: {"type":"message_stop"}`+"\n\n", r), nil
		})}}
		events, err := client.InvokeStream(context.Background(), llm.InvokeRequest{})
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for event := range events {
			if _, ok := event.(llm.StreamUsageEvent); ok {
				count++
			}
		}
		if (count > 0) != (strings.Contains(usageJSON, "input_tokens\":0")) {
			t.Fatalf("usage JSON=%q events=%d", usageJSON, count)
		}
	}
}

// Stops after the consumer has received the snapshot, with the HTTP reader
// blocked on the request context rather than on a timing-dependent sleep.
type cancelSSEBody struct {
	io.Reader
	ctx    context.Context
	closed chan struct{}
}

func (b *cancelSSEBody) Read(p []byte) (int, error) {
	n, err := b.Reader.Read(p)
	if err == io.EOF {
		<-b.ctx.Done()
		return 0, b.ctx.Err()
	}
	return n, err
}
func (b *cancelSSEBody) Close() error { close(b.closed); return nil }
func TestInvokeStreamCancellationRetainsDeliveredUsage(t *testing.T) {
	closed := make(chan struct{})
	client := &Client{BaseURL: "https://fixture.invalid", ModelName: "fixture", HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		resp := httpResponse(200, "", r)
		resp.Body = &cancelSSEBody{Reader: strings.NewReader(`data: {"type":"message_start","message":{"id":"msg_cancel","usage":{"input_tokens":100,"output_tokens":1}}}` + "\n\n"), ctx: r.Context(), closed: closed}
		return resp, nil
	})}}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	events, err := client.InvokeStream(ctx, llm.InvokeRequest{})
	if err != nil {
		t.Fatal(err)
	}
	observed := false
	for event := range events {
		if e, ok := event.(llm.StreamUsageEvent); ok {
			if e.Usage.TotalTokens != 101 {
				t.Fatal(e.Usage)
			}
			observed = true
			cancel()
		}
	}
	if !observed {
		t.Fatal("usage not delivered")
	}
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("body not closed")
	}
}

func TestInvokeStreamAbandonedUsageConsumerClosesBody(t *testing.T) {
	closed := make(chan struct{})
	prefix := `data: {"type":"message_start","message":{"id":"msg_cancel","usage":{"input_tokens":100,"output_tokens":1}}}` + "\n\n" + strings.Repeat(`data: {"type":"message_delta","usage":{"output_tokens":7}}`+"\n\n", 1000)
	client := &Client{BaseURL: "https://fixture.invalid", ModelName: "fixture", HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		resp := httpResponse(200, "", r)
		resp.Body = &cancelSSEBody{Reader: strings.NewReader(prefix), ctx: r.Context(), closed: closed}
		return resp, nil
	})}}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	events, err := client.InvokeStream(ctx, llm.InvokeRequest{})
	if err != nil {
		t.Fatal(err)
	}
	// Receive the first snapshot and abandon the channel. More than the channel's
	// capacity of later snapshots exercises cancellation-aware backpressure.
	for event := range events {
		if _, ok := event.(llm.StreamUsageEvent); ok {
			break
		}
	}
	cancel()
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("abandoned consumer retained HTTP body")
	}
}
