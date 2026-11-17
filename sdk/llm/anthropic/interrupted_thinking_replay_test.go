package anthropic

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const interruptedThinkingTextStream = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m2\",\"usage\":{\"input_tokens\":5}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

// interruptedThinkingTransport streams one unsigned thinking_delta on the
// first request and then either cuts the stream (cut) or hangs until the
// request is canceled. Later requests get a normal text turn.
type interruptedThinkingTransport struct {
	mu     sync.Mutex
	bodies []string
	cut    bool
}

func (t *interruptedThinkingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	b, _ := io.ReadAll(r.Body)
	t.mu.Lock()
	t.bodies = append(t.bodies, string(b))
	n := len(t.bodies)
	t.mu.Unlock()
	hdr := http.Header{"Content-Type": {"text/event-stream"}}
	if n > 1 {
		return &http.Response{StatusCode: 200, Status: "200 OK", Header: hdr, Body: io.NopCloser(strings.NewReader(interruptedThinkingTextStream)), Request: r}, nil
	}
	pr, pw := io.Pipe()
	go func() {
		_, _ = io.WriteString(pw, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"usage\":{\"input_tokens\":5}}}\n\n"+
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n"+
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"partial reasoning\"}}\n\n")
		if t.cut {
			_ = pw.CloseWithError(io.ErrUnexpectedEOF)
			return
		}
		<-r.Context().Done()
		_ = pw.CloseWithError(r.Context().Err())
	}()
	return &http.Response{StatusCode: 200, Status: "200 OK", Header: hdr, Body: pr, Request: r}, nil
}

// #241: thinking interrupted before its signature arrived is never written
// to history, so no later request replays an unsigned thinking block.
func TestInterruptedUnsignedThinkingIsNotReplayed(t *testing.T) {
	budget := 1024
	for _, tc := range []struct {
		name string
		cut  bool
		idle time.Duration
		// run drives the interrupted turn and, when needed, the next one.
		run func(ag *agent.Agent)
	}{
		{name: "steering", run: func(ag *agent.Agent) {
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			steer := make(chan agent.SteeringMsg, 1)
			sent := false
			for ev := range ag.QueryStreamWithSteering(ctx, llm.TextContent("think"), steer) {
				if _, ok := ev.(agent.ThinkingDeltaEvent); ok && !sent {
					sent = true
					steer <- agent.SteeringMsg{Content: "do something else"}
				}
			}
		}},
		{name: "stream idle recovery", idle: 300 * time.Millisecond, run: func(ag *agent.Agent) {
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			for range ag.QueryStream(ctx, llm.TextContent("think")) {
			}
		}},
		{name: "mid-stream error then next turn", cut: true, run: func(ag *agent.Agent) {
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			for range ag.QueryStream(ctx, llm.TextContent("think")) {
			}
			for range ag.QueryStream(ctx, llm.TextContent("continue")) {
			}
		}},
		{name: "cancel then next turn", run: func(ag *agent.Agent) {
			ctx1, cancel1 := context.WithTimeout(context.Background(), 15*time.Second)
			for ev := range ag.QueryStream(ctx1, llm.TextContent("think")) {
				if _, ok := ev.(agent.ThinkingDeltaEvent); ok {
					cancel1()
				}
			}
			cancel1()
			ctx2, cancel2 := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel2()
			for range ag.QueryStream(ctx2, llm.TextContent("continue")) {
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := &interruptedThinkingTransport{cut: tc.cut}
			client := &Client{ModelName: "claude-test", BaseURL: "http://fake.invalid", MaxRetries: 1, MaxTokens: 2048,
				ThinkingBudgetTokens: &budget, HTTPClient: &http.Client{Transport: tr}}
			ag, err := agent.New(agent.Config{LLM: client, StreamIdleTimeout: tc.idle, StreamIdleMaxRecoveries: 1, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			tc.run(ag)
			tr.mu.Lock()
			defer tr.mu.Unlock()
			if len(tr.bodies) < 2 {
				t.Fatalf("expected a follow-up request, got %d", len(tr.bodies))
			}
			if strings.Contains(tr.bodies[1], "partial reasoning") {
				t.Fatalf("follow-up request replays the unsigned partial thinking block: %s", tr.bodies[1])
			}
			for _, m := range ag.Messages() {
				for _, b := range m.Content.Blocks {
					if b.Type == "thinking" && b.Signature == "" {
						t.Fatalf("history kept an unsigned thinking block: %#v", m)
					}
				}
			}
		})
	}
}
