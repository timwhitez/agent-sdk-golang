package agent

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
)

type anthropicUsageTransport func(*http.Request) (*http.Response, error)

func (f anthropicUsageTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func anthropicUsageStart(id string, output int) string {
	return fmt.Sprintf(`data: {"type":"message_start","message":{"id":%q,"usage":{"input_tokens":100,"output_tokens":%d,"cache_read_input_tokens":30,"cache_creation_input_tokens":5}}}`+"\n\n", id, output)
}

const anthropicUsageFailure = `data: {"type":"error","error":{"type":"overloaded_error","message":"fixture overloaded"}}` + "\n\n"
const anthropicUsageText = `data: {"type":"content_block_delta","delta":{"text":"ok"}}` + "\n\n"
const anthropicUsageDone = `data: {"type":"message_stop"}` + "\n\n"

func TestAnthropicAttemptUsageAccounting(t *testing.T) {
	for _, tt := range []struct {
		name, first, second string
		totals              []int
		cancelBackoff       bool
	}{
		{"start error retry", anthropicUsageStart("msg_failed", 1) + anthropicUsageFailure, anthropicUsageStart("msg_success", 3) + anthropicUsageText + anthropicUsageDone, []int{136, 138}, false},
		{"cumulative snapshots settle once", anthropicUsageStart("msg_failed", 1) + `data: {"type":"message_delta","usage":{"output_tokens":2}}` + "\n\n" + `data: {"type":"message_delta","usage":{"output_tokens":5}}` + "\n\n" + `data: {"type":"message_delta","usage":{"output_tokens":7}}` + "\n\n" + anthropicUsageFailure, anthropicUsageStart("msg_success", 3) + anthropicUsageText + anthropicUsageDone, []int{142, 138}, false},
		{"retry exhaustion", anthropicUsageStart("msg_failed", 1) + anthropicUsageFailure, anthropicUsageStart("msg_success", 3) + anthropicUsageFailure, []int{136, 138}, false},
		{"cancel during backoff", anthropicUsageStart("msg_failed", 1) + anthropicUsageFailure, "", []int{136}, true},
		{"text then error", anthropicUsageStart("msg_failed", 1) + anthropicUsageText + anthropicUsageFailure, "", []int{136}, false},
		{"EOF", anthropicUsageStart("msg_failed", 1), "", []int{136}, false},
		{"resource failure is not retried", anthropicUsageStart("msg_failed", 1) + strings.Repeat("data: {\n\n", 16), "", []int{136}, false},
		{"unknown failed usage", `data: {"type":"message_start","message":{"id":"msg_failed"}}` + "\n\n" + anthropicUsageFailure, anthropicUsageStart("msg_success", 3) + anthropicUsageText + anthropicUsageDone, []int{138}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int32
			client := &anthropic.Client{BaseURL: "https://fixture.invalid", ModelName: "fixture", MaxRetries: 1, HTTPClient: &http.Client{Transport: anthropicUsageTransport(func(r *http.Request) (*http.Response, error) {
				wire := tt.first
				if calls.Add(1) > 1 {
					wire = tt.second
				}
				if wire == "" {
					t.Error("unexpected retry")
					wire = anthropicUsageFailure
				}
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(wire)), Request: r}, nil
			})}}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			config := Config{LLM: client, InvokeRetryMaxAttempts: 2, Warningf: func(string, ...any) {}}
			if tt.cancelBackoff {
				config.InvokeRetryBackoff = time.Hour
				config.Warningf = func(string, ...any) { cancel() }
			}
			ag, err := New(config)
			if err != nil {
				t.Fatal(err)
			}
			var usage, accounting []observedUsage
			var ids, accountingIDs []string
			for envelope := range ag.QueryStreamEnveloped(ctx, llm.TextContent("hello")) {
				switch e := envelope.Event.(type) {
				case UsageEvent:
					usage = append(usage, observedUsage{total: e.Usage.TotalTokens, attempt: envelope.InvokeAttempt, frameID: envelope.FrameID})
					ids = append(ids, e.ResponseID)
					if e.Usage.PromptTokens != 135 || e.Usage.PromptCachedTokens == nil || *e.Usage.PromptCachedTokens != 30 || e.Usage.PromptCacheCreationTokens == nil || *e.Usage.PromptCacheCreationTokens != 5 {
						t.Fatalf("usage=%+v", e.Usage)
					}
				case AccountingEvent:
					if e.CorrelationKind != "response" {
						continue
					}
					if e.Payload.Usage == nil || e.Payload.Usage.TotalTokens == nil {
						t.Fatal("missing accounting usage")
					}
					accounting = append(accounting, observedUsage{total: int(*e.Payload.Usage.TotalTokens), attempt: envelope.InvokeAttempt, frameID: envelope.FrameID})
					accountingIDs = append(accountingIDs, e.ResponseID)
				}
			}
			wantCalls := 1
			if tt.second != "" {
				wantCalls = 2
			}
			if int(calls.Load()) != wantCalls {
				t.Fatalf("calls=%d want %d", calls.Load(), wantCalls)
			}
			for name, got := range map[string][]observedUsage{"usage": usage, "accounting": accounting} {
				if len(got) != len(tt.totals) {
					t.Fatalf("%s=%+v want totals %v", name, got, tt.totals)
				}
				for i, e := range got {
					attempt := uint64(i + 1)
					if tt.name == "unknown failed usage" {
						attempt = 2
					}
					if e.total != tt.totals[i] || e.attempt != attempt || e.frameID == "" || e.frameID != got[0].frameID {
						t.Fatalf("%s[%d]=%+v", name, i, e)
					}
				}
			}
			for i, id := range ids {
				want := "msg_failed"
				if usage[i].attempt == 2 {
					want = "msg_success"
				}
				if id != want || accountingIDs[i] != want {
					t.Fatalf("response IDs usage=%v accounting=%v", ids, accountingIDs)
				}
			}
		})
	}
}

func TestStreamMetadataKeepsOnlyLatestUsage(t *testing.T) {
	b := &streamMetadataBuffer{}
	b.add(llm.StreamResponseEvent{ResponseID: "response"})
	for i := 1; i <= 1000; i++ {
		b.add(llm.StreamUsageEvent{Usage: llm.Usage{TotalTokens: i}})
	}
	if len(b.events) != 2 || b.usage == nil || b.usage.TotalTokens != 1000 {
		t.Fatalf("events=%d usage=%+v", len(b.events), b.usage)
	}
	seen := 0
	if err := b.flush(func(ev llm.StreamEvent) error {
		if e, ok := ev.(llm.StreamUsageEvent); ok {
			seen++
			if e.Usage.TotalTokens != 1000 {
				t.Fatal(e.Usage)
			}
		}
		return nil
	}); err != nil || seen != 1 {
		t.Fatalf("err=%v usage events=%d", err, seen)
	}
}

// The reader waits for cancellation after a fully delivered SSE prefix. User
// cancellation is triggered by the visible TextDeltaEvent (a delivery barrier),
// while idle termination uses the Agent's actual idle timer and no sleeps.
type anthropicStalledBody struct {
	io.Reader
	ctx    context.Context
	closed chan struct{}
}

func (b *anthropicStalledBody) Read(p []byte) (int, error) {
	n, err := b.Reader.Read(p)
	if err == io.EOF {
		<-b.ctx.Done()
		return 0, b.ctx.Err()
	}
	return n, err
}
func (b *anthropicStalledBody) Close() error { close(b.closed); return nil }

func TestAnthropicAttemptUsageOnIdleAndUserCancellation(t *testing.T) {
	for _, cancelOnText := range []bool{false, true} {
		name := "idle"
		if cancelOnText {
			name = "user cancel after text"
		}
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			closed := make(chan struct{})
			client := &anthropic.Client{BaseURL: "https://fixture.invalid", ModelName: "fixture", MaxRetries: 1, HTTPClient: &http.Client{Transport: anthropicUsageTransport(func(r *http.Request) (*http.Response, error) {
				calls.Add(1)
				prefix := anthropicUsageStart("msg_stalled", 1)
				if cancelOnText {
					prefix += anthropicUsageText
				}
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: &anthropicStalledBody{Reader: strings.NewReader(prefix), ctx: r.Context(), closed: closed}, Request: r}, nil
			})}}
			ag, err := New(Config{LLM: client, InvokeRetryMaxAttempts: 2, StreamIdleTimeout: 100 * time.Millisecond, StreamIdleMaxRecoveries: 0, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			usageCount, accountingCount := 0, 0
			for envelope := range ag.QueryStreamEnveloped(ctx, llm.TextContent("hello")) {
				switch e := envelope.Event.(type) {
				case TextDeltaEvent:
					if cancelOnText {
						cancel()
					}
				case UsageEvent:
					usageCount++
					if e.ResponseID != "msg_stalled" || e.Usage.TotalTokens != 136 || envelope.InvokeAttempt != 1 || envelope.FrameID == "" {
						t.Fatalf("usage=%+v envelope=%+v", e, envelope)
					}
				case AccountingEvent:
					if e.CorrelationKind == "response" {
						accountingCount++
						if e.ResponseID != "msg_stalled" || e.Payload.Usage == nil || e.Payload.Usage.TotalTokens == nil || *e.Payload.Usage.TotalTokens != 136 || envelope.InvokeAttempt != 1 {
							t.Fatalf("accounting=%+v", e)
						}
					}
				}
			}
			if calls.Load() != 1 || usageCount != 1 || accountingCount != 1 {
				t.Fatalf("calls=%d usage=%d accounting=%d", calls.Load(), usageCount, accountingCount)
			}
			select {
			case <-closed:
			case <-time.After(time.Second):
				t.Fatal("HTTP body left open")
			}
		})
	}
}
