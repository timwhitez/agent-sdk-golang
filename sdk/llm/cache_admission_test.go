package llm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
)

func admissionModel(provider string, transport cacheWireTransport, warning func(string, ...any)) llm.StreamingChatModel {
	client := &http.Client{Transport: transport}
	var model llm.StreamingChatModel
	switch provider {
	case "anthropic":
		model = &anthropic.Client{HTTPClient: client, BaseURL: "https://fixture.invalid", APIKey: "private-key", ModelName: "fixture", MaxTokens: 64, MaxRetries: 1}
	case "chat":
		model = &openai.ChatClient{HTTPClient: client, BaseURL: "https://fixture.invalid", APIKey: "private-key", ModelName: "fixture", MaxRetries: 1}
	case "responses":
		model = &openai.ResponsesClient{HTTPClient: client, BaseURL: "https://fixture.invalid", APIKey: "private-key", ModelName: "fixture", MaxRetries: 1}
	default:
		panic("unknown fixture provider")
	}
	model.(llm.WarningSinkSetter).SetWarningf(warning)
	return model
}

func admissionRequest(t *testing.T, policy llm.CacheDirectivePolicy) llm.InvokeRequest {
	t.Helper()
	request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleSystem, Content: llm.TextContent("system"), Cache: true}, {Role: llm.RoleUser, Content: llm.TextContent("hello")}}}
	view, err := llm.NewCacheTargetView(request)
	if err != nil {
		t.Fatal(err)
	}
	request.CachePlan, err = view.Bind([]llm.CacheDirective{{Target: llm.CacheTarget{Kind: llm.CacheAfterMessage, MessageIndex: 1}, Policy: policy}})
	if err != nil {
		t.Fatal(err)
	}
	return request
}

func callAdmissionModel(ctx context.Context, model llm.StreamingChatModel, request llm.InvokeRequest, stream bool) (*llm.Completion, error) {
	if !stream {
		return model.Invoke(ctx, request)
	}
	events, err := model.InvokeStream(ctx, request)
	if err != nil {
		return nil, err
	}
	completion := &llm.Completion{}
	for event := range events {
		switch e := event.(type) {
		case llm.StreamErrorEvent:
			err = e.Err
			if err == nil {
				err = errors.New("stream failure")
			}
		case llm.StreamTextDeltaEvent:
			completion.Content.Text += e.Delta
		}
	}
	return completion, err
}

func TestCacheAdmissionRejectsBeforeNetwork(t *testing.T) {
	for _, provider := range []string{"anthropic", "chat", "responses"} {
		for _, stream := range []bool{false, true} {
			for _, failure := range []string{"required", "unbound", "stale", "stale-best-effort", "clone-stale", "uncloneable", "view-overwrite", "view-clear", "schema", "canceled"} {
				t.Run(fmt.Sprintf("%s/%v/%s", provider, stream, failure), func(t *testing.T) {
					request := admissionRequest(t, llm.CacheRequired)
					reason := "unsupported_target"
					index := 0
					switch failure {
					case "view-overwrite", "view-clear":
						view, err := llm.NewCacheTargetView(request)
						if err != nil {
							t.Fatal(err)
						}
						request.CachePlan, err = view.Bind(request.CachePlan.Directives)
						if err != nil {
							t.Fatal(err)
						}
						if failure == "view-overwrite" {
							request.Messages[1].Content.Text = "private-rebound"
							fresh, err := llm.NewCacheTargetView(request)
							if err != nil {
								t.Fatal(err)
							}
							*view = *fresh
							reason, index = "stale_request", -1
						} else {
							*view = llm.CacheTargetView{}
						}
					case "unbound":
						request.CachePlan = &llm.CachePlan{RequestFingerprint: "private-fingerprint", Directives: request.CachePlan.Directives}
						reason, index = "unbound_plan", -1
					case "uncloneable":
						request.Tools = []llm.ToolDefinition{{Name: "private-tool", Parameters: map[string]any{"private-secret": func() {}}}}
						reason, index = "uncloneable_request", -1
					case "stale", "stale-best-effort", "clone-stale":
						if failure == "stale-best-effort" {
							request.CachePlan.Directives[0].Policy = llm.CacheBestEffort
						}
						if failure == "clone-stale" {
							var err error
							request, err = llm.CloneInvokeRequest(request)
							if err != nil {
								t.Fatal(err)
							}
						}
						request.Messages[1].Content.Text = "private-changed"
						reason, index = "stale_request", -1
					case "schema":
						request.CachePlan.SchemaVersion++
						reason, index = "unsupported_schema", -1
					}
					var calls atomic.Int32
					model := admissionModel(provider, func(r *http.Request) (*http.Response, error) { calls.Add(1); return nil, errors.New("must not send") }, func(string, ...any) { t.Error("rejected intent emitted skip warning") })
					ctx, cancel := context.WithCancel(context.Background())
					defer cancel()
					if failure == "canceled" {
						cancel()
					}
					_, err := callAdmissionModel(ctx, model, request, stream)
					if failure == "canceled" {
						if !errors.Is(err, context.Canceled) {
							t.Fatal(err)
						}
					} else {
						assertCacheViewError(t, err, reason, index)
					}
					if calls.Load() != 0 {
						t.Fatal("rejected request reached HTTP")
					}
				})
			}
		}
	}
}

var admissionSuccess = map[string]string{
	"anthropic": `{"id":"fixture","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
	"chat":      `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
	"responses": `{"id":"r","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`,
}

func admissionSSE(provider string) string {
	switch provider {
	case "anthropic":
		return "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	case "chat":
		return "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	default:
		return "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":" + admissionSuccess[provider] + "}\n\n"
	}
}

func TestCacheAdmissionBestEffortWireDiagnosticsAndIsolation(t *testing.T) {
	for _, provider := range []string{"anthropic", "chat", "responses"} {
		for _, stream := range []bool{false, true} {
			for _, status := range []int{200, 401} {
				t.Run(fmt.Sprintf("%s/%v/%d", provider, stream, status), func(t *testing.T) {
					var baseline []byte
					for _, planned := range []bool{false, true} {
						request := admissionRequest(t, llm.CacheBestEffort)
						if !planned {
							request.CachePlan = nil
						}
						before, err := llm.CloneInvokeRequest(request)
						if err != nil {
							t.Fatal(err)
						}
						var calls atomic.Int32
						var payload []byte
						var warnings []string
						model := admissionModel(provider, func(r *http.Request) (*http.Response, error) {
							calls.Add(1)
							var err error
							payload, err = io.ReadAll(r.Body)
							if err != nil {
								return nil, err
							}
							body := admissionSuccess[provider]
							if stream {
								body = admissionSSE(provider)
							}
							if status == 401 {
								body = `{"error":{"message":"fixture rejected"}}`
							}
							return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
						}, func(format string, args ...any) { warnings = append(warnings, fmt.Sprintf(format, args...)) })
						completion, err := callAdmissionModel(context.Background(), model, request, stream)
						if calls.Load() != 1 || (err != nil) != (status == 401) {
							t.Fatalf("calls=%d err=%v", calls.Load(), err)
						}
						if !reflect.DeepEqual(request, before) {
							t.Fatal("admission mutated input")
						}
						if !planned {
							baseline = payload
						} else if !bytes.Equal(payload, baseline) {
							t.Fatal("skip changed legacy wire")
						}
						if planned {
							if !reflect.DeepEqual(warnings, []string{"cache_plan_skipped: directive 0: unsupported_target"}) {
								t.Fatalf("warnings=%v", warnings)
							}
							if !stream && status == 200 && !reflect.DeepEqual(completion.Diagnostics, []llm.Diagnostic{{Kind: "cache_plan_skipped", Message: "directive 0: unsupported_target"}}) {
								t.Fatal("missing typed completion diagnostic")
							}
						} else if len(warnings) != 0 {
							t.Fatal("nil plan warned")
						}
					}
				})
			}
			t.Run(fmt.Sprintf("%s/%v/warning-callback-mutation", provider, stream), func(t *testing.T) {
				request := admissionRequest(t, llm.CacheBestEffort)
				model := admissionModel(provider, func(r *http.Request) (*http.Response, error) {
					body, _ := io.ReadAll(r.Body)
					if bytes.Contains(body, []byte("private-mutated")) || !bytes.Contains(body, []byte("hello")) {
						t.Error("caller mutation reached owned wire")
					}
					return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
				}, func(string, ...any) { request.Messages[1].Content.Text = "private-mutated" })
				_, _ = callAdmissionModel(context.Background(), model, request, stream)
			})
		}
	}
}

func TestCacheAdmissionBindingPersistenceAndDiagnosticBound(t *testing.T) {
	request := admissionRequest(t, llm.CacheBestEffort)
	encoded, err := json.Marshal(request.CachePlan)
	if err != nil || strings.Contains(string(encoded), "hello") {
		t.Fatal("binding leaked via JSON")
	}
	var restored llm.CachePlan
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatal(err)
	}
	request.CachePlan = &restored
	_, _, err = llm.AdmitCachePlan(context.Background(), request, &openai.ChatClient{}, nil)
	assertCacheViewError(t, err, "unbound_plan", -1)
	request = llm.InvokeRequest{Messages: make([]llm.Message, 40)}
	directives := make([]llm.CacheDirective, 40)
	for i := range directives {
		request.Messages[i] = llm.Message{Role: llm.RoleUser, Content: llm.TextContent("private-prompt")}
		directives[i] = llm.CacheDirective{Target: llm.CacheTarget{Kind: llm.CacheAfterMessage, MessageIndex: i}, Policy: llm.CacheBestEffort}
	}
	view, err := llm.NewCacheTargetView(request)
	if err != nil {
		t.Fatal(err)
	}
	request.CachePlan, err = view.Bind(directives)
	if err != nil {
		t.Fatal(err)
	}
	directives[0].Target.MessageIndex = -1
	_, diagnostics, err := llm.AdmitCachePlan(context.Background(), request, &openai.ChatClient{}, nil)
	if err != nil || len(diagnostics) != 33 || diagnostics[32].Message != "8 additional directives skipped" {
		t.Fatal("bounded diagnostics", err, diagnostics)
	}
	encoded, _ = json.Marshal(diagnostics)
	if strings.Contains(string(encoded), "private-") {
		t.Fatal("diagnostics leaked content")
	}
}

type admissionCancelBody struct {
	initial *strings.Reader
	ctx     context.Context
	closed  atomic.Bool
}

func (b *admissionCancelBody) Read(p []byte) (int, error) {
	if b.initial.Len() > 0 {
		return b.initial.Read(p)
	}
	<-b.ctx.Done()
	return 0, b.ctx.Err()
}

func (b *admissionCancelBody) Close() error { b.closed.Store(true); return nil }

func TestCacheAdmissionPartialStreamCancellation(t *testing.T) {
	for _, provider := range []string{"anthropic", "chat", "responses"} {
		t.Run(provider, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			prefix := strings.SplitN(admissionSSE(provider), "\n\n", 2)[0] + "\n\n"
			body := &admissionCancelBody{initial: strings.NewReader(prefix), ctx: ctx}
			var calls atomic.Int32
			model := admissionModel(provider, func(r *http.Request) (*http.Response, error) {
				calls.Add(1)
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: body, Request: r}, nil
			}, func(string, ...any) {})
			events, err := model.InvokeStream(ctx, admissionRequest(t, llm.CacheBestEffort))
			if err != nil {
				t.Fatal(err)
			}
			sawText := false
			deadline := time.NewTimer(2 * time.Second)
			defer deadline.Stop()
			for {
				select {
				case event, ok := <-events:
					if !ok {
						if !sawText || !body.closed.Load() || calls.Load() != 1 {
							t.Fatal("partial stream was not canceled/closed exactly once")
						}
						return
					}
					if e, ok := event.(llm.StreamTextDeltaEvent); ok && e.Delta == "ok" {
						sawText = true
						cancel()
					}
				case <-deadline.C:
					t.Fatal("partial stream did not stop after cancellation")
				}
			}
		})
	}
}

func BenchmarkCacheAdmission(b *testing.B) {
	request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent(strings.Repeat("x", 1024))}}}
	view, err := llm.NewCacheTargetView(request)
	if err != nil {
		b.Fatal(err)
	}
	request.CachePlan, err = view.Bind([]llm.CacheDirective{{Target: llm.CacheTarget{Kind: llm.CacheAfterMessage}, Policy: llm.CacheBestEffort}})
	if err != nil {
		b.Fatal(err)
	}
	model := &openai.ChatClient{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := llm.AdmitCachePlan(context.Background(), request, model, nil); err != nil {
			b.Fatal(err)
		}
	}
}
