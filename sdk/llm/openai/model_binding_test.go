package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type bindingWireTransport func(*http.Request) (*http.Response, error)

func (f bindingWireTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type bindingDynamicValue int

func (bindingDynamicValue) MarshalJSON() ([]byte, error) { panic("untrusted marshaler executed") }

func TestOpenAIFrameBindingOwnsConfigurationAndWire(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			name := "chat"
			if responses {
				name = "responses"
			}
			if stream {
				name += "/stream"
			}
			t.Run(name, func(t *testing.T) {
				temperature, topP, seed, maxTokens := 0.2, 0.7, 13, 123
				extra := map[string]any{"fixture": map[string]any{"list": []string{"old"}, "large": int64(9007199254740993)}}
				body := map[string]any{"nested": map[string]any{"old": true}, "raw": json.RawMessage(`{"x":1}`)}
				codes := map[int]struct{}{500: {}}
				var payloads, endpoints, keys []string
				httpClient := &http.Client{Transport: bindingWireTransport(func(r *http.Request) (*http.Response, error) {
					data, _ := io.ReadAll(r.Body)
					payloads = append(payloads, string(data))
					endpoints = append(endpoints, r.URL.String())
					keys = append(keys, r.Header.Get("Authorization"))
					return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
				})}
				chat := &ChatClient{HTTPClient: httpClient, BaseURL: "https://fixture.invalid", APIKey: "fixture-key", ProviderLabel: "fixture-label", ModelName: "old", Extra: extra, ExtraBody: body, Temperature: &temperature, TopP: &topP, Seed: &seed, MaxCompletionTokens: &maxTokens, RetryableStatusCodes: codes, MaxRetries: 1, UseLegacyMaxTokens: true, ParallelToolCalls: true, ServiceTier: "auto", ReasoningEffort: "low"}
				response := &ResponsesClient{HTTPClient: httpClient, BaseURL: "https://fixture.invalid", APIKey: "fixture-key", ProviderLabel: "fixture-label", ModelName: "old", Extra: extra, ExtraBody: body, Temperature: &temperature, TopP: &topP, Seed: &seed, MaxOutputTokens: &maxTokens, RetryableStatusCodes: codes, MaxRetries: 1, ForceStringInput: true, ServiceTier: "auto", ReasoningEffort: "low"}
				var source llm.ChatModel = chat
				if responses {
					source = response
				}
				invoke := func(model llm.ChatModel) {
					request := llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("fixture")}}
					if stream {
						ch, err := model.(llm.StreamingChatModel).InvokeStream(context.Background(), request)
						if err == nil {
							for range ch {
							}
						}
					} else {
						_, _ = model.Invoke(context.Background(), request)
					}
				}
				invoke(source)
				bound, known, err := llm.BindFrameModel(context.Background(), source)
				if err != nil || !known || !reflect.DeepEqual(source, bound) {
					t.Fatal("binding values changed", known, err)
				}
				temperature, topP, seed, maxTokens = 0.8, 0.9, 99, 999
				extra["fixture"].(map[string]any)["list"].([]string)[0] = "new"
				body["nested"].(map[string]any)["old"] = false
				body["raw"].(json.RawMessage)[5] = '2'
				delete(codes, 500)
				chat.ModelName, response.ModelName = "new", "new"
				chat.APIKey, response.APIKey = "new-key", "new-key"
				chat.BaseURL, response.BaseURL = "https://new.invalid", "https://new.invalid"
				invoke(bound)
				if len(payloads) != 2 || payloads[0] != payloads[1] || endpoints[0] != endpoints[1] || keys[0] != keys[1] {
					t.Fatal("bound wire changed after source mutation")
				}
				if !strings.Contains(payloads[1], `9007199254740993`) || bound.Model() != "old" || bound.Provider() != "fixture-label" {
					t.Fatal("configuration identity/precision changed")
				}
				if responses {
					c := bound.(*ResponsesClient)
					if c.HTTPClient != httpClient || len(c.RetryableStatusCodes) != 1 || *c.MaxOutputTokens != 123 {
						t.Fatal("response configuration alias")
					}
				} else {
					c := bound.(*ChatClient)
					if c.HTTPClient != httpClient || len(c.RetryableStatusCodes) != 1 || *c.MaxCompletionTokens != 123 {
						t.Fatal("chat configuration alias")
					}
				}
			})
		}
	}
}

func TestOpenAIFrameBindingNilEmptyAndFailure(t *testing.T) {
	for _, empty := range []bool{false, true} {
		var extra map[string]any
		var codes map[int]struct{}
		if empty {
			extra = map[string]any{}
			codes = map[int]struct{}{}
		}
		for _, model := range []llm.ChatModel{&ChatClient{Extra: extra, ExtraBody: extra, RetryableStatusCodes: codes}, &ResponsesClient{Extra: extra, ExtraBody: extra, RetryableStatusCodes: codes}} {
			bound, known, err := llm.BindFrameModel(context.Background(), model)
			if err != nil || !known || !reflect.DeepEqual(model, bound) {
				t.Fatal("nil/empty changed", known, err)
			}
		}
	}
	for _, value := range []any{make(chan int), bindingDynamicValue(1)} {
		for _, inBody := range []bool{false, true} {
			extra := map[string]any{"PRIVATE_SECRET_KEY": value}
			chat, responses := &ChatClient{}, &ResponsesClient{}
			if inBody {
				chat.ExtraBody, responses.ExtraBody = extra, extra
			} else {
				chat.Extra, responses.Extra = extra, extra
			}
			for _, model := range []llm.ChatModel{chat, responses} {
				bound, known, err := llm.BindFrameModel(context.Background(), model)
				if err == nil || known || bound != nil || strings.Contains(err.Error(), "PRIVATE") {
					t.Fatal("unsafe extras not rejected privately", known, err)
				}
			}
		}
	}
}

func TestOpenAIFrameBindingKeepsCompatibilityDowngrade(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("responses=%v/stream=%v", responses, stream), func(t *testing.T) {
				var payloads, warnings []string
				httpClient := &http.Client{Transport: bindingWireTransport(func(r *http.Request) (*http.Response, error) {
					data, _ := io.ReadAll(r.Body)
					payloads = append(payloads, string(data))
					status, body := 401, "fixture final rejection"
					if len(payloads)%2 == 1 {
						status, body = 400, "unknown field reasoning_effort extra_body; unsupported thinking; MissingParameter input.content"
					}
					return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
				})}
				var source llm.ChatModel = &ChatClient{HTTPClient: httpClient, BaseURL: "https://fixture.invalid", ModelName: "fixture", MaxRetries: 2, ReasoningEffort: "low", Extra: map[string]any{"thinking": true}, ExtraBody: map[string]any{"enable_thinking": true}}
				if responses {
					source = &ResponsesClient{HTTPClient: httpClient, BaseURL: "https://fixture.invalid", ModelName: "fixture", MaxRetries: 2, ReasoningEffort: "low", Extra: map[string]any{"thinking": true}, ExtraBody: map[string]any{"enable_thinking": true}}
				}
				ctx := llm.WithWarningSink(context.Background(), func(f string, args ...any) { warnings = append(warnings, fmt.Sprintf(f, args...)) })
				invoke := func(model llm.ChatModel) {
					request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "text", Text: "fixture"}}}}}}
					if stream {
						ch, err := model.(llm.StreamingChatModel).InvokeStream(ctx, request)
						if err == nil {
							for range ch {
							}
						}
					} else {
						_, _ = model.Invoke(ctx, request)
					}
				}
				invoke(source)
				originalWarnings := append([]string(nil), warnings...)
				warnings = nil
				bound, known, err := llm.BindFrameModel(ctx, source)
				if err != nil || !known {
					t.Fatal(known, err)
				}
				invoke(bound)
				if len(payloads) != 4 || payloads[0] == payloads[1] || payloads[0] != payloads[2] || payloads[1] != payloads[3] || !reflect.DeepEqual(warnings, originalWarnings) {
					t.Fatal("binding changed compatibility retry wire/diagnostics")
				}
				if !reflect.DeepEqual(source, bound) {
					t.Fatal("downgrade mutated bound configuration")
				}
			})
		}
	}
}

func BenchmarkOpenAIFrameBinding(b *testing.B) {
	client := &ChatClient{ModelName: "fixture", Extra: map[string]any{"nested": map[string]any{"enabled": true, "values": []string{"a", "b"}}}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, known, err := llm.BindFrameModel(context.Background(), client); err != nil || !known {
			b.Fatal(known, err)
		}
	}
}
