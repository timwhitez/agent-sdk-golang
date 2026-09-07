package llm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
)

type cacheWireTransport func(*http.Request) (*http.Response, error)

func (f cacheWireTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestCachePlanAttachmentPreservesLegacyWireGolden(t *testing.T) {
	for _, provider := range []string{"chat", "responses", "anthropic"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", provider, stream), func(t *testing.T) {
				var baseline []byte
				for _, plan := range []*llm.CachePlan{nil, {Directives: []llm.CacheDirective{}}, {
					SchemaVersion: -1, RequestFingerprint: "private-plan-marker",
					Directives: []llm.CacheDirective{{Target: llm.CacheTarget{Kind: llm.CacheAfterMessageBlock, MessageIndex: -1, ExpectedObjectFingerprint: "private-plan-marker"}, Policy: llm.CacheRequired, TTL: "unsupported"}},
				}} {
					calls := 0
					var payload []byte
					httpClient := &http.Client{Transport: cacheWireTransport(func(r *http.Request) (*http.Response, error) {
						calls++
						var err error
						payload, err = io.ReadAll(r.Body)
						if err != nil {
							return nil, err
						}
						return &http.Response{StatusCode: http.StatusUnauthorized, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture rejected"}}`)), Request: r}, nil
					})}
					var model llm.StreamingChatModel
					switch provider {
					case "chat":
						model = &openai.ChatClient{HTTPClient: httpClient, BaseURL: "https://fixture.invalid", ModelName: "fixture", MaxRetries: 1}
					case "responses":
						model = &openai.ResponsesClient{HTTPClient: httpClient, BaseURL: "https://fixture.invalid", ModelName: "fixture", MaxRetries: 1}
					case "anthropic":
						model = &anthropic.Client{HTTPClient: httpClient, BaseURL: "https://fixture.invalid", APIKey: "fixture-key", ModelName: "fixture", MaxTokens: 64, MaxRetries: 1, MaxCachedToolDefinitions: 1}
					}
					request := llm.InvokeRequest{
						Messages:  []llm.Message{{Role: llm.RoleSystem, Content: llm.TextContent("system"), Cache: true}, {Role: llm.RoleUser, Content: llm.TextContent("hello"), Cache: true}},
						Tools:     []llm.ToolDefinition{{Name: "work", Description: "fixture", Parameters: map[string]any{"type": "object"}}},
						CachePlan: plan,
					}
					before, err := json.Marshal(request.Messages)
					if err != nil {
						t.Fatal(err)
					}
					planBefore := llm.CloneCachePlan(plan)
					failures := 0
					if stream {
						events, err := model.InvokeStream(context.Background(), request)
						if err != nil {
							failures++
						} else {
							for event := range events {
								if _, ok := event.(llm.StreamErrorEvent); ok {
									failures++
								}
							}
						}
					} else if _, err := model.Invoke(context.Background(), request); err != nil {
						failures++
					}
					if calls != 1 || failures != 1 {
						t.Fatalf("requests/failures=%d/%d", calls, failures)
					}
					after, err := json.Marshal(request.Messages)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(before, after) || !reflect.DeepEqual(plan, planBefore) {
						t.Fatal("serializer mutated request history or plan")
					}
					if bytes.Contains(payload, []byte("private-plan-marker")) || bytes.Contains(payload, []byte("CachePlan")) {
						t.Fatal("plan leaked to Provider")
					}
					if baseline == nil {
						baseline = payload
					} else if !bytes.Equal(payload, baseline) {
						t.Fatal("inert plan changed legacy payload")
					}
				}
				// Fixed local wire golden: no remote endpoint is contacted.
				golden := cacheWireGoldens[fmt.Sprintf("%s/%v", provider, stream)]
				if string(baseline) != golden {
					t.Fatalf("wire golden mismatch: %s", baseline)
				}
			})
		}
	}
}

var cacheWireGoldens = map[string]string{
	"chat/false":      `{"messages":[{"content":"system","role":"system"},{"content":"hello","role":"user"}],"model":"fixture","tool_choice":"auto","tools":[{"function":{"description":"fixture","name":"work","parameters":{"type":"object"}},"type":"function"}]}`,
	"chat/true":       `{"messages":[{"content":"system","role":"system"},{"content":"hello","role":"user"}],"model":"fixture","stream":true,"stream_options":{"include_usage":true},"tool_choice":"auto","tools":[{"function":{"description":"fixture","name":"work","parameters":{"type":"object"}},"type":"function"}]}`,
	"responses/false": `{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}],"instructions":"system","model":"fixture","tools":[{"type":"function","name":"work","description":"fixture","parameters":{"type":"object"}}]}`,
	"responses/true":  `{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}],"instructions":"system","model":"fixture","stream":true,"tools":[{"type":"function","name":"work","description":"fixture","parameters":{"type":"object"}}]}`,
	"anthropic/false": `{"model":"fixture","max_tokens":64,"system":[{"type":"text","text":"system","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}],"tools":[{"name":"work","description":"fixture","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],"tool_choice":{"type":"auto"}}`,
	"anthropic/true":  `{"model":"fixture","max_tokens":64,"system":[{"type":"text","text":"system","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}],"tools":[{"name":"work","description":"fixture","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],"tool_choice":{"type":"auto"},"stream":true}`,
}
