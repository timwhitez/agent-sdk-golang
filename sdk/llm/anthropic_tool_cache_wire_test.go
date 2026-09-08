package llm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
)

func toolCacheRequest() llm.InvokeRequest {
	request := llm.InvokeRequest{Messages: []llm.Message{
		{Role: llm.RoleSystem, Cache: true, Content: llm.Content{Text: "system", Blocks: []llm.ContentBlock{{Type: "text", Text: "section"}}}},
		{Role: llm.RoleUser, Cache: true, Content: llm.TextContent("hello")},
		{Role: llm.RoleAssistant, Cache: true, ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "tool_0", Arguments: `{}`}}, {ID: "b", Function: llm.FunctionCall{Name: "tool_1", Arguments: `{}`}}}},
		{Role: llm.RoleTool, ToolCallID: "a", Cache: true, Content: llm.TextContent("result_a")},
		{Role: llm.RoleTool, ToolCallID: "b", Cache: true, Content: llm.TextContent("result_b")},
	}}
	for i := 0; i < 6; i++ {
		request.Tools = append(request.Tools, llm.ToolDefinition{Name: fmt.Sprintf("tool_%d", i), Parameters: map[string]any{"type": "object", "title": "removed-title", "properties": map[string]any{"cache_control": map[string]any{"type": "string"}}}})
	}
	return request
}

func bindToolCache(t *testing.T, request *llm.InvokeRequest, directives []llm.CacheDirective) {
	t.Helper()
	view, err := llm.NewCacheTargetView(*request)
	if err != nil {
		t.Fatal(err)
	}
	request.CachePlan, err = view.Bind(directives)
	if err != nil {
		t.Fatal(err)
	}
}

func toolDirective(index int, policy llm.CacheDirectivePolicy, ttl llm.CacheTTL) llm.CacheDirective {
	return llm.CacheDirective{Target: llm.CacheTarget{Kind: llm.CacheAfterToolDefinition, ToolIndex: index}, Policy: policy, TTL: ttl}
}

// Only remove wire cache fields. A schema property named cache_control is data,
// not a breakpoint, and must remain untouched.
func clearWireCache(payload map[string]any) {
	if blocks, ok := payload["system"].([]any); ok {
		for _, b := range blocks {
			delete(b.(map[string]any), "cache_control")
		}
	}
	for _, m := range payload["messages"].([]any) {
		for _, b := range m.(map[string]any)["content"].([]any) {
			delete(b.(map[string]any), "cache_control")
		}
	}
	for _, tool := range payload["tools"].([]any) {
		delete(tool.(map[string]any), "cache_control")
	}
}

func TestAnthropicExactToolCacheWireAndLegacyPrecedence(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, status := range []int{200, 401} {
			t.Run(fmt.Sprintf("%v/%d", stream, status), func(t *testing.T) {
				var baseline map[string]any
				for _, planned := range []bool{false, true} {
					request := toolCacheRequest()
					if planned {
						bindToolCache(t, &request, []llm.CacheDirective{toolDirective(3, llm.CacheRequired, llm.CacheTTL5Minutes), toolDirective(0, llm.CacheBestEffort, llm.CacheTTLProviderDefault)})
					}
					before, err := llm.CloneInvokeRequest(request)
					if err != nil {
						t.Fatal(err)
					}
					var payload map[string]any
					var warnings []string
					var calls atomic.Int32
					model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
						calls.Add(1)
						data, err := io.ReadAll(r.Body)
						if err != nil {
							return nil, err
						}
						if err := json.Unmarshal(data, &payload); err != nil {
							return nil, err
						}
						body := admissionSuccess["anthropic"]
						if stream {
							body = admissionSSE("anthropic")
						}
						if status == 401 {
							body = `{"error":{"message":"fixture rejected"}}`
						}
						return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
					}, func(f string, a ...any) { warnings = append(warnings, fmt.Sprintf(f, a...)) })
					model.(*anthropic.Client).MaxCachedToolDefinitions = 2
					completion, err := callAdmissionModel(context.Background(), model, request, stream)
					if calls.Load() != 1 || (err != nil) != (status == 401) {
						t.Fatalf("calls=%d err=%v", calls.Load(), err)
					}
					if !reflect.DeepEqual(before, request) {
						t.Fatal("mapping mutated request/history/plan")
					}
					if !planned {
						baseline = payload
						continue
					}
					clearWireCache(baseline)
					tools := baseline["tools"].([]any)
					tools[0].(map[string]any)["cache_control"] = map[string]any{"type": "ephemeral"}
					tools[3].(map[string]any)["cache_control"] = map[string]any{"type": "ephemeral", "ttl": "5m"}
					if !reflect.DeepEqual(payload, baseline) {
						t.Fatalf("exact breakpoint/legacy precedence mismatch: %#v", payload)
					}
					want := []string{"cache_plan_accepted: directive 0: explicit plan takes precedence over legacy cache hints", "cache_plan_accepted: directive 1: explicit plan takes precedence over legacy cache hints"}
					if !reflect.DeepEqual(warnings, want) {
						t.Fatalf("diagnostic golden=%v", warnings)
					}
					if !stream && status == 200 && (len(completion.Diagnostics) != 2 || completion.Diagnostics[0].Kind != "cache_plan_accepted") {
						t.Fatal("missing typed accepted diagnostics")
					}
				}
			})
		}
	}
}

func TestAnthropicToolCacheFailuresBeforeNetwork(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, failure := range []string{"ttl", "overflow", "stale", "out-of-range", "duplicate"} {
			t.Run(fmt.Sprintf("%v/%s", stream, failure), func(t *testing.T) {
				request := toolCacheRequest()
				directives := []llm.CacheDirective{toolDirective(0, llm.CacheRequired, "")}
				reason, index := "unsupported_ttl", 0
				if failure == "ttl" {
					directives = append(directives, toolDirective(1, llm.CacheRequired, llm.CacheTTL1Hour))
					reason, index = "ttl_order_conflict", 1
				}
				if failure == "overflow" {
					for i := 1; i < 5; i++ {
						directives = append(directives, toolDirective(i, llm.CacheRequired, ""))
					}
					reason, index = "breakpoint_limit", 4
				}
				bindToolCache(t, &request, directives)
				switch failure {
				case "stale":
					request.Tools[0], request.Tools[1] = request.Tools[1], request.Tools[0]
					reason, index = "stale_request", -1
				case "out-of-range":
					request.CachePlan.Directives[0].Target.ToolIndex = 100
					reason = "invalid_target"
				case "duplicate":
					request.CachePlan.Directives = append(request.CachePlan.Directives, request.CachePlan.Directives[0])
					reason, index = "duplicate_boundary", 1
				}
				var calls atomic.Int32
				model := admissionModel("anthropic", func(*http.Request) (*http.Response, error) { calls.Add(1); return nil, fmt.Errorf("must not send") }, func(string, ...any) { t.Error("rejected plan emitted accepted/skip warning") })
				_, err := callAdmissionModel(context.Background(), model, request, stream)
				assertCacheViewError(t, err, reason, index)
				if calls.Load() != 0 {
					t.Fatal("invalid plan reached network")
				}
			})
		}
	}
}

func TestAnthropicToolCacheRequiredCapacityAndCallerIsolation(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			request := toolCacheRequest()
			var directives []llm.CacheDirective
			for i := 0; i < 4; i++ {
				directives = append(directives, toolDirective(i, llm.CacheBestEffort, ""))
			}
			directives = append(directives, toolDirective(5, llm.CacheRequired, ""), toolDirective(4, llm.CacheRequired, ""))
			bindToolCache(t, &request, directives)
			var got []int
			mutated := false
			model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
				var payload map[string]any
				data, _ := io.ReadAll(r.Body)
				if err := json.Unmarshal(data, &payload); err != nil {
					return nil, err
				}
				for i, v := range payload["tools"].([]any) {
					tool := v.(map[string]any)
					if tool["name"] != fmt.Sprintf("tool_%d", i) {
						t.Error("caller reorder changed owned wire")
					}
					if tool["cache_control"] != nil {
						got = append(got, i)
					}
				}
				return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
			}, func(string, ...any) {
				if !mutated {
					mutated = true
					request.Tools[0], request.Tools[5] = request.Tools[5], request.Tools[0]
				}
			})
			_, _ = callAdmissionModel(context.Background(), model, request, stream)
			if !reflect.DeepEqual(got, []int{0, 1, 4, 5}) {
				t.Fatalf("required capacity/selection=%v", got)
			}
		})
	}
}

func TestCacheAdmissionMixedSummaryDoesNotMislabelAccepted(t *testing.T) {
	request := llm.InvokeRequest{Tools: []llm.ToolDefinition{{Name: "private-tool"}}}
	var directives []llm.CacheDirective
	for i := 0; i < 32; i++ {
		request.Messages = append(request.Messages, llm.Message{Role: llm.RoleUser, Content: llm.TextContent("private-text")})
		directives = append(directives, llm.CacheDirective{Target: llm.CacheTarget{Kind: llm.CacheAfterMessageBlock, MessageIndex: i}, Policy: llm.CacheBestEffort, TTL: llm.CacheTTL1Hour})
	}
	directives = append(directives, toolDirective(0, llm.CacheRequired, ""))
	bindToolCache(t, &request, directives)
	owned, diagnostics, err := llm.AdmitCachePlan(context.Background(), request, &anthropic.Client{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(owned.CachePlan.Directives) != 1 || len(diagnostics) != 33 || diagnostics[32].Kind != "cache_plan_summary" || diagnostics[32].Message != "1 additional directives: 1 accepted, 0 skipped" {
		t.Fatal("mixed decision summary", diagnostics)
	}
	encoded, _ := json.Marshal(diagnostics)
	if strings.Contains(string(encoded), "private-") {
		t.Fatal("summary leaked source content")
	}
}

func TestAnthropicExplicitToolCacheSurvivesBetaRetry(t *testing.T) {
	for _, test := range []struct {
		stream bool
		ttl    llm.CacheTTL
		mixed  bool
	}{{false, llm.CacheTTL5Minutes, false}, {true, llm.CacheTTL5Minutes, false}, {false, llm.CacheTTL1Hour, false}, {true, llm.CacheTTL1Hour, false}, {false, llm.CacheTTL1Hour, true}, {true, llm.CacheTTL1Hour, true}} {
		stream := test.stream
		t.Run(fmt.Sprintf("%v/%s/mixed=%v", stream, test.ttl, test.mixed), func(t *testing.T) {
			request := toolCacheRequest()
			directives := []llm.CacheDirective{toolDirective(2, llm.CacheRequired, test.ttl)}
			if test.mixed {
				directives = append(directives, toolDirective(4, llm.CacheRequired, llm.CacheTTL5Minutes))
			}
			bindToolCache(t, &request, directives)
			before, err := llm.CloneInvokeRequest(request)
			if err != nil {
				t.Fatal(err)
			}
			var payloads []string
			var accepted atomic.Int32
			model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
				data, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				payloads = append(payloads, string(data))
				status, body := 200, admissionSuccess["anthropic"]
				if stream {
					body = admissionSSE("anthropic")
				}
				if len(payloads) == 1 {
					status, body = 400, `{"error":{"message":"unsupported beta header"}}`
				}
				return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
			}, func(format string, args ...any) {
				if strings.HasPrefix(fmt.Sprintf(format, args...), "cache_plan_accepted:") {
					accepted.Add(1)
				}
			})
			client := model.(*anthropic.Client)
			client.Beta = []string{"unsupported-beta"}
			_, err = callAdmissionModel(context.Background(), model, request, stream)
			if err != nil || len(payloads) != 2 || payloads[0] != payloads[1] || accepted.Load() != int32(len(directives)) {
				t.Fatalf("retry count=%d accepted=%d err=%v", len(payloads), accepted.Load(), err)
			}
			if test.ttl == llm.CacheTTL1Hour && !strings.Contains(payloads[0], `"ttl":"1h"`) {
				t.Fatal("long TTL fixture did not reach wire")
			}
			if !reflect.DeepEqual(request, before) || !reflect.DeepEqual(client.Beta, []string{"unsupported-beta"}) {
				t.Fatal("compat retry mutated source")
			}
		})
	}
}
