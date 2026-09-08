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
)

func messageDirective(index int, policy llm.CacheDirectivePolicy) llm.CacheDirective {
	return llm.CacheDirective{Target: llm.CacheTarget{Kind: llm.CacheAfterMessage, MessageIndex: index}, Policy: policy}
}

func TestAnthropicMessageCacheJointWireGolden(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			var baseline map[string]any
			for _, planned := range []bool{false, true} {
				request := toolCacheRequest()
				if planned {
					bindToolCache(t, &request, []llm.CacheDirective{messageDirective(0, llm.CacheRequired), messageDirective(2, llm.CacheRequired), messageDirective(3, llm.CacheRequired), toolDirective(1, llm.CacheRequired, llm.CacheTTL5Minutes)})
				}
				before, err := llm.CloneInvokeRequest(request)
				if err != nil {
					t.Fatal(err)
				}
				var payload map[string]any
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
					return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
				}, func(string, ...any) {})
				if _, err := callAdmissionModel(context.Background(), model, request, stream); err != nil || calls.Load() != 1 {
					t.Fatal(err, calls.Load())
				}
				if !reflect.DeepEqual(request, before) {
					t.Fatal("message mapping mutated source")
				}
				if !planned {
					baseline = payload
					continue
				}
				clearWireCache(baseline)
				cache := map[string]any{"type": "ephemeral"}
				baseline["system"].([]any)[1].(map[string]any)["cache_control"] = cache
				messages := baseline["messages"].([]any)
				messages[1].(map[string]any)["content"].([]any)[1].(map[string]any)["cache_control"] = cache
				messages[2].(map[string]any)["content"].([]any)[0].(map[string]any)["cache_control"] = cache
				baseline["tools"].([]any)[1].(map[string]any)["cache_control"] = map[string]any{"type": "ephemeral", "ttl": "5m"}
				if !reflect.DeepEqual(payload, baseline) {
					t.Fatalf("source-to-wire joint golden mismatch: %#v", payload)
				}
			}
		})
	}
}

func TestAnthropicMessageCacheRejectsUnmappableBeforeCapacity(t *testing.T) {
	for _, block := range []llm.ContentBlock{{Type: "thinking", Thinking: "private-thinking"}, {Type: "redacted_thinking", Data: "private-redacted"}, {Type: "unknown", Text: "private-unknown"}, {Type: "text"}, {Type: "document", Source: &llm.DocSrc{Data: "private-data", MediaType: "application/pdf"}}, {Type: "image_url"}} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%v", block.Type, stream), func(t *testing.T) {
				request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.Content{Text: "visible", Blocks: []llm.ContentBlock{block}}}}}
				bindToolCache(t, &request, []llm.CacheDirective{messageDirective(0, llm.CacheRequired)})
				var calls atomic.Int32
				model := admissionModel("anthropic", func(*http.Request) (*http.Response, error) { calls.Add(1); return nil, fmt.Errorf("must not send") }, func(string, ...any) { t.Error("rejected mapping warned as accepted") })
				_, err := callAdmissionModel(context.Background(), model, request, stream)
				assertCacheViewError(t, err, "unmappable_target", 0)
				if calls.Load() != 0 {
					t.Fatal("unmappable boundary reached HTTP")
				}
			})
		}
	}
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("capacity/%v", stream), func(t *testing.T) {
			request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.Content{Text: "visible", Blocks: []llm.ContentBlock{{Type: "unknown"}}}}}}
			directives := []llm.CacheDirective{messageDirective(0, llm.CacheBestEffort)}
			for i := 1; i < 5; i++ {
				request.Messages = append(request.Messages, llm.Message{Role: llm.RoleUser, Content: llm.TextContent(fmt.Sprint(i))})
				directives = append(directives, messageDirective(i, llm.CacheBestEffort))
			}
			directives[4].Policy = llm.CacheRequired
			bindToolCache(t, &request, directives)
			var got []int
			model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
				var payload map[string]any
				data, _ := io.ReadAll(r.Body)
				if err := json.Unmarshal(data, &payload); err != nil {
					return nil, err
				}
				for i, m := range payload["messages"].([]any) {
					blocks := m.(map[string]any)["content"].([]any)
					if blocks[len(blocks)-1].(map[string]any)["cache_control"] != nil {
						got = append(got, i)
					}
				}
				return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
			}, func(string, ...any) {})
			_, _ = callAdmissionModel(context.Background(), model, request, stream)
			if !reflect.DeepEqual(got, []int{1, 2, 3, 4}) {
				t.Fatalf("unmappable target consumed capacity: %v", got)
			}
		})
	}
}

func TestAnthropicMessageCacheCollapsedSystemBoundary(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, early := range []bool{false, true} {
			t.Run(fmt.Sprintf("%v/early=%v", stream, early), func(t *testing.T) {
				request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleSystem, Content: llm.TextContent("first")}, {Role: llm.RoleSystem, Content: llm.Content{Text: "second", Blocks: []llm.ContentBlock{{Type: "text", Text: "third"}}}}, {Role: llm.RoleUser, Content: llm.TextContent("hello")}}}
				index := 1
				if early {
					index = 0
				}
				bindToolCache(t, &request, []llm.CacheDirective{messageDirective(index, llm.CacheRequired)})
				var calls atomic.Int32
				model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
					calls.Add(1)
					var payload map[string]any
					data, _ := io.ReadAll(r.Body)
					if err := json.Unmarshal(data, &payload); err != nil {
						return nil, err
					}
					want := []any{map[string]any{"type": "text", "text": "first\n\nsecond\n\nthird", "cache_control": map[string]any{"type": "ephemeral"}}}
					if !reflect.DeepEqual(payload["system"], want) {
						t.Error("collapsed system string was split or rewritten")
					}
					return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
				}, func(string, ...any) {})
				_, err := callAdmissionModel(context.Background(), model, request, stream)
				if early {
					assertCacheViewError(t, err, "unmappable_target", 0)
					if calls.Load() != 0 {
						t.Fatal("invented earlier collapsed boundary")
					}
				} else if calls.Load() != 1 {
					t.Fatal("whole collapsed boundary not sent", err)
				}
			})
		}
	}
}

func TestAnthropicMessageCacheOpaqueAndReorderedResults(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			request := toolCacheRequest()
			request.Messages[3], request.Messages[4] = request.Messages[4], request.Messages[3]
			var err error
			request.Messages[1].Content, err = llm.WithProviderState(request.Messages[1].Content, []llm.ProviderState{{Provider: "fixture", Kind: "opaque", Data: json.RawMessage(`{"secret":"private-opaque"}`)}})
			if err != nil {
				t.Fatal(err)
			}
			bindToolCache(t, &request, []llm.CacheDirective{messageDirective(1, llm.CacheRequired), messageDirective(3, llm.CacheRequired)})
			var calls atomic.Int32
			model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
				calls.Add(1)
				data, _ := io.ReadAll(r.Body)
				if strings.Contains(string(data), "private-opaque") {
					t.Error("opaque data entered wire")
				}
				var payload map[string]any
				if err := json.Unmarshal(data, &payload); err != nil {
					return nil, err
				}
				messages := payload["messages"].([]any)
				if messages[0].(map[string]any)["content"].([]any)[0].(map[string]any)["cache_control"] == nil {
					t.Error("opaque block shifted visible boundary")
				}
				results := messages[2].(map[string]any)["content"].([]any)
				if results[0].(map[string]any)["tool_use_id"] != "b" || results[0].(map[string]any)["cache_control"] == nil || results[1].(map[string]any)["cache_control"] != nil {
					t.Error("result reordering changed source association")
				}
				return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
			}, func(string, ...any) {})
			_, _ = callAdmissionModel(context.Background(), model, request, stream)
			if calls.Load() != 1 {
				t.Fatal("mapped request not sent")
			}
		})
	}
}

func TestAnthropicMessageCacheImageRoleEligibility(t *testing.T) {
	for _, role := range []llm.Role{llm.RoleSystem, llm.RoleUser, llm.RoleAssistant} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%v", role, stream), func(t *testing.T) {
				request := llm.InvokeRequest{Messages: []llm.Message{{Role: role, Content: llm.Content{Text: "visible", Blocks: []llm.ContentBlock{{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://fixture.invalid/image.png"}}}}}}}
				bindToolCache(t, &request, []llm.CacheDirective{messageDirective(0, llm.CacheRequired)})
				var calls atomic.Int32
				model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
					calls.Add(1)
					var payload map[string]any
					data, _ := io.ReadAll(r.Body)
					if err := json.Unmarshal(data, &payload); err != nil {
						return nil, err
					}
					blocks := payload["messages"].([]any)[0].(map[string]any)["content"].([]any)
					if blocks[1].(map[string]any)["type"] != "image" || blocks[1].(map[string]any)["cache_control"] == nil || blocks[0].(map[string]any)["cache_control"] != nil {
						t.Error("image endpoint mapping")
					}
					return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
				}, func(string, ...any) {})
				_, err := callAdmissionModel(context.Background(), model, request, stream)
				if role == llm.RoleUser {
					if calls.Load() != 1 {
						t.Fatal("user image boundary not mapped", err)
					}
				} else {
					assertCacheViewError(t, err, "unmappable_target", 0)
					if calls.Load() != 0 {
						t.Fatal("unsupported role image reached HTTP")
					}
				}
			})
		}
	}
}

func TestAnthropicMessageCacheEachMergedResult(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, reorder := range []bool{false, true} {
			for _, selected := range [][]int{{3}, {4}, {3, 4}} {
				t.Run(fmt.Sprintf("%v/reorder=%v/selected=%v", stream, reorder, selected), func(t *testing.T) {
					request := toolCacheRequest()
					if reorder {
						request.Messages[3], request.Messages[4] = request.Messages[4], request.Messages[3]
					}
					var directives []llm.CacheDirective
					for _, index := range selected {
						directives = append(directives, messageDirective(index, llm.CacheRequired))
					}
					bindToolCache(t, &request, directives)
					var calls atomic.Int32
					model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
						calls.Add(1)
						data, _ := io.ReadAll(r.Body)
						var payload map[string]any
						if err := json.Unmarshal(data, &payload); err != nil {
							return nil, err
						}
						results := payload["messages"].([]any)[2].(map[string]any)["content"].([]any)
						if len(results) != 2 {
							t.Error("tool result group changed")
							return nil, fmt.Errorf("fixture group mismatch")
						}
						for offset, raw := range results {
							block := raw.(map[string]any)
							wantCache := false
							for _, index := range selected {
								if index == 3+offset {
									wantCache = true
								}
							}
							if block["tool_use_id"] != request.Messages[3+offset].ToolCallID || (block["cache_control"] != nil) != wantCache {
								t.Errorf("source %d cache/id mismatch: %#v", 3+offset, block)
							}
						}
						return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
					}, func(string, ...any) {})
					_, _ = callAdmissionModel(context.Background(), model, request, stream)
					if calls.Load() != 1 {
						t.Fatal("mapped request not sent")
					}
				})
			}
		}
	}
}
