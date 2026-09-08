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

func blockDirective(message, ordinal int, policy llm.CacheDirectivePolicy) llm.CacheDirective {
	return llm.CacheDirective{Target: llm.CacheTarget{Kind: llm.CacheAfterMessageBlock, MessageIndex: message, BlockOrdinal: ordinal}, Policy: policy}
}

func blockCacheRequest(t *testing.T) llm.InvokeRequest {
	t.Helper()
	request := toolCacheRequest()
	request.Messages[0].Content = llm.Content{Text: "system-top", Blocks: []llm.ContentBlock{{Type: "text"}, {Type: "text", Text: "section"}}}
	request.Messages[1].Content = llm.Content{Text: "user-top", Blocks: []llm.ContentBlock{{Type: "unknown"}, {Type: "text"}, {Type: "text", Text: "part"}, {Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://fixture.invalid/image.png"}}, {Type: "document", Source: &llm.DocSrc{Data: "private-document", MediaType: "application/pdf"}}}}
	var err error
	request.Messages[1].Content, err = llm.WithProviderState(request.Messages[1].Content, []llm.ProviderState{{Provider: "fixture", Kind: "opaque", Data: json.RawMessage(`{"secret":"private-opaque"}`)}})
	if err != nil {
		t.Fatal(err)
	}
	blocks := request.Messages[1].Content.Blocks
	request.Messages[1].Content.Blocks = append([]llm.ContentBlock{blocks[len(blocks)-1]}, blocks[:len(blocks)-1]...)
	request.Messages[2].Content = llm.Content{Text: "assistant", Blocks: []llm.ContentBlock{{Type: "thinking", Thinking: "fixture-thinking", Signature: "fixture-signature"}}}
	return request
}

func TestAnthropicBlockCacheOrdinalWireGolden(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			var baseline map[string]any
			for _, planned := range []bool{false, true} {
				request := blockCacheRequest(t)
				if planned {
					bindToolCache(t, &request, []llm.CacheDirective{blockDirective(0, 1, llm.CacheRequired), blockDirective(1, 2, llm.CacheRequired), blockDirective(2, 1, llm.CacheRequired), blockDirective(4, 0, llm.CacheRequired)})
				}
				before, err := llm.CloneInvokeRequest(request)
				if err != nil {
					t.Fatal(err)
				}
				var payload map[string]any
				var calls atomic.Int32
				model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
					calls.Add(1)
					data, _ := io.ReadAll(r.Body)
					if strings.Contains(string(data), "private-opaque") {
						t.Error("opaque state leaked")
					}
					if err := json.Unmarshal(data, &payload); err != nil {
						return nil, err
					}
					return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
				}, func(string, ...any) {})
				_, _ = callAdmissionModel(context.Background(), model, request, stream)
				if calls.Load() != 1 || !reflect.DeepEqual(request, before) {
					t.Fatal("request not sent or source mutated")
				}
				if !planned {
					baseline = payload
					continue
				}
				clearWireCache(baseline)
				cache := map[string]any{"type": "ephemeral"}
				baseline["system"].([]any)[2].(map[string]any)["cache_control"] = cache
				messages := baseline["messages"].([]any)
				messages[0].(map[string]any)["content"].([]any)[4].(map[string]any)["cache_control"] = cache
				messages[1].(map[string]any)["content"].([]any)[2].(map[string]any)["cache_control"] = cache
				messages[2].(map[string]any)["content"].([]any)[1].(map[string]any)["cache_control"] = cache
				if !reflect.DeepEqual(payload, baseline) {
					t.Fatalf("logical ordinal/source/wire golden mismatch: %#v", payload)
				}
			}
		})
	}
}

func TestAnthropicBlockCacheEachToolUseAndResult(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, reorder := range []bool{false, true} {
			t.Run(fmt.Sprintf("%v/%v", stream, reorder), func(t *testing.T) {
				request := blockCacheRequest(t)
				if reorder {
					request.Messages[3], request.Messages[4] = request.Messages[4], request.Messages[3]
				}
				bindToolCache(t, &request, []llm.CacheDirective{blockDirective(2, 1, llm.CacheRequired), blockDirective(2, 2, llm.CacheRequired), blockDirective(3, 0, llm.CacheRequired), blockDirective(4, 0, llm.CacheRequired)})
				var calls atomic.Int32
				model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
					calls.Add(1)
					data, _ := io.ReadAll(r.Body)
					var payload map[string]any
					if err := json.Unmarshal(data, &payload); err != nil {
						return nil, err
					}
					messages := payload["messages"].([]any)
					assistant := messages[1].(map[string]any)["content"].([]any)
					for i, block := range assistant {
						want := i == 2 || i == 3
						if (block.(map[string]any)["cache_control"] != nil) != want {
							t.Errorf("assistant block %d cache mismatch", i)
						}
					}
					results := messages[2].(map[string]any)["content"].([]any)
					for i, block := range results {
						b := block.(map[string]any)
						if b["cache_control"] == nil || b["tool_use_id"] != request.Messages[3+i].ToolCallID {
							t.Errorf("result %d source/cache mismatch", i)
						}
					}
					return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
				}, func(string, ...any) {})
				_, _ = callAdmissionModel(context.Background(), model, request, stream)
				if calls.Load() != 1 {
					t.Fatal("mapping not sent")
				}
			})
		}
	}
}

func TestAnthropicBlockCacheRejectsPlaceholderAndInvalidOrdinal(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, failure := range []string{"document", "out-of-range", "duplicate-alias", "stale"} {
			t.Run(fmt.Sprintf("%v/%s", stream, failure), func(t *testing.T) {
				request := blockCacheRequest(t)
				bindToolCache(t, &request, []llm.CacheDirective{blockDirective(1, 1, llm.CacheRequired)})
				reason, index := "unmappable_target", 0
				switch failure {
				case "document":
					request.CachePlan.Directives[0].Target.BlockOrdinal = 3
				case "out-of-range":
					request.CachePlan.Directives[0].Target.BlockOrdinal = 99
					reason = "invalid_target"
				case "duplicate-alias":
					request.CachePlan.Directives = []llm.CacheDirective{blockDirective(4, 0, llm.CacheRequired), messageDirective(4, llm.CacheRequired)}
					reason, index = "duplicate_boundary", 1
				case "stale":
					request.Messages[1].Content.Blocks[3].Text = "changed"
					reason, index = "stale_request", -1
				}
				var calls atomic.Int32
				model := admissionModel("anthropic", func(*http.Request) (*http.Response, error) { calls.Add(1); return nil, fmt.Errorf("must not send") }, func(string, ...any) { t.Error("invalid target warned as accepted") })
				_, err := callAdmissionModel(context.Background(), model, request, stream)
				assertCacheViewError(t, err, reason, index)
				if calls.Load() != 0 {
					t.Fatal("invalid block target reached HTTP")
				}
			})
		}
	}
}

func TestAnthropicBlockCacheCollapsedSystem(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, ordinal := range []int{0, 1} {
			t.Run(fmt.Sprintf("%v/%d", stream, ordinal), func(t *testing.T) {
				request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleSystem, Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "text", Text: "one"}, {Type: "text", Text: "two"}}}}, {Role: llm.RoleUser, Content: llm.TextContent("hello")}}}
				bindToolCache(t, &request, []llm.CacheDirective{blockDirective(0, ordinal, llm.CacheRequired)})
				var calls atomic.Int32
				model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
					calls.Add(1)
					data, _ := io.ReadAll(r.Body)
					var payload map[string]any
					if err := json.Unmarshal(data, &payload); err != nil {
						return nil, err
					}
					want := []any{map[string]any{"type": "text", "text": "one\n\ntwo", "cache_control": map[string]any{"type": "ephemeral"}}}
					if !reflect.DeepEqual(payload["system"], want) {
						t.Error("collapsed string split or changed")
					}
					return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
				}, func(string, ...any) {})
				_, err := callAdmissionModel(context.Background(), model, request, stream)
				if ordinal == 0 {
					assertCacheViewError(t, err, "unmappable_target", 0)
					if calls.Load() != 0 {
						t.Fatal("earlier collapsed block sent")
					}
				} else if calls.Load() != 1 {
					t.Fatal("final collapsed block not mapped", err)
				}
			})
		}
	}
}

func TestCacheTargetsProjectionIsOwnedAndIdentical(t *testing.T) {
	request := blockCacheRequest(t)
	view, err := llm.NewCacheTargetView(request)
	if err != nil {
		t.Fatal(err)
	}
	targets := llm.CacheTargets(request)
	if !reflect.DeepEqual(targets, view.Targets()) {
		t.Fatal("projection drifted from binding")
	}
	targets[0].Source = "mutated"
	if reflect.DeepEqual(targets, llm.CacheTargets(request)) || view.Targets()[0].Source == "mutated" {
		t.Fatal("projection aliasing")
	}
	encoded, _ := json.Marshal(llm.CacheTargets(request))
	if strings.Contains(string(encoded), "private-") || strings.Contains(string(encoded), "fixture-thinking") {
		t.Fatal("projection leaked content")
	}
}

func TestAnthropicBlockCacheEachSourcePosition(t *testing.T) {
	for _, position := range []struct {
		message, ordinal int
		key              string
	}{
		{0, 0, "s:0"}, {0, 1, "s:2"}, {1, 0, "m:0:0"}, {1, 1, "m:0:3"}, {1, 2, "m:0:4"},
		{2, 0, "m:1:0"}, {2, 1, "m:1:2"}, {2, 2, "m:1:3"}, {3, 0, "m:2:0"}, {4, 0, "m:2:1"},
	} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%d/%d/%v", position.message, position.ordinal, stream), func(t *testing.T) {
				request := blockCacheRequest(t)
				bindToolCache(t, &request, []llm.CacheDirective{blockDirective(position.message, position.ordinal, llm.CacheRequired)})
				var got []string
				model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
					data, _ := io.ReadAll(r.Body)
					var payload map[string]any
					if err := json.Unmarshal(data, &payload); err != nil {
						return nil, err
					}
					for i, block := range payload["system"].([]any) {
						if block.(map[string]any)["cache_control"] != nil {
							got = append(got, fmt.Sprintf("s:%d", i))
						}
					}
					for i, message := range payload["messages"].([]any) {
						for j, block := range message.(map[string]any)["content"].([]any) {
							if block.(map[string]any)["cache_control"] != nil {
								got = append(got, fmt.Sprintf("m:%d:%d", i, j))
							}
						}
					}
					for i, tool := range payload["tools"].([]any) {
						if tool.(map[string]any)["cache_control"] != nil {
							got = append(got, fmt.Sprintf("t:%d", i))
						}
					}
					return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
				}, func(string, ...any) {})
				_, _ = callAdmissionModel(context.Background(), model, request, stream)
				if !reflect.DeepEqual(got, []string{position.key}) {
					t.Fatalf("cache positions=%v want%s", got, position.key)
				}
			})
		}
	}
}

func TestAnthropicBlockCachePlaceholderDoesNotConsumeCapacity(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.Content{Text: "top", Blocks: []llm.ContentBlock{{Type: "document", Source: &llm.DocSrc{Data: "private-document", MediaType: "application/pdf"}}, {Type: "text", Text: "a"}, {Type: "text", Text: "b"}, {Type: "text", Text: "c"}, {Type: "text", Text: "d"}}}}}}
			directives := []llm.CacheDirective{blockDirective(0, 1, llm.CacheBestEffort)}
			for ordinal := 2; ordinal <= 5; ordinal++ {
				directives = append(directives, blockDirective(0, ordinal, llm.CacheBestEffort))
			}
			directives[4].Policy = llm.CacheRequired
			bindToolCache(t, &request, directives)
			var got []int
			model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
				data, _ := io.ReadAll(r.Body)
				var payload map[string]any
				if err := json.Unmarshal(data, &payload); err != nil {
					return nil, err
				}
				for i, block := range payload["messages"].([]any)[0].(map[string]any)["content"].([]any) {
					if block.(map[string]any)["cache_control"] != nil {
						got = append(got, i)
					}
				}
				return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
			}, func(string, ...any) {})
			_, _ = callAdmissionModel(context.Background(), model, request, stream)
			if !reflect.DeepEqual(got, []int{2, 3, 4, 5}) {
				t.Fatalf("placeholder consumed a slot: %v", got)
			}
		})
	}
}
