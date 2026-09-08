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

func ttlDirective(d llm.CacheDirective, ttl llm.CacheTTL) llm.CacheDirective { d.TTL = ttl; return d }

func TestAnthropicMixedTTLWireGolden(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			request := toolCacheRequest()
			// Deliberately not wire order: results, system, tools, assistant.
			bindToolCache(t, &request, []llm.CacheDirective{ttlDirective(messageDirective(4, llm.CacheRequired), llm.CacheTTL5Minutes), ttlDirective(messageDirective(0, llm.CacheRequired), llm.CacheTTL1Hour), toolDirective(1, llm.CacheRequired, llm.CacheTTL1Hour), messageDirective(2, llm.CacheRequired)})
			before, err := llm.CloneInvokeRequest(request)
			if err != nil {
				t.Fatal(err)
			}
			var got map[string]any
			var warnings []string
			var calls atomic.Int32
			model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
				calls.Add(1)
				data, _ := io.ReadAll(r.Body)
				if err := json.Unmarshal(data, &got); err != nil {
					return nil, err
				}
				body := admissionSuccess["anthropic"]
				if stream {
					body = admissionSSE("anthropic")
				}
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
			}, func(f string, a ...any) { warnings = append(warnings, fmt.Sprintf(f, a...)) })
			if _, err := callAdmissionModel(context.Background(), model, request, stream); err != nil || calls.Load() != 1 {
				t.Fatal(err, calls.Load())
			}
			long := map[string]any{"type": "ephemeral", "ttl": "1h"}
			if !reflect.DeepEqual(got["tools"].([]any)[1].(map[string]any)["cache_control"], long) || !reflect.DeepEqual(got["system"].([]any)[1].(map[string]any)["cache_control"], long) {
				t.Fatal("long TTL mapped to wrong prefix")
			}
			messages := got["messages"].([]any)
			if !reflect.DeepEqual(messages[1].(map[string]any)["content"].([]any)[1].(map[string]any)["cache_control"], map[string]any{"type": "ephemeral"}) {
				t.Fatal("default TTL changed")
			}
			if !reflect.DeepEqual(messages[2].(map[string]any)["content"].([]any)[1].(map[string]any)["cache_control"], map[string]any{"type": "ephemeral", "ttl": "5m"}) {
				t.Fatal("short TTL result mismatch")
			}
			if len(warnings) != 4 || !reflect.DeepEqual(request, before) {
				t.Fatal("diagnostics/source mutation")
			}
		})
	}
}

func TestAnthropicRequiredTTLConflictsUseWireOrder(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, kind := range []string{"tools", "hierarchy", "merged", "blocks"} {
			for _, reverse := range []bool{false, true} {
				t.Run(fmt.Sprintf("%v/%s/%v", stream, kind, reverse), func(t *testing.T) {
					request := toolCacheRequest()
					var directives []llm.CacheDirective
					switch kind {
					case "tools":
						directives = []llm.CacheDirective{toolDirective(0, llm.CacheRequired, ""), toolDirective(5, llm.CacheRequired, llm.CacheTTL1Hour)}
					case "hierarchy":
						directives = []llm.CacheDirective{toolDirective(0, llm.CacheRequired, ""), ttlDirective(messageDirective(0, llm.CacheRequired), llm.CacheTTL1Hour)}
					case "merged":
						request.Messages[3], request.Messages[4] = request.Messages[4], request.Messages[3]
						directives = []llm.CacheDirective{messageDirective(3, llm.CacheRequired), ttlDirective(messageDirective(4, llm.CacheRequired), llm.CacheTTL1Hour)}
					case "blocks":
						directives = []llm.CacheDirective{blockDirective(2, 0, llm.CacheRequired), ttlDirective(blockDirective(2, 1, llm.CacheRequired), llm.CacheTTL1Hour)}
					}
					if reverse {
						directives[0], directives[1] = directives[1], directives[0]
					}
					bindToolCache(t, &request, directives)
					var calls atomic.Int32
					model := admissionModel("anthropic", func(*http.Request) (*http.Response, error) { calls.Add(1); return nil, fmt.Errorf("must not send") }, func(string, ...any) { t.Error("rejected plan produced partial diagnostics") })
					_, err := callAdmissionModel(context.Background(), model, request, stream)
					assertCacheViewError(t, err, "ttl_order_conflict", 1)
					if calls.Load() != 0 {
						t.Fatal("TTL conflict reached network")
					}
				})
			}
		}
	}
}

func TestAnthropicBestEffortTTLConflictDoesNotConsumeCapacity(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			request := toolCacheRequest()
			bindToolCache(t, &request, []llm.CacheDirective{toolDirective(0, llm.CacheBestEffort, ""), toolDirective(4, llm.CacheRequired, llm.CacheTTL1Hour), toolDirective(1, llm.CacheBestEffort, llm.CacheTTL1Hour), toolDirective(2, llm.CacheBestEffort, llm.CacheTTL1Hour), toolDirective(5, llm.CacheBestEffort, ""), toolDirective(3, llm.CacheBestEffort, llm.CacheTTL1Hour)})
			var got []int
			var warnings []string
			model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
				data, _ := io.ReadAll(r.Body)
				var payload map[string]any
				if err := json.Unmarshal(data, &payload); err != nil {
					return nil, err
				}
				for i, t := range payload["tools"].([]any) {
					if t.(map[string]any)["cache_control"] != nil {
						got = append(got, i)
					}
				}
				return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
			}, func(f string, a ...any) { warnings = append(warnings, fmt.Sprintf(f, a...)) })
			_, _ = callAdmissionModel(context.Background(), model, request, stream)
			if !reflect.DeepEqual(got, []int{1, 2, 4, 5}) {
				t.Fatalf("TTL selection/capacity=%v", got)
			}
			if len(warnings) != 6 || !strings.Contains(warnings[0], "ttl_order_conflict") || !strings.Contains(warnings[5], "breakpoint_limit") {
				t.Fatal("decision reasons", warnings)
			}
		})
	}
}

func TestAnthropicCollapsedSystemLongTTL(t *testing.T) {
	request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleSystem, Content: llm.TextContent("one")}, {Role: llm.RoleSystem, Content: llm.TextContent("two")}, {Role: llm.RoleUser, Content: llm.TextContent("hello")}}}
	bindToolCache(t, &request, []llm.CacheDirective{messageDirective(2, llm.CacheRequired), ttlDirective(blockDirective(1, 0, llm.CacheRequired), llm.CacheTTL1Hour)})
	var calls atomic.Int32
	model := admissionModel("anthropic", func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		data, _ := io.ReadAll(r.Body)
		var payload map[string]any
		if err := json.Unmarshal(data, &payload); err != nil {
			return nil, err
		}
		want := []any{map[string]any{"type": "text", "text": "one\n\ntwo", "cache_control": map[string]any{"type": "ephemeral", "ttl": "1h"}}}
		if !reflect.DeepEqual(payload["system"], want) {
			t.Error("collapsed long TTL changed boundary")
		}
		return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture"}}`)), Request: r}, nil
	}, func(string, ...any) {})
	_, _ = model.Invoke(context.Background(), request)
	if calls.Load() != 1 {
		t.Fatal("valid collapsed TTL not sent")
	}
}

type ttlOrderModel struct {
	cacheDecisionModel
	hook func(llm.InvokeRequest, []llm.CacheTarget) []int
}

func (m *ttlOrderModel) PromptCacheTTLOrder(request llm.InvokeRequest, targets []llm.CacheTarget) []int {
	return m.hook(request, targets)
}

func TestCacheTTLOrderHookOwnershipAndFailures(t *testing.T) {
	request, view, plan := cacheDecisionFixture(t)
	plan.Directives[2].TTL = llm.CacheTTL1Hour
	before, _ := llm.CloneInvokeRequest(request)
	model := &ttlOrderModel{cacheDecisionModel: cacheDecisionModel{caps: llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 4, SupportedTTLs: []llm.CacheTTL{llm.CacheTTL1Hour}}}}
	positions := []int{2, 3, 1}
	model.hook = func(copy llm.InvokeRequest, targets []llm.CacheTarget) []int {
		copy.Messages[0].Content.Text = "private-change"
		targets[0].MessageIndex = 99
		return positions
	}
	result, err := view.Decide(request, plan, model)
	if err != nil {
		t.Fatal(err)
	}
	positions[0] = -1
	if !reflect.DeepEqual(request, before) || len(result.Plan.Directives) != 3 || result.Plan.Directives[0].Target.MessageIndex != 0 {
		t.Fatal("hook aliases escaped")
	}
	for _, bad := range []struct {
		positions []int
		reason    string
		index     int
	}{{[]int{1}, "invalid_capabilities", -1}, {[]int{-1, -1, -1}, "unmappable_target", 0}, {[]int{0, 1, 0}, "duplicate_boundary", 2}} {
		model.hook = func(llm.InvokeRequest, []llm.CacheTarget) []int { return bad.positions }
		plan.Directives[0].Policy = llm.CacheRequired
		result, err := view.Decide(request, plan, model)
		if result != nil || err == nil {
			t.Fatal("bad order returned partial plan")
		}
		assertCacheViewError(t, err, bad.reason, bad.index)
	}
}

func TestCacheTTLBestEffortKeepsInputPriority(t *testing.T) {
	request, view, plan := cacheDecisionFixture(t)
	for i := range plan.Directives {
		plan.Directives[i].Policy = llm.CacheBestEffort
	}
	plan.Directives[1].TTL = llm.CacheTTL1Hour
	model := &ttlOrderModel{cacheDecisionModel: cacheDecisionModel{caps: llm.PromptCacheCapabilities{ExplicitMessageBoundary: true, MaxBreakpoints: 2, SupportedTTLs: []llm.CacheTTL{llm.CacheTTL1Hour}}}}
	model.hook = func(llm.InvokeRequest, []llm.CacheTarget) []int { return []int{0, 1, 2} }
	result, err := view.Decide(request, plan, model)
	if err != nil {
		t.Fatal(err)
	}
	want := []llm.CacheDirectiveDecision{{DirectiveIndex: 0, Accepted: true, Reason: "accepted"}, {DirectiveIndex: 1, Reason: "ttl_order_conflict"}, {DirectiveIndex: 2, Accepted: true, Reason: "accepted"}}
	if !reflect.DeepEqual(result.Directives, want) || !reflect.DeepEqual(result.Plan.Directives, []llm.CacheDirective{plan.Directives[0], plan.Directives[2]}) {
		t.Fatalf("optional priority/capacity mismatch: %+v", result)
	}
}
