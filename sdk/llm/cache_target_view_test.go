package llm_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func cacheViewFixture(t *testing.T) llm.InvokeRequest {
	t.Helper()
	content, err := llm.WithProviderState(llm.Content{Text: "private-text", Blocks: []llm.ContentBlock{
		{Type: "thinking", Thinking: "private-thinking", Signature: "private-signature"},
		{Type: "text"}, {Type: "text", Text: "visible"},
		{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://fixture.invalid/image"}},
		{Type: "document", Source: &llm.DocSrc{Data: "private-document", MediaType: "application/pdf"}},
		{Type: "redacted_thinking", Data: "private-redacted"}, {Type: "unknown", Text: "private-unknown"},
		{Type: "image_url"}, {Type: "document"},
	}}, []llm.ProviderState{{Provider: "fixture", Kind: "opaque", Data: json.RawMessage(`{"secret":"private-opaque"}`)}})
	if err != nil {
		t.Fatal(err)
	}
	temperature := 0.5
	return llm.InvokeRequest{
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, Content: content, ToolCalls: []llm.ToolCall{{ID: "private-call", Function: llm.FunctionCall{Name: "private-tool", Arguments: `{}`}}}},
			{Role: llm.RoleTool, ToolCallID: "private-call", Content: llm.TextContent("private-result")},
			{Role: llm.RoleUser, Content: llm.TextContent(" \t ")},
			{Role: "unknown", Content: llm.TextContent("ignored")},
		},
		Tools:       []llm.ToolDefinition{{Name: "private-tool", Parameters: map[string]any{"type": "object", "nested": map[string]any{"value": "private-schema"}}}},
		Temperature: &temperature,
		Responses:   &llm.ResponsesOptions{Include: []string{"private-include"}, OutputSchema: map[string]any{"type": "object"}},
	}
}

func TestCacheTargetViewLogicalGoldenAndPrivacy(t *testing.T) {
	request := cacheViewFixture(t)
	view, err := llm.NewCacheTargetView(request)
	if err != nil {
		t.Fatal(err)
	}
	var rows []string
	for _, descriptor := range view.Targets() {
		target := descriptor.Target
		rows = append(rows, fmt.Sprintf("%s:%d:%d:%d:%s:%d", target.Kind, target.MessageIndex, target.BlockOrdinal, target.ToolIndex, descriptor.Source, descriptor.SourceIndex))
	}
	want := []string{
		"after_message_block:0:0:0:text:-1", "after_message_block:0:1:0:content_block:2",
		"after_message_block:0:2:0:content_block:3", "after_message_block:0:3:0:content_block:4",
		"after_message_block:0:4:0:tool_call:0", "after_message:0:0:0:tool_call:0",
		"after_message_block:1:0:0:tool_result:-1", "after_message:1:0:0:tool_result:-1",
		"after_tool_definition:0:0:0:tool_definition:0",
	}
	if !reflect.DeepEqual(rows, want) {
		t.Fatalf("logical target golden: %v", rows)
	}
	targets := view.Targets()
	targets[0].Source = "caller-change"
	if view.Targets()[0].Source != "text" {
		t.Fatal("Targets exposed owned slice")
	}
	for _, value := range []any{view.Targets(), view} {
		encoded, err := json.Marshal(value)
		if err != nil || strings.Contains(string(encoded), "private-") {
			t.Fatal("public JSON projection leaked request content")
		}
		if strings.Contains(fmt.Sprintf("%+v %#v", value, value), "private-") {
			t.Fatal("default formatting leaked request content")
		}
	}
	var nilView *llm.CacheTargetView
	if nilView.Targets() != nil || nilView.Validate(request, nil) != nil {
		t.Fatal("nil/no-plan compatibility")
	}
}

func TestCacheTargetViewValidationAndAliases(t *testing.T) {
	request := cacheViewFixture(t)
	view, err := llm.NewCacheTargetView(request)
	if err != nil {
		t.Fatal(err)
	}
	planFor := func(target llm.CacheTarget) *llm.CachePlan {
		return &llm.CachePlan{SchemaVersion: llm.CachePlanSchemaVersion, Directives: []llm.CacheDirective{{Target: target, Policy: llm.CacheRequired}}}
	}
	for _, descriptor := range view.Targets() {
		if err := view.Validate(request, planFor(descriptor.Target)); err != nil {
			t.Fatal(err)
		}
	}
	base := planFor(view.Targets()[0].Target)
	for _, test := range []struct {
		name, reason string
		index        int
		change       func(*llm.CachePlan)
	}{
		{"schema", "unsupported_schema", -1, func(p *llm.CachePlan) { p.SchemaVersion++ }},
		{"request-fingerprint", "unsupported_fingerprint", -1, func(p *llm.CachePlan) { p.RequestFingerprint = "private-fingerprint" }},
		{"object-fingerprint", "unsupported_fingerprint", 0, func(p *llm.CachePlan) { p.Directives[0].Target.ExpectedObjectFingerprint = "private-object" }},
		{"policy", "invalid_policy", 0, func(p *llm.CachePlan) { p.Directives[0].Policy = "private-policy" }},
		{"ttl", "invalid_ttl", 0, func(p *llm.CachePlan) { p.Directives[0].TTL = "private-ttl" }},
		{"kind", "invalid_target", 0, func(p *llm.CachePlan) { p.Directives[0].Target.Kind = "private-kind" }},
		{"negative-message", "invalid_target", 0, func(p *llm.CachePlan) { p.Directives[0].Target.MessageIndex = -1 }},
		{"missing-message", "invalid_target", 0, func(p *llm.CachePlan) { p.Directives[0].Target.MessageIndex = 99 }},
		{"missing-block", "invalid_target", 0, func(p *llm.CachePlan) { p.Directives[0].Target.BlockOrdinal = 99 }},
		{"empty-message", "invalid_target", 0, func(p *llm.CachePlan) {
			p.Directives[0].Target = llm.CacheTarget{Kind: llm.CacheAfterMessage, MessageIndex: 2}
		}},
		{"missing-tool", "invalid_target", 0, func(p *llm.CachePlan) {
			p.Directives[0].Target = llm.CacheTarget{Kind: llm.CacheAfterToolDefinition, ToolIndex: -1}
		}},
		{"duplicate", "duplicate_boundary", 1, func(p *llm.CachePlan) { p.Directives = append(p.Directives, p.Directives[0]) }},
		{"alias-conflict", "duplicate_boundary", 1, func(p *llm.CachePlan) {
			p.Directives[0].Target = llm.CacheTarget{Kind: llm.CacheAfterMessage, MessageIndex: 0}
			p.Directives = append(p.Directives, llm.CacheDirective{Target: llm.CacheTarget{Kind: llm.CacheAfterMessageBlock, MessageIndex: 0, BlockOrdinal: 4}, Policy: llm.CacheBestEffort, TTL: llm.CacheTTL1Hour})
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			plan := llm.CloneCachePlan(base)
			test.change(plan)
			before := llm.CloneCachePlan(plan)
			assertCacheViewError(t, view.Validate(request, plan), test.reason, test.index)
			if !reflect.DeepEqual(plan, before) {
				t.Fatal("validation mutated plan")
			}
		})
	}
	// Unselected index fields retain the existing CacheTarget contract.
	base.Directives[0].Target.ToolIndex = -99
	base.Directives[0].Policy = llm.CacheBestEffort
	base.Directives[0].TTL = llm.CacheTTL5Minutes
	if err := view.Validate(request, base); err != nil {
		t.Fatal(err)
	}
	assertCacheViewError(t, (&llm.CacheTargetView{}).Validate(request, base), "missing_view", -1)
}

func assertCacheViewError(t *testing.T, err error, reason string, index int) {
	t.Helper()
	var typed *llm.CachePlanValidationError
	if !errors.As(err, &typed) || typed.Reason != reason || typed.DirectiveIndex != index {
		t.Fatalf("want %s/%d, got %v", reason, index, err)
	}
	if strings.Contains(err.Error(), "private-") || errors.Unwrap(err) != nil {
		t.Fatal("diagnostic exposed caller content or wrapped error")
	}
}

func TestCacheTargetViewOwnedSnapshotAndStaleRequest(t *testing.T) {
	for _, mutate := range []struct {
		name string
		fn   func(*llm.InvokeRequest)
	}{
		{"text", func(r *llm.InvokeRequest) { r.Messages[0].Content.Text = "changed" }},
		{"block-pointer", func(r *llm.InvokeRequest) { r.Messages[0].Content.Blocks[3].ImageURL.URL = "changed" }},
		{"tool-schema", func(r *llm.InvokeRequest) { r.Tools[0].Parameters["nested"].(map[string]any)["value"] = "changed" }},
		{"call", func(r *llm.InvokeRequest) { r.Messages[0].ToolCalls[0].ID = "changed" }},
		{"result-reorder", func(r *llm.InvokeRequest) { r.Messages[0], r.Messages[1] = r.Messages[1], r.Messages[0] }},
		{"compaction", func(r *llm.InvokeRequest) { r.Messages = r.Messages[2:] }},
		{"temperature", func(r *llm.InvokeRequest) { *r.Temperature = 0.1 }},
		{"options", func(r *llm.InvokeRequest) { r.DisableThinking = true }},
		{"responses-slice", func(r *llm.InvokeRequest) { r.Responses.Include[0] = "changed" }},
		{"responses-schema", func(r *llm.InvokeRequest) { r.Responses.OutputSchema["type"] = "changed" }},
	} {
		t.Run(mutate.name, func(t *testing.T) {
			request := cacheViewFixture(t)
			original, err := llm.CloneInvokeRequest(request)
			if err != nil {
				t.Fatal(err)
			}
			view, err := llm.NewCacheTargetView(request)
			if err != nil {
				t.Fatal(err)
			}
			plan := &llm.CachePlan{SchemaVersion: llm.CachePlanSchemaVersion}
			request.CachePlan = &llm.CachePlan{RequestFingerprint: "not-request-identity"}
			if err := view.Validate(request, plan); err != nil {
				t.Fatal(err)
			}
			mutate.fn(&request)
			assertCacheViewError(t, view.Validate(request, plan), "stale_request", -1)
			if err := view.Validate(original, plan); err != nil {
				t.Fatal("source mutation changed snapshot", err)
			}
		})
	}
	request := cacheViewFixture(t)
	request.Tools[0].Parameters["private-secret"] = func() {}
	view, err := llm.NewCacheTargetView(request)
	assertCacheViewError(t, err, "uncloneable_request", -1)
	if view != nil {
		t.Fatal("failed construction returned a view")
	}
}

func TestCacheTargetViewConservativeEmptyIdentity(t *testing.T) {
	view, err := llm.NewCacheTargetView(llm.InvokeRequest{})
	if err != nil {
		t.Fatal(err)
	}
	plan := &llm.CachePlan{SchemaVersion: llm.CachePlanSchemaVersion}
	if err := view.Validate(llm.InvokeRequest{}, plan); err != nil {
		t.Fatal(err)
	}
	for _, request := range []llm.InvokeRequest{
		{Messages: []llm.Message{}}, {Tools: []llm.ToolDefinition{}}, {Responses: &llm.ResponsesOptions{}},
	} {
		assertCacheViewError(t, view.Validate(request, plan), "stale_request", -1)
	}
	// Per-result boundaries retain their identity even when legal result order changes.
	request := llm.InvokeRequest{Messages: []llm.Message{
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "a"}, {ID: "b"}}},
		{Role: llm.RoleTool, ToolCallID: "a"}, {Role: llm.RoleTool, ToolCallID: "b"},
	}, Tools: []llm.ToolDefinition{{Name: "a"}, {Name: "b"}}}
	view, err = llm.NewCacheTargetView(request)
	if err != nil {
		t.Fatal(err)
	}
	request.Messages[1], request.Messages[2] = request.Messages[2], request.Messages[1]
	assertCacheViewError(t, view.Validate(request, plan), "stale_request", -1)
	request.Messages[1], request.Messages[2] = request.Messages[2], request.Messages[1]
	request.Tools[0], request.Tools[1] = request.Tools[1], request.Tools[0]
	assertCacheViewError(t, view.Validate(request, plan), "stale_request", -1)
}

func TestCacheTargetViewConcurrentReads(t *testing.T) {
	request := cacheViewFixture(t)
	view, err := llm.NewCacheTargetView(request)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				plan := &llm.CachePlan{SchemaVersion: llm.CachePlanSchemaVersion, Directives: []llm.CacheDirective{{Target: view.Targets()[0].Target, Policy: llm.CacheRequired}}}
				if err := view.Validate(request, plan); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()
}

func BenchmarkCacheTargetView(b *testing.B) {
	for _, count := range []int{1, 32, 256} {
		b.Run(fmt.Sprint(count), func(b *testing.B) {
			request := llm.InvokeRequest{Messages: make([]llm.Message, count)}
			for i := range request.Messages {
				request.Messages[i] = llm.Message{Role: llm.RoleUser, Content: llm.TextContent(strings.Repeat("x", 1024))}
			}
			b.Run("snapshot", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if _, err := llm.NewCacheTargetView(request); err != nil {
						b.Fatal(err)
					}
				}
			})
			view, err := llm.NewCacheTargetView(request)
			if err != nil {
				b.Fatal(err)
			}
			plan := &llm.CachePlan{SchemaVersion: llm.CachePlanSchemaVersion, Directives: []llm.CacheDirective{{Target: llm.CacheTarget{Kind: llm.CacheAfterMessage, MessageIndex: count - 1}, Policy: llm.CacheRequired}}}
			b.Run("validate-last-boundary", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if err := view.Validate(request, plan); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

func ExampleCacheTargetView() {
	request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("hello")}}}
	view, err := llm.NewCacheTargetView(request)
	if err != nil {
		panic(err)
	}
	plan := &llm.CachePlan{SchemaVersion: llm.CachePlanSchemaVersion, Directives: []llm.CacheDirective{{
		Target: view.Targets()[0].Target, Policy: llm.CacheRequired,
	}}}
	// Retain the original view. A new view after editing would not detect staleness.
	fmt.Println(view.Validate(request, plan))
	request.Messages[0].Content.Text = "changed"
	fmt.Println(view.Validate(request, plan))
	// Output:
	// <nil>
	// cache plan: stale_request (directive -1)
}
