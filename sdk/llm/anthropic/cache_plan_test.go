package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestToolCacheMappingGuardDoesNotPartiallyRewrite(t *testing.T) {
	for _, target := range []llm.CacheTarget{{Kind: llm.CacheAfterMessage}, {Kind: llm.CacheAfterToolDefinition, ToolIndex: -1}, {Kind: llm.CacheAfterToolDefinition, ToolIndex: 1}} {
		tools := []toolParam{{Name: "fixture", CacheCtrl: &cacheControl{Type: "ephemeral"}}}
		before, _ := json.Marshal(tools)
		var system any
		err := applyCachePlan(&llm.CachePlan{Directives: []llm.CacheDirective{{Target: target}}}, &system, nil, tools, nil, nil)
		after, _ := json.Marshal(tools)
		if err == nil || string(before) != string(after) {
			t.Fatal("invalid mapping changed payload", err)
		}
	}
	for _, plan := range []*llm.CachePlan{
		{Directives: make([]llm.CacheDirective, 5)},
		{Directives: []llm.CacheDirective{{Target: llm.CacheTarget{Kind: llm.CacheAfterToolDefinition}, TTL: llm.CacheTTL1Hour}}},
	} {
		tools := []toolParam{{CacheCtrl: &cacheControl{Type: "ephemeral"}}}
		var system any
		if err := applyCachePlan(plan, &system, nil, tools, nil, nil); err == nil || tools[0].CacheCtrl == nil {
			t.Fatal("limit/TTL guard failed")
		}
	}
}

func BenchmarkToolCacheRequestBuilder(b *testing.B) {
	for _, mode := range []string{"legacy", "explicit", "message", "block"} {
		b.Run(mode, func(b *testing.B) {
			client := &Client{ModelName: "fixture", MaxTokens: 64, MaxCachedToolDefinitions: 1}
			request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleSystem, Content: llm.TextContent(strings.Repeat("x", 1024)), Cache: true}, {Role: llm.RoleUser, Content: llm.TextContent("hello"), Cache: true}}, Tools: []llm.ToolDefinition{{Name: "first"}, {Name: "second"}}}
			if mode != "legacy" {
				view, err := llm.NewCacheTargetView(request)
				if err != nil {
					b.Fatal(err)
				}
				target := llm.CacheTarget{Kind: llm.CacheAfterToolDefinition}
				if mode == "message" {
					target = llm.CacheTarget{Kind: llm.CacheAfterMessage, MessageIndex: 1}
				}
				if mode == "block" {
					target = llm.CacheTarget{Kind: llm.CacheAfterMessageBlock, MessageIndex: 1}
				}
				request.CachePlan, err = view.Bind([]llm.CacheDirective{{Target: target, Policy: llm.CacheRequired, TTL: llm.CacheTTL5Minutes}})
				if err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := client.buildRequest(request, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestMessageMappingGuardDoesNotPartiallyWrapSystem(t *testing.T) {
	var system any = "original\n\ntext"
	plan := &llm.CachePlan{Directives: []llm.CacheDirective{
		{Target: llm.CacheTarget{Kind: llm.CacheAfterMessage}},
		{Target: llm.CacheTarget{Kind: llm.CacheAfterToolDefinition, ToolIndex: 99}},
	}}
	locations := []messageCacheLocation{{system: true, plain: true, eligible: true}}
	if err := applyCachePlan(plan, &system, nil, nil, locations, nil); err == nil {
		t.Fatal("invalid second destination accepted")
	}
	if system != "original\n\ntext" {
		t.Fatal("failed mapping partially rewrote system")
	}
}
