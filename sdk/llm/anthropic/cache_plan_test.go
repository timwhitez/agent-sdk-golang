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
		err := applyToolCachePlan(&llm.CachePlan{Directives: []llm.CacheDirective{{Target: target}}}, nil, nil, tools)
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
		if err := applyToolCachePlan(plan, nil, nil, tools); err == nil || tools[0].CacheCtrl == nil {
			t.Fatal("limit/TTL guard failed")
		}
	}
}

func BenchmarkToolCacheRequestBuilder(b *testing.B) {
	for _, mode := range []string{"legacy", "explicit"} {
		b.Run(mode, func(b *testing.B) {
			client := &Client{ModelName: "fixture", MaxTokens: 64, MaxCachedToolDefinitions: 1}
			request := llm.InvokeRequest{Messages: []llm.Message{{Role: llm.RoleSystem, Content: llm.TextContent(strings.Repeat("x", 1024)), Cache: true}, {Role: llm.RoleUser, Content: llm.TextContent("hello"), Cache: true}}, Tools: []llm.ToolDefinition{{Name: "first"}, {Name: "second"}}}
			if mode == "explicit" {
				view, err := llm.NewCacheTargetView(request)
				if err != nil {
					b.Fatal(err)
				}
				request.CachePlan, err = view.Bind([]llm.CacheDirective{{Target: llm.CacheTarget{Kind: llm.CacheAfterToolDefinition}, Policy: llm.CacheRequired, TTL: llm.CacheTTL5Minutes}})
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
