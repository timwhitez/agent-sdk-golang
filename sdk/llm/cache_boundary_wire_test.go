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
)

// Characterize existing wire boundaries, including quirks, before introducing
// normalized targets. A captured request is not proof of provider acceptance.
func TestAnthropicLegacyCacheBoundaryWireGolden(t *testing.T) {
	withState := func(content llm.Content) llm.Content {
		t.Helper()
		out, err := llm.WithProviderState(content, []llm.ProviderState{{
			Provider: "openai-responses", Kind: "response.output_item.v1",
			Data: json.RawMessage(`{"encrypted_content":"opaque-fixture-marker"}`),
		}})
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	tools := []llm.ToolDefinition{
		{Name: "first", Parameters: map[string]any{"type": "object", "title": "removed-title"}},
		{Name: "second", Parameters: map[string]any{"type": "object"}},
	}
	toolMessages := []llm.Message{
		{Role: llm.RoleAssistant, Cache: true, Content: withState(llm.Content{Text: "intro", Blocks: []llm.ContentBlock{{Type: "text", Text: "detail"}}}), ToolCalls: []llm.ToolCall{
			{ID: "call_a", Function: llm.FunctionCall{Name: "first", Arguments: `{"x":1}`}},
			{ID: "call_b", Function: llm.FunctionCall{Name: "second", Arguments: `{}`}},
		}},
		{Role: llm.RoleTool, ToolCallID: "call_a", Cache: true, IsError: true, Content: withState(llm.Content{Text: "result", Blocks: []llm.ContentBlock{{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://fixture.invalid/image.png"}}}})},
		{Role: llm.RoleTool, ToolCallID: "call_b", Content: withState(llm.Content{})},
	}
	assistant := `{"role":"assistant","content":[{"type":"text","text":"intro"},{"type":"text","text":"detail"},{"type":"tool_use","id":"call_a","name":"first","input":{"x":1}},{"type":"tool_use","id":"call_b","name":"second","input":{},"cache_control":{"type":"ephemeral"}}]}`
	resultA := `{"type":"tool_result","tool_use_id":"call_a","content":[{"type":"text","text":"result"},{"type":"image","source":{"type":"url","url":"https://fixture.invalid/image.png"}}],"is_error":true,"cache_control":{"type":"ephemeral"}}`
	resultB := `{"type":"tool_result","tool_use_id":"call_b","content":"(no output)"}`
	reordered := llm.CloneMessages(toolMessages)
	reordered[1], reordered[2] = reordered[2], reordered[1]
	for _, fixture := range []struct {
		name     string
		messages []llm.Message
		golden   string
	}{
		{
			name: "text-blocks-empty-and-opaque",
			messages: []llm.Message{
				{Role: llm.RoleSystem, Cache: true, Content: withState(llm.Content{Text: "system", Blocks: []llm.ContentBlock{{Type: "text", Text: "section"}}})},
				{Role: llm.RoleUser, Cache: true, Content: withState(llm.TextContent(" \t "))},
				{Role: llm.RoleUser, Cache: true, Content: withState(llm.Content{Text: "user", Blocks: []llm.ContentBlock{{Type: "text", Text: "tail"}}})},
				// Unlike empty Content.Text, an explicit empty text block is retained.
				{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "text"}}}},
			},
			golden: `{"system":[{"type":"text","text":"system","cache_control":{"type":"ephemeral"}},{"type":"text","text":"section","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"user"},{"type":"text","text":"tail","cache_control":{"type":"ephemeral"}}]},{"role":"user","content":[{"type":"text"}]}]}`,
		},
		{name: "assistant-final-tool-use-and-merged-results", messages: toolMessages,
			golden: `{"messages":[` + assistant + `,{"role":"user","content":[` + resultA + `,` + resultB + `]}]}`},
		{name: "reordered-results-retain-own-boundary", messages: reordered,
			golden: `{"messages":[` + assistant + `,{"role":"user","content":[` + resultB + `,` + resultA + `]}]}`},
	} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", fixture.name, stream), func(t *testing.T) {
				var baseline []byte
				for _, plan := range []*llm.CachePlan{nil, {Directives: []llm.CacheDirective{}}, {
					SchemaVersion: -1, RequestFingerprint: "private-plan-marker",
					Directives: []llm.CacheDirective{{Target: llm.CacheTarget{Kind: llm.CacheAfterMessageBlock, MessageIndex: -1}, Policy: llm.CacheRequired}},
				}} {
					request := llm.InvokeRequest{Messages: llm.CloneMessages(fixture.messages), Tools: tools, CachePlan: plan}
					before, err := llm.CloneInvokeRequest(request)
					if err != nil {
						t.Fatal(err)
					}
					var payload []byte
					calls := 0
					client := &anthropic.Client{ModelName: "fixture", MaxTokens: 64, MaxRetries: 1, MaxCachedToolDefinitions: 1, APIKey: "fixture-key", BaseURL: "https://fixture.invalid",
						HTTPClient: &http.Client{Transport: cacheWireTransport(func(r *http.Request) (*http.Response, error) {
							calls++
							var err error
							payload, err = io.ReadAll(r.Body)
							if err != nil {
								return nil, err
							}
							return &http.Response{StatusCode: 401, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":{"message":"fixture rejected"}}`)), Request: r}, nil
						})}}
					failures := 0
					if stream {
						events, err := client.InvokeStream(context.Background(), request)
						if err != nil {
							failures++
						} else {
							for event := range events {
								if _, ok := event.(llm.StreamErrorEvent); ok {
									failures++
								}
							}
						}
					} else if _, err := client.Invoke(context.Background(), request); err != nil {
						failures++
					}
					if calls != 1 || failures != 1 {
						t.Fatalf("requests/failures=%d/%d", calls, failures)
					}
					if !reflect.DeepEqual(request, before) {
						t.Fatal("serializer mutated request or plan")
					}
					for _, marker := range []string{"private-plan-marker", "opaque-fixture-marker", "fixture-key", "removed-title"} {
						if bytes.Contains(payload, []byte(marker)) {
							t.Fatal("non-wire fixture metadata leaked")
						}
					}
					if baseline != nil && !bytes.Equal(baseline, payload) {
						t.Fatal("inert plan changed wire payload")
					}
					baseline = payload
				}
				var got, want map[string]any
				if err := json.Unmarshal(baseline, &got); err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal([]byte(fixture.golden), &want); err != nil {
					t.Fatal(err)
				}
				want["model"], want["max_tokens"] = "fixture", float64(64)
				want["tool_choice"] = map[string]any{"type": "auto"}
				var goldenTools any
				if err := json.Unmarshal([]byte(`[{"name":"first","description":"","input_schema":{"type":"object"}},{"name":"second","description":"","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}]`), &goldenTools); err != nil {
					t.Fatal(err)
				}
				want["tools"] = goldenTools
				if stream {
					want["stream"] = true
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("wire boundary golden mismatch: got %s", baseline)
				}
			})
		}
	}
}
